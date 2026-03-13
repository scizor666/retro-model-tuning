"""
Fine-tuning Gemma-3 4B for retro game FAQ assistant.

Uses plain transformers + peft + bitsandbytes (NOT Unsloth) because Unsloth's
Gemma-3 multimodal patches have a Triton causal mask bug when training text-only.
"""

import os
import json
import torch
from dataclasses import dataclass
from typing import Any
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# CRITICAL: Use the large NVMe drive for HuggingFace caching.
os.environ["HF_HOME"] = "/mnt/ntfs/hf_cache"
os.makedirs("/mnt/ntfs/hf_cache", exist_ok=True)

# 1. Configuration
max_seq_length = 2048
model_id = "unsloth/gemma-3-4b-it-bnb-4bit"

# 2. Load tokenizer (note: this model uses the same tokenizer as google/gemma-3-4b-it)
print(f"Loading tokenizer for {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 3. Load model in 4-bit
print(f"Loading {model_id} in 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model = prepare_model_for_kbit_training(model)

# 4. Add LoRA adapters
print("Adding LoRA adapters...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.0,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 5. Pre-tokenize in a plain Python loop — no dataset.map(), no dill pickling.
# WHY: TRL's tokenize_fn closure captures the tokenizer which contains an unpicklable
# ConfigModuleInstance object. Tokenizing here avoids all serialization issues.
dataset_path = "data/synthetic/persona_dataset.json"
print("Loading synthetic dataset...")
if not os.path.exists(dataset_path):
    print(f"ERROR: Dataset not found at {dataset_path}.")
    exit()

with open(dataset_path, "r") as f:
    raw_data = json.load(f)

print(f"Tokenizing {len(raw_data)} examples...")
input_ids_list = []
labels_list = []

for item in raw_data:
    # Convert ShareGPT format to HF chat template format
    messages = []
    for turn in item.get("conversations", []):
        role = turn.get("from", turn.get("role", ""))
        content = turn.get("value", turn.get("content", ""))
        # Map roles to HF roles
        if role in ("human", "user"):
            messages.append({"role": "user", "content": content})
        elif role in ("gpt", "assistant"):
            messages.append({"role": "assistant", "content": content})
        elif role == "system":
            messages.append({"role": "system", "content": content})

    if len(messages) < 2:
        continue

    # Apply the model's chat template to the full conversation
    # To mask labels, we need to find the spans of assistant responses.
    # A simple but effective way: 
    # 1. Tokenize messages one by one to find boundaries.
    # 2. Build full tokens and full labels mask.
    
    full_ids = []
    full_labels = []
    
    # We always start with the system prompt (if present)
    # Note: labels = -100 for system/user, actual IDs for assistant.
    
    current_messages = []
    for turn in item.get("conversations", []):
        role = turn.get("from", turn.get("role", ""))
        content = turn.get("value", turn.get("content", ""))
        
        if role in ("human", "user"):
            current_messages.append({"role": "user", "content": content})
            res = tokenizer.apply_chat_template(current_messages, tokenize=True, add_generation_prompt=False)
            if hasattr(res, "input_ids"):
                tokens = res.input_ids
            elif isinstance(res, dict) and "input_ids" in res:
                tokens = res["input_ids"]
            else:
                tokens = res
            
            tokens = list(tokens) # Ensure plain list
            full_ids = tokens
            full_labels = [-100] * len(tokens)
            
        elif role in ("gpt", "assistant"):
            prev_len = len(full_ids)
            current_messages.append({"role": "assistant", "content": content})
            res = tokenizer.apply_chat_template(current_messages, tokenize=True, add_generation_prompt=False)
            if hasattr(res, "input_ids"):
                tokens = res.input_ids
            elif isinstance(res, dict) and "input_ids" in res:
                tokens = res["input_ids"]
            else:
                tokens = res
                
            tokens = list(tokens) # Ensure plain list
            full_ids = tokens
            new_labels = tokens.copy()
            for i in range(prev_len):
                new_labels[i] = -100
            full_labels = new_labels
            
        elif role == "system":
            current_messages.append({"role": "system", "content": content})
            res = tokenizer.apply_chat_template(current_messages, tokenize=True, add_generation_prompt=False)
            if hasattr(res, "input_ids"):
                tokens = res.input_ids
            elif isinstance(res, dict) and "input_ids" in res:
                tokens = res["input_ids"]
            else:
                tokens = res
                
            tokens = list(tokens) # Ensure plain list
            full_ids = tokens
            full_labels = [-100] * len(tokens)

    if len(full_ids) < 2:
        continue

    input_ids_list.append(full_ids)
    labels_list.append(full_labels)

dataset = Dataset.from_dict({
    "input_ids": input_ids_list,
    "labels": labels_list,
})
print(f"Dataset ready: {len(dataset)} examples.")


# 6. Data collator — pads input_ids and labels, builds attention_mask from scratch.
@dataclass
class CausalLMDataCollator:
    pad_token_id: int

    def __call__(self, examples: list[dict]) -> dict:
        max_len = max(len(ex["input_ids"]) for ex in examples)
        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []
        batch_token_type_ids = []

        for ex in examples:
            ids = ex["input_ids"]
            lbls = ex["labels"]
            pad_len = max_len - len(ids)

            batch_input_ids.append(ids + [self.pad_token_id] * pad_len)
            batch_labels.append(lbls + [-100] * pad_len)  # -100 masks padding in loss
            batch_attention_mask.append([1] * len(ids) + [0] * pad_len)
            # token_type_ids: 0 = text, 1 = image. All zeros for text-only training.
            # Required by Gemma-3's create_causal_mask_mapping during training.
            batch_token_type_ids.append([0] * max_len)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(batch_token_type_ids, dtype=torch.long),
        }


data_collator = CausalLMDataCollator(pad_token_id=tokenizer.pad_token_id)

# 7. Training
print("Setting up trainer...")
training_args = TrainingArguments(
    output_dir="outputs",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,   # effective batch size = 8
    warmup_steps=10,
    num_train_epochs=2,
    learning_rate=2e-5,
    bf16=True,
    logging_steps=5,
    optim="paged_adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=42,
    report_to="none",
    remove_unused_columns=False,     # We handle columns ourselves
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    data_collator=data_collator,
    args=SFTConfig(
        output_dir="outputs",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        num_train_epochs=2,
        learning_rate=2e-5,
        bf16=True,
        logging_steps=5,
        optim="paged_adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        report_to="none",
        remove_unused_columns=False,
        dataset_num_proc=1,
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=max_seq_length,
    ),
)

# 8. Train!
print("Starting Fine-tuning!")
trainer_stats = trainer.train()

# 9. Save LoRA adapter
print("Saving LoRA adapter to loras/gemma-3-persona-assistant...")
os.makedirs("loras/gemma-3-persona-assistant", exist_ok=True)
model.save_pretrained("loras/gemma-3-persona-assistant")
tokenizer.save_pretrained("loras/gemma-3-persona-assistant")

print("Training Complete! The adapter is saved.")
