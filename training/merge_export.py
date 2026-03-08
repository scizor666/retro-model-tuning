"""
Merge LoRA adapter into base model and export to GGUF for Ollama.

Uses plain transformers + peft (NOT Unsloth) to match how we trained.
Saves everything to /mnt/ntfs (687 GB free) to avoid filling sda1 (4.3 GB free).

Steps:
  1. Load base model in 4-bit (fits in 12GB VRAM) + merge LoRA adapter
  2. Cast merged model to float16 in-memory
  3. Save merged model in float16 to /mnt/ntfs/gemma3-retro-merged/
  4. Convert to GGUF with llama.cpp
  5. Import into Ollama
"""

import os
import sys
import subprocess
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from peft.utils import prepare_model_for_kbit_training

# Route HF cache and all output to the large NVMe drive
os.environ["HF_HOME"] = "/mnt/ntfs/hf_cache"
OUTPUT_DIR    = "/mnt/ntfs/gemma3-retro-merged"   # merged HF model
GGUF_F16_PATH = "/mnt/ntfs/gemma3-retro-f16.gguf"
GGUF_Q4_PATH  = "/mnt/ntfs/gemma3-retro-Q4_K_M.gguf"
LLAMA_CPP_DIR = "/mnt/ntfs/llama.cpp"

adapter_path = "loras/gemma-3-retro-assistant"
model_id     = "unsloth/gemma-3-4b-it-bnb-4bit"

# ── 1. Load base model in 4-bit ─────────────────────────────────────────────
# Use 4-bit so the 4B model fits comfortably in VRAM (12GB RTX 4070 Ti).
# We merge then cast to float16 before saving.
print(f"Loading base model {model_id} in 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# ── 2. Merge LoRA adapter ───────────────────────────────────────────────────
# Note: PEFT will dequantize NF4 weights to float16 during merge_and_unload().
# A rounding-error warning is expected but the result is usable.
print(f"Loading and merging LoRA adapter from {adapter_path}...")
model = PeftModel.from_pretrained(model, adapter_path)
print("Merging weights (this dequantizes NF4 → float16)...")
model = model.merge_and_unload()

# ── 4. Save merged model ────────────────────────────────────────────────────
# transformers 5.2.0 has a bug in save_pretrained() for models loaded through the
# new core_model_loading.py path — reverse_weight_conversion raises NotImplementedError.
# Workaround: save the state dict directly with safetensors, then copy config files.
print("Casting merged parameters to float16 in-place...")
for param in model.parameters():
    if param.data.dtype != torch.float16:
        param.data = param.data.to(torch.float16)

print(f"Saving merged model to {OUTPUT_DIR} ...")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save state dict directly via safetensors
import json
import shutil
from safetensors.torch import save_file as save_safetensors

# Filter out residual bitsandbytes/quantization metadata tensors
# These are not needed for the merged f16 model and confuse llama.cpp
quant_terms = [".quant_state", ".absmax", ".nested_absmax", ".quant_map", ".nested_quant_map"]
state_dict = {
    k: v for k, v in model.state_dict().items()
    if not any(term in k for term in quant_terms)
}
print(f"Filtered state_dict: {len(state_dict)} tensors remaining.")
# Shard to 4GB max to avoid memory issues
MAX_SHARD_SIZE = 4 * 1024**3
shards = {}
current_shard = {}
current_size = 0
shard_idx = 0
index_map = {}

for name, tensor in state_dict.items():
    tensor_size = tensor.numel() * tensor.element_size()
    if current_size + tensor_size > MAX_SHARD_SIZE and current_shard:
        filename = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
        shards[filename] = current_shard
        for n in current_shard:
            index_map[n] = filename
        shard_idx += 1
        current_shard = {}
        current_size = 0
    current_shard[name] = tensor.contiguous()
    current_size += tensor_size

if current_shard:
    filename = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
    shards[filename] = current_shard
    for n in current_shard:
        index_map[n] = filename
    shard_idx += 1

# Fix the filenames now that we know the total shard count
total_shards = shard_idx
renamed_shards = {}
for old_name, shard in shards.items():
    idx = int(old_name.split("-")[1])
    new_name = f"model-{idx+1:05d}-of-{total_shards:05d}.safetensors"
    renamed_shards[new_name] = shard
    for n in shard:
        index_map[n] = new_name

for filename, shard in renamed_shards.items():
    save_safetensors(shard, os.path.join(OUTPUT_DIR, filename))
    print(f"  Saved {filename}")

# Write the safetensors index
index = {"metadata": {"total_size": sum(t.numel() * t.element_size() for t in state_dict.values())}, "weight_map": index_map}
with open(os.path.join(OUTPUT_DIR, "model.safetensors.index.json"), "w") as f:
    json.dump(index, f, indent=2)

# Copy config files from HF cache
hf_cache = os.path.expanduser("~/.cache/huggingface/hub/models--unsloth--gemma-3-4b-it-bnb-4bit/snapshots/eb03c885bc2cc913fe792994bc766006f14ad72d")
for fname in ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
    src = os.path.join(hf_cache, fname)
    dst = os.path.join(OUTPUT_DIR, fname)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"  Copied {fname}")

# Also save tokenizer explicitly
tokenizer.save_pretrained(OUTPUT_DIR)

# Clean up config.json to remove quantization info (crucial for llama.cpp)
cfg_path = os.path.join(OUTPUT_DIR, "config.json")
if os.path.exists(cfg_path):
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    if "quantization_config" in cfg:
        del cfg["quantization_config"]
        # Also ensure torch_dtype is set correctly
        cfg["torch_dtype"] = "float16"
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print("  Cleaned quantization_config from config.json")

print("Merged model saved.")

# Free VRAM before running conversion
del model
torch.cuda.empty_cache()

# ── 5. Convert to GGUF (f16) ────────────────────────────────────────────────
convert_script = os.path.join(LLAMA_CPP_DIR, "convert_hf_to_gguf.py")
if not os.path.exists(convert_script):
    print(f"\nERROR: llama.cpp not found at {LLAMA_CPP_DIR}")
    sys.exit(1)

print(f"\nConverting to GGUF (f16) → {GGUF_F16_PATH} ...")
subprocess.run([
    sys.executable, convert_script,
    OUTPUT_DIR,
    "--outtype", "f16",
    "--outfile", GGUF_F16_PATH,
], check=True)

# ── 6. Quantize to Q4_K_M ───────────────────────────────────────────────────
llama_quantize = os.path.join(LLAMA_CPP_DIR, "build", "bin", "llama-quantize")
if not os.path.exists(llama_quantize):
    print(f"\nERROR: llama-quantize not found at {llama_quantize}")
    print("Build it with: cmake --build /mnt/ntfs/llama.cpp/build --target llama-quantize")
    sys.exit(1)

print(f"Quantizing to Q4_K_M → {GGUF_Q4_PATH} ...")
subprocess.run([llama_quantize, GGUF_F16_PATH, GGUF_Q4_PATH, "Q4_K_M"], check=True)
print(f"\nGGUF saved: {GGUF_Q4_PATH}")

# Clean up the large intermediate f16 gguf to save space
if os.path.exists(GGUF_F16_PATH):
    os.remove(GGUF_F16_PATH)
    print(f"Removed intermediate {GGUF_F16_PATH}")

# ── 7. Create Ollama Modelfile and import ───────────────────────────────────
modelfile_path = "/mnt/ntfs/Modelfile-retro-assistant"
with open(modelfile_path, "w") as f:
    f.write(f"""FROM {GGUF_Q4_PATH}

SYSTEM \"\"\"You are a knowledgeable retro gaming assistant specializing in classic video games, FAQs, walkthroughs, and tips for retro consoles like SNES, NES, Sega Genesis, Game Boy, and more. Answer questions concisely and accurately based on your training data.\"\"\"

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
""")
print(f"\nModelfile written to {modelfile_path}")

print("\nImporting into Ollama...")
result = subprocess.run(
    ["ollama", "create", "retro-assistant", "-f", modelfile_path],
    capture_output=False
)
if result.returncode == 0:
    print("\n✓ Model imported into Ollama successfully!")
    print("\nTest it with:")
    print("  ollama run retro-assistant 'What are good tips for beating the final boss in Chrono Trigger?'")
else:
    print(f"\nOllama import failed (exit code {result.returncode}).")
    print("You can import manually with:")
    print(f"  ollama create retro-assistant -f {modelfile_path}")
