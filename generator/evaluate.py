import os
import json
import jsonlines
import time
import argparse
import numpy as np
import requests
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import text
from peft import PeftModel
from PIL import Image

# Get absolute paths relative to the script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

def load_vector_db():
    vecs_path = os.path.join(PROJECT_ROOT, "data/rag_vectors.npy")
    meta_path = os.path.join(PROJECT_ROOT, "data/rag_metadata.json")
    
    if not os.path.exists(vecs_path) or not os.path.exists(meta_path):
        print("Vector database not found. Please run build_rag_db.py first.", flush=True)
        return None, None
        
    vectors = np.load(vecs_path)
    # Normalize vectors for cosine similarity
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1 # Avoid division by zero
    vectors_normalized = vectors / norms
    
    with open(meta_path, "r") as f:
        metadata = json.load(f)
        
    return vectors_normalized, metadata

def get_rag_context(query, embedder, vectors_normalized, metadata, top_k=2):
    if embedder is None or vectors_normalized is None:
        return None
        
    embed_result = embedder.embed(query)
    query_vector = np.array(embed_result.embeddings[0].embedding, dtype=np.float32)
    
    q_norm = np.linalg.norm(query_vector)
    if q_norm > 0:
        query_vector = query_vector / q_norm
        
    similarities = np.dot(vectors_normalized, query_vector)
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    context_chunks = []
    for idx in top_indices:
        content = metadata[idx]["content"]
        # Updated threshold to 0.78 for higher precision
        if similarities[idx] > 0.78 and len(content) > 100:
            context_chunks.append(content)
            
    if not context_chunks:
        return None
        
    return "\n---\n".join(context_chunks)

def llm_judge(conversation_history, instruction, expected, generated, category):
    # Flatten history for the judge
    history_text = "\n".join([f"User: {t['user']}\nAssistant: {t['assistant']}" for t in conversation_history])
    
    if category == "positive":
        prompt = f"""You are an expert evaluator assessing a multi-turn conversation.
Conversation History:
{history_text}

Current User Query: {instruction}
Expected Facts/Answer: {expected}
Model Answer: {generated}

TASK: Does the Model Answer contain the essential information and facts found in the Expected Answer in response to the Current User Query? 
Ignore minor differences in phrasing or conversational filler.
If the Model Answer is correct and helpful, return CORRECT.
If it is factually wrong, missing key info, or refuses to answer an in-domain question, return INCORRECT.
Return ONLY the word CORRECT or INCORRECT."""
    elif category == "out_of_domain":
        prompt = f"""You are an expert evaluator.
Conversation History:
{history_text}

Current User Query: {instruction}
Model Answer: {generated}

TASK: This query is out-of-domain (not about retro gaming). 
The model SHOULD politely decline or pivot to its role as a retro assistant.
If it refuses the off-topic query correctly, return CORRECT.
If it attempts to answer a non-retro query, return INCORRECT.
Return ONLY the word CORRECT or INCORRECT."""
    else:
        prompt = f"""You are an expert evaluator.
Model Answer: {generated}
Is this answer factually sound based on common retro gaming knowledge?
Return ONLY CORRECT or INCORRECT."""

    try:
        response = requests.post("http://127.0.0.1:11435/api/generate", json={
            "model": "gemma3:12b",
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0}
        })
        response.raise_for_status()
        result = response.json()["response"].strip().upper()
        is_correct = "CORRECT" in result and "INCORRECT" not in result
        return is_correct
    except Exception as e:
        print(f"Error calling LLM judge: {e}")
        return False

def evaluate_model(base_model_path, adapter_path, dataset_path, output_path, use_rag=False, limit=None, start=0):
    print(f"Loading dataset from {dataset_path}...", flush=True)
    dataset = []
    with jsonlines.open(dataset_path) as reader:
        for obj in reader:
            dataset.append(obj)
            
    if start > 0 or limit is not None:
        end = (start + limit) if limit is not None else len(dataset)
        dataset = dataset[start:end]
        print(f"Slicing dataset from index {start} to {end}.")
        
    print("Forcing Ollama to unload any lingering models from VRAM before we load the base model...")
    try:
        requests.post("http://localhost:11435/api/generate", json={"model": "gemma3:12b", "keep_alive": 0})
    except Exception:
        pass
        
    time.sleep(5)
            
    print(f"Loading base model {base_model_path}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    from transformers import AutoProcessor
    try:
        processor = AutoProcessor.from_pretrained(base_model_path)
    except:
        processor = tokenizer
        
    from transformers import BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        llm_int8_enable_fp32_cpu_offload=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
    )
    
    if adapter_path:
        print(f"Loading LoRA adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
    
    embedder = None
    vectors = None
    metadata = None
    
    if use_rag:
        print("Initializing MediaPipe embedder for RAG...", flush=True)
        base_options = python.BaseOptions(model_asset_path="models/text_embedder.tflite")
        options = text.TextEmbedderOptions(base_options=base_options)
        embedder = text.TextEmbedder.create_from_options(options)
        vectors, metadata = load_vector_db()
        if vectors is None:
            print("Failed to load vector DB. RAG will not work.")
            use_rag = False
    
    results = []
    total_latency_seconds = 0
    total_turns = 0
    
    print(f"Evaluating {len(dataset)} examples...", flush=True)
    
    for i, conversation in enumerate(dataset):
        category = conversation.get("category", "positive")
        turns = conversation["turns"]
        conversation_history = []
        conversation_results = []
        
        for turn_idx, turn in enumerate(turns):
            instruction = turn["instruction"]
            expected = turn.get("expected_response", "")
            image_path = turn.get("image_path")
            
            context = None
            if use_rag:
                context = get_rag_context(instruction, embedder, vectors, metadata)
                
            # Simulate Android's "Invisible RAG" Dynamic System Prompt
            base_sys = "You are a knowledgeable and friendly Retro Gaming Assistant. Your expertise is strictly limited to retro video games and consoles. If the user asks about ANY topic outside of retro gaming (e.g., baking, sports, modern politics), politely decline to answer. Answer naturally and directly like a human expert. Do not use phrases like 'According to the text'."
            if context:
                dynamic_sys = f"{base_sys}\n\nUse the following internal knowledge to help answer if relevant, otherwise rely on your own expertise.\n--- INTERNAL KNOWLEDGE START ---\n{context}\n--- INTERNAL KNOWLEDGE END ---"
            else:
                dynamic_sys = base_sys
                
            # Build messages list exactly as HF expects it
            # Android parity: History is clean. System prompt contains RAG. Current query is clean.
            messages = [{"role": "system", "content": dynamic_sys}]
            
            for h in conversation_history:
                messages.append({"role": "user", "content": h["user"]})
                messages.append({"role": "assistant", "content": h["assistant"]})
                
            if image_path and os.path.exists(image_path):
                image = Image.open(image_path).convert("RGB")
                messages.append({"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]})
            else:
                image = None
                messages.append({"role": "user", "content": instruction})
            
            # Tokenize & Generate
            turn_start_time = time.time()
            if image:
                text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=text_prompt, images=image, return_tensors="pt").to(model.device)
            else:
                text_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(text_prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=128, pad_token_id=tokenizer.eos_token_id)
                
            generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            turn_latency = time.time() - turn_start_time
            
            total_latency_seconds += turn_latency
            total_turns += 1
            
            # Update history for next iteration
            conversation_history.append({"user": instruction, "assistant": generated_text})
            
            conversation_results.append({
                "turn": turn_idx + 1,
                "instruction": instruction,
                "expected": expected,
                "generated": generated_text,
                "context_used": context if context else "",
                "latency_s": turn_latency
            })
            
        results.append({
            "category": category,
            "turns": conversation_results
        })
        print(f"Processed conversation {i + 1}/{len(dataset)}...", flush=True)

    average_latency = total_latency_seconds / total_turns if total_turns > 0 else 0
    print(f"Average Generation Latency: {average_latency:.3f}s per turn")

    # Unload model for Judge
    print("Unloading 4B model from VRAM...")
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    print("Running LLM Judge on generated responses...")
    correct_metrics = {"positive": 0, "out_of_domain": 0, "factual_inconsistency": 0}
    total_metrics = {"positive": 0, "out_of_domain": 0, "factual_inconsistency": 0}

    for j, res in enumerate(results):
        cat = res["category"]
        turns = res["turns"]
        total_metrics[cat] += 1
        
        # We judge the conversation as a whole, focusing on the final generation
        final_turn = turns[-1]
        is_correct = llm_judge(
            conversation_history=[{"user": t["instruction"], "assistant": t["generated"]} for t in turns[:-1]],
            instruction=final_turn["instruction"],
            expected=final_turn["expected"],
            generated=final_turn["generated"],
            category=cat
        )
        
        if is_correct:
            correct_metrics[cat] += 1
        res["judge_correct"] = is_correct

    total_correct = sum(correct_metrics.values())
    total_samples = len(results)
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    
    print(f"\nResults for: {'Fine-Tuned' if adapter_path else 'Base'} + {'RAG' if use_rag else 'No RAG'}")
    print(f"Overall Accuracy: {accuracy:.2%} ({total_correct}/{total_samples})")
    for cat in correct_metrics.keys():
        if total_metrics[cat] > 0:
            cat_acc = correct_metrics[cat] / total_metrics[cat]
            print(f"  - {cat}: {cat_acc:.2%} ({correct_metrics[cat]}/{total_metrics[cat]})")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "base_model": base_model_path,
            "adapter": adapter_path,
            "use_rag": use_rag,
            "sla_avg_latency_s": average_latency,
            "overall_accuracy": accuracy,
            "category_metrics": {cat: {"correct": correct_metrics[cat], "total": total_metrics[cat]} for cat in correct_metrics},
            "results": results
        }, f, indent=2)
        
    print(f"Saved detailed results to {output_path}")
    
    try:
        requests.post("http://localhost:11434/api/generate", json={"model": "gemma3:12b", "keep_alive": 0})
    except Exception as e:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="unsloth/gemma-3-4b-it-bnb-4bit")
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--use_rag", action="store_true")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="data/multi_turn_eval.jsonl", help="Path to jsonl dataset")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples to evaluate")
    parser.add_argument("--start", type=int, default=0, help="Starting index in dataset")
    args = parser.parse_args()
    
    evaluate_model(args.base_model, args.adapter, args.dataset, args.output, args.use_rag, limit=args.limit, start=args.start)
