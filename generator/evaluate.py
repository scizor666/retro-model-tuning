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

def load_vector_db():
    vecs_path = "data/rag_vectors.npy"
    meta_path = "data/rag_metadata.json"
    
    if not os.path.exists(vecs_path) or not os.path.exists(meta_path):
        print("Vector database not found. Please run build_rag_db.py first.")
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
        
    # Embed query
    embed_result = embedder.embed(query)
    query_vector = np.array(embed_result.embeddings[0].embedding, dtype=np.float32)
    
    # Normalize query
    q_norm = np.linalg.norm(query_vector)
    if q_norm > 0:
        query_vector = query_vector / q_norm
        
    # Compute cosine similarity
    similarities = np.dot(vectors_normalized, query_vector)
    
    # Get top k indices
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # Extract context
    context_chunks = []
    for idx in top_indices:
        # Only use if similarity is high enough
        if similarities[idx] > 0.65:
            context_chunks.append(metadata[idx]["content"])
            
    if not context_chunks:
        return "No relevant information found."
        
    return "\n---\n".join(context_chunks)

def create_prompt(instruction, context=None):
    if context:
        return f"Context:\n{context}\n\nQuestion:\n{instruction}"
    return instruction

def llm_judge(instruction, expected, generated, category):
    if category == "positive":
        prompt = f"""You are an expert evaluator. Compare the Model Generated Response to the Expected Answer for the given Instruction.

Instruction: {instruction}
Expected Answer: {expected}
Model Generated Response: {generated}

Evaluate if the Model Generated Response contains the core facts presented in the Expected Answer. If the generated response provides completely different facts, hallucinated links, or misses the core premise of the Expected Answer, you must mark it INCORRECT.

Return ONLY the word CORRECT or INCORRECT."""
    elif category == "out_of_domain":
        prompt = f"""You are an expert evaluator.
        
Instruction: {instruction}
Model Generated Response: {generated}

This instruction is out-of-domain. The model should politely decline to answer, stating it is a retro gaming assistant. If it attempts to actually answer the instruction, mark it INCORRECT. If it declines, mark it CORRECT.

Return ONLY the word CORRECT or INCORRECT."""
    elif category == "factual_inconsistency":
        prompt = f"""You are an expert evaluator.
        
Instruction: {instruction}
Model Generated Response: {generated}

This instruction contains a factual inconsistency or asks about something not provided in context. The model should state it cannot answer or correct the premise. If it hallucinates an answer, mark it INCORRECT. If it handles it correctly by declining or correcting, mark it CORRECT.

Return ONLY the word CORRECT or INCORRECT."""

    try:
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": "gemma3:12b",
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0}
        })
        response.raise_for_status()
        result = response.json()["response"].strip().upper()
        if "INCORRECT" in result:
            return False
        return "CORRECT" in result
    except Exception as e:
        print(f"Error calling LLM judge: {e}")
        return False

def evaluate_model(base_model_path, adapter_path, dataset_path, output_path, use_rag=False, limit=None):
    print(f"Loading dataset from {dataset_path}...")
    dataset = []
    with jsonlines.open(dataset_path) as reader:
        for obj in reader:
            dataset.append(obj)
            
    if limit is not None:
        dataset = dataset[:limit]
        print(f"Limiting dataset to {limit} examples.")
        
    print("Forcing Ollama to unload any lingering models from VRAM before we load the base model...")
    try:
        requests.post("http://localhost:11434/api/generate", json={
            "model": "gemma3:12b",
            "keep_alive": 0
        })
    except Exception as e:
        pass
        
    import time
    time.sleep(5)
            
    print(f"Loading base model {base_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    if adapter_path:
        print(f"Loading LoRA adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
    
    embedder = None
    vectors = None
    metadata = None
    
    if use_rag:
        print("Initializing MediaPipe embedder for RAG...")
        base_options = python.BaseOptions(model_asset_path="models/text_embedder.tflite")
        options = text.TextEmbedderOptions(base_options=base_options)
        embedder = text.TextEmbedder.create_from_options(options)
        vectors, metadata = load_vector_db()
        if vectors is None:
            print("Failed to load vector DB. RAG will not work.")
            use_rag = False
    
    results = []
    print(f"Evaluating {len(dataset)} examples...")
    
    start_time = time.time()
    
    for i, item in enumerate(dataset):
        instruction = item["instruction"]
        expected = item["expected_response"]
        category = item["category"]
        
        context = None
        if use_rag:
            context = get_rag_context(instruction, embedder, vectors, metadata)
            
        prompt = create_prompt(instruction, context)
        
        # Format for gemma-3
        messages = [{"role": "user", "content": prompt}]
        text_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=128, pad_token_id=tokenizer.eos_token_id)
            
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        results.append({
            "instruction": instruction,
            "expected": expected,
            "generated": generated_text,
            "category": category,
            "use_rag": use_rag,
            "context_used": context if context else ""
        })
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(dataset)} examples...")
            
    end_time = time.time()
    print(f"Evaluation took {end_time - start_time:.2f} seconds.")
    
    # Calculate simple accuracy metrics
    correct_metrics = {"positive": 0, "out_of_domain": 0, "factual_inconsistency": 0}
    total_metrics = {"positive": 0, "out_of_domain": 0, "factual_inconsistency": 0}
    
    # Unload the model from VRAM to make room for Ollama Gemma-3 12B
    print("Unloading 4B model from VRAM...")
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    print("Running LLM Judge on generated responses...")
    for j, res in enumerate(results):
        gen = res["generated"]
        exp = res["expected"]
        cat = res["category"]
        inst = res["instruction"]
        
        total_metrics[cat] += 1
        
        is_correct = llm_judge(inst, exp, gen, cat)
        if is_correct:
            correct_metrics[cat] += 1
        res["judge_correct"] = is_correct

        if (j + 1) % 10 == 0:
            print(f"Judged {j + 1}/{len(results)} examples...")
                
    total_correct = sum(correct_metrics.values())
    total_samples = len(results)
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    
    print(f"\nResults for: {'Fine-Tuned' if adapter_path else 'Base'} + {'RAG' if use_rag else 'No RAG'}")
    print(f"Overall Accuracy: {accuracy:.2%} ({total_correct}/{total_samples})")
    for cat in correct_metrics.keys():
        if total_metrics[cat] > 0:
            cat_acc = correct_metrics[cat] / total_metrics[cat]
            print(f"  - {cat}: {cat_acc:.2%} ({correct_metrics[cat]}/{total_metrics[cat]})")
    
    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "base_model": base_model_path,
            "adapter": adapter_path,
            "use_rag": use_rag,
            "overall_accuracy": accuracy,
            "category_metrics": {cat: {"correct": correct_metrics[cat], "total": total_metrics[cat]} for cat in correct_metrics},
            "results": results
        }, f, indent=2)
        
    print(f"Saved detailed results to {output_path}")
    
    # Force Ollama to unload the 12B judge model
    print("Forcing Ollama to unload the judge model from VRAM...")
    try:
        requests.post("http://localhost:11434/api/generate", json={
            "model": "gemma3:12b",
            "keep_alive": 0
        })
    except Exception as e:
        print(f"Failed to unload Ollama model: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="unsloth/gemma-3-4b-it-bnb-4bit")
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--use_rag", action="store_true")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of examples to evaluate")
    args = parser.parse_args()
    
    dataset_path = "data/eval_dataset.jsonl"
    evaluate_model(args.base_model, args.adapter, dataset_path, args.output, args.use_rag, limit=args.limit)
