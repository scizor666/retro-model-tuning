import json
import jsonlines
import glob
import os
import requests
import time

OLLAMA_API = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:12b" # Or "gemma3" depending on correct Ollama tag

SYSTEM_PROMPT = """You are a synthetic data generator for a retro gaming troubleshooting assistant.
I will provide you with a raw text chunk from a retro gaming wiki, forum, or documentation.
Your job is to read the text and extract 1-3 highly specific Question & Answer pairs based ONLY on the facts in the text.
The questions should sound like a user asking for help with an emulator, game, or setup.
The answers should be detailed, accurate, and helpful.

OUTPUT FORMAT:
Return ONLY a valid JSON array of objects with "instruction" and "response" keys. Do not include markdown code blocks.
Example:
[
  {
    "instruction": "Why does my audio crackle in SNES9x when playing Chrono Trigger?",
    "response": "Audio crackling in SNES9x is usually caused by the audio buffer size being too low..."
  }
]
"""

def generate_qa_pairs(chunk_text):
    prompt = f"Extract QA pairs from this text:\n\n{chunk_text}"
    
    payload = {
        "model": MODEL_NAME,
        "system": SYSTEM_PROMPT,
        "prompt": prompt,
        "stream": False,
        "format": "json" # Force JSON mode if supported
    }
    
    try:
        response = requests.post(OLLAMA_API, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        
        # The response should be a JSON array string
        raw_text = result.get('response', '[]').strip()
        
        # Strip markdown code blocks if the model ignored the formatting instruction
        if raw_text.startswith("```"):
            lines = raw_text.splitlines()
            if lines[0].startswith("```"): lines = lines[1:]
            if lines[-1].startswith("```"): lines = lines[:-1]
            raw_text = "\n".join(lines).strip()
            
        parsed = json.loads(raw_text)
        
        # If it returned a dict wrapper, find the list, or wrap literal dict
        if isinstance(parsed, dict):
            if 'instruction' in parsed and 'response' in parsed:
                parsed = [parsed]
            else:
                for k, v in parsed.items():
                    if isinstance(v, list):
                        parsed = v
                        break
        
        if not isinstance(parsed, list):
            print(f"Warning: Model returned non-list JSON: {raw_text[:100]}")
            return []
            
        return parsed
    except Exception as e:
        print(f"Error generating QA for chunk: {e}\nRaw output: {result.get('response', '')[:200] if 'result' in locals() else ''}")
        return []

if __name__ == "__main__":
    os.makedirs('data/synthetic', exist_ok=True)
    chunk_files = glob.glob('data/chunked/chunked_*.jsonl')
    
    output_file = 'data/synthetic/raw_qa_pairs.jsonl'
    
    print(f"Starting Q&A generation from {len(chunk_files)} chunk files using {MODEL_NAME}...")
    
    total_generated = 0
    with jsonlines.open(output_file, mode='w') as writer:
        for f in chunk_files:
            print(f"Processing: {f}")
            with jsonlines.open(f) as reader:
                for obj in reader:
                    chunk = obj.get('chunk', '')
                    source = obj.get('source', '')
                    url = obj.get('url', '')
                    
                    if not chunk: continue
                    
                    qa_pairs = generate_qa_pairs(chunk)
                    
                    for qa in qa_pairs:
                        # Validate the format
                        if 'instruction' in qa and 'response' in qa:
                            # Add provenence data
                            qa['source'] = source
                            qa['url'] = url
                            writer.write(qa)
                            total_generated += 1
                            print(f"  Generated: {qa.get('instruction', '')[:50]}... ({source})")
                            
                    time.sleep(0.5) # Let the GPU breathe
                    
    print(f"\nFinished! Generated {total_generated} QA pairs.")
