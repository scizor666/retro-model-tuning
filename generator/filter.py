import json
import jsonlines
import os
import random

def filter_and_format_dataset(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    valid_pairs = []
    
    with jsonlines.open(input_file) as reader:
        for obj in reader:
            instruction = obj.get('instruction', '').strip()
            response = obj.get('response', '').strip()
            
            # 1. Basic length filters
            if len(instruction) < 10 or len(response) < 15:
                continue
                
            # 2. Filter out hallucinated JSON anomalies or bad prompts
            bad_phrases = ['here are', 'here is a qa pair', 'extract qa pairs', 'json']
            if any(p in instruction.lower() for p in bad_phrases):
                continue
            if any(p in response.lower() for p in bad_phrases):
                continue
                
            # 3. Format as ShareGPT
            sharegpt_format = {
                "conversations": [
                    {"from": "human", "value": instruction},
                    {"from": "gpt", "value": response}
                ],
                "source": obj.get('source', 'unknown'),
                "url": obj.get('url', '')
            }
            valid_pairs.append(sharegpt_format)

    print(f"Loaded {len(valid_pairs)} valid QA pairs after filtering out bad generations.")
    
    # Shuffle the dataset for training
    random.shuffle(valid_pairs)
    
    # Save as standard JSON array as Unsloth expects this often
    with open(output_file, 'w') as f:
        json.dump(valid_pairs, f, indent=2)
        
    print(f"Saved formatted dataset to {output_file}")

if __name__ == "__main__":
    input_file = 'data/synthetic/raw_qa_pairs.jsonl'
    output_file = 'data/synthetic/sharegpt_dataset.json'
    
    filter_and_format_dataset(input_file, output_file)
