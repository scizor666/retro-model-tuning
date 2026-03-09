import json
import jsonlines
import random
import os

def generate_negative_prompts(count):
    topics = [
        "cooking", "finance", "medical", "automotive", 
        "modern programming", "home repair", "fitness", 
        "politics", "travel", "astronomy", "law",
        "quantum physics", "modern music", "gardening",
        "philosophy", "modern smartphones"
    ]
    
    templates = [
        "How do I {}?",
        "What is the best way to {}?",
        "Explain {} in simple terms.",
        "Can you write a guide on {}?",
        "What are the pros and cons of {}?",
        "Give me the top 5 tips for {}."
    ]
    
    specifics = {
        "cooking": ["bake a sourdough bread", "grill a medium rare steak", "make sushi from scratch", "prepare a vegan lasagna"],
        "finance": ["invest $10,000 in the stock market", "file taxes as a freelancer", "save for retirement", "understand cryptocurrency"],
        "medical": ["treat a common cold", "understand the symptoms of flu", "lower blood pressure naturally"],
        "automotive": ["change a car tire", "replace spark plugs on a 2018 Honda Civic", "jump start a car battery"],
        "modern programming": ["build a REST API in Node.js", "configure a Kubernetes cluster", "write a React hook", "setup CI/CD in GitHub Actions"],
        "home repair": ["fix a leaky faucet", "install drywall", "unclog a drain", "rewire a light switch"],
        "fitness": ["build a weekly workout routine", "lose 10 pounds in a month", "improve marathon time", "do a proper deadlift"]
    }

    # Generate combination of topics and templates
    prompts = []
    
    # Fill from specific templates first
    for topic, specific_list in specifics.items():
        for specific in specific_list:
            prompts.append(f"How do I {specific}?")
            prompts.append(f"What is the best way to {specific}?")
            prompts.append(f"Can you provide a guide to {specific}?")

    # Add generic ones to fill up to `count`
    while len(prompts) < count:
        topic = random.choice(topics)
        template = random.choice(templates)
        if topic in specifics:
            specific = random.choice(specifics[topic])
            prompts.append(template.format(specific))
        else:
            prompts.append(template.format(topic))
            
    # Deduplicate and return required count
    prompts = list(set(prompts))
    random.shuffle(prompts)
    return prompts[:count]

def create_eval_dataset():
    input_file = "data/synthetic/raw_qa_pairs.jsonl"
    output_file = "data/eval_dataset.jsonl"

    print(f"Reading from {input_file}...")
    positive_examples = []
    
    if os.path.exists(input_file):
        with jsonlines.open(input_file) as reader:
            for obj in reader:
                positive_examples.append(obj)
    
    # We will use 20% of the dataset as evaluation (or up to 150 examples)
    eval_size = min(len(positive_examples), 150)
    
    random.seed(42)  # For reproducibility
    random.shuffle(positive_examples)
    selected_positive = positive_examples[:eval_size]

    eval_data = []

    # Add positive examples
    for ex in selected_positive:
        eval_data.append({
            "category": "positive",
            "instruction": ex.get("instruction", ""),
            "expected_response": ex.get("response", ""),
            "source": ex.get("source", "")
        })

    # Generate an equal number of negative out-of-domain examples
    negative_prompts = generate_negative_prompts(eval_size)

    for np in negative_prompts:
        eval_data.append({
            "category": "out_of_domain",
            "instruction": np,
            "expected_response": "I am a retro gaming assistant and can only help with retro gaming, emulators, and related topics.",
            "source": "manual_negative"
        })

    # Add factual inconsistency examples (RAG negative)
    # Take another random subset of positives and give them a wrong expected output (to test RAG hallucination guard)
    factual_subset = positive_examples[eval_size:eval_size + 20]
    for ex in factual_subset:
        eval_data.append({
            "category": "factual_inconsistency",
            "instruction": ex.get("instruction", ""),
            "expected_response": "I cannot answer this based on the provided context.",
            "source": ex.get("source", "") + "_negative"
        })

    print(f"Generated {len(eval_data)} evaluation examples ({eval_size} positive, {len(negative_prompts)} out-of-domain, {len(factual_subset)} factual negative).")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with jsonlines.open(output_file, mode='w') as writer:
        writer.write_all(eval_data)
        
    print(f"Saved evaluation dataset to {output_file}")

if __name__ == "__main__":
    create_eval_dataset()
