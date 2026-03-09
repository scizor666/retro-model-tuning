#!/bin/bash
set -e

# Activate the existing virtual environment in the repository
source venv/bin/activate

# Ensure our new dependencies are installed in the venv
pip install -q jsonlines mediapipe

mkdir -p data/eval_results

echo "Running Case 1: Bare Model"
python generator/evaluate.py --base_model unsloth/gemma-3-4b-it-bnb-4bit --output data/eval_results/eval_bare.json

echo -e "\nRunning Case 2: Bare Model + RAG"
python generator/evaluate.py --base_model unsloth/gemma-3-4b-it-bnb-4bit --use_rag --output data/eval_results/eval_bare_rag.json

echo -e "\nRunning Case 3: Fine-Tuned Model"
python generator/evaluate.py --base_model unsloth/gemma-3-4b-it-bnb-4bit --adapter loras/gemma-3-retro-assistant --output data/eval_results/eval_tuned.json

echo -e "\nRunning Case 4: Fine-Tuned Model + RAG"
python generator/evaluate.py --base_model unsloth/gemma-3-4b-it-bnb-4bit --adapter loras/gemma-3-retro-assistant --use_rag --output data/eval_results/eval_tuned_rag.json

echo -e "\nAll full benchmark evaluations complete."
