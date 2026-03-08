# Synthetic Data Generation

For a language model to be safely fine-tuned for a specific task (like Answering Technical Questions), it needs data formatted exactly as Instruction/Response pairs.

Wikis and forum threads are raw text, not clean Q&A pairs.

To bridge this gap without spending months writing hundreds of Q&A pairs by hand, we employ **Synthetic Data Generation**. We use a massive, highly capable model locally (Google's `Gemma 3 12B`) to "read" our text chunks and roleplay as both the user and the expert.

## The Generator Script (`generator/synthesize_qa.py`)

1. **The Engine:** Ollama. We use Ollama because it runs quantized models locally with high Apple/NVIDIA optimization and exposes an easy-to-use API on `localhost:11434`.
2. **The Prompt:** We pass the raw scraped text chunk to Gemma 3 alongside a strict `SYSTEM_PROMPT`. You can see this system prompt in the script; it restricts the model to ONLY output facts found in the text, and to format them as JSON arrays.
3. **The Extraction:** Gemma 3 acts as the "Teacher". It figures out what the most interesting fact in the text is, invents a user Question asking about it, and writes the expertly formulated Answer.
4. **Resiliency:** Because LLMs occasionally hallucinate formatting (e.g., returning single objects instead of arrays, or wrapping everything in ````json` markdown blocks), the script has robust parsing logic to clean the output and extract the Python dictionary.

## The Filter (`generator/filter.py`)

LLMs aren't perfect. Sometimes `Gemma 3 12B` includes filler words ("Here are the QA pairs you requested:") or writes questions that are too short to be meaningful.

This script sanitizes the generator's raw output by:
1. Enforcing minimum character limits (Questions > 10 chars, Answers > 15 chars).
2. Deleting any objects containing banned filler phrases.
3. Shuffling the dataset randomly (critical for machine learning to prevent the model from memorizing sequential "batches" of questions on the same subject).
4. Restructuring the JSON into the **ShareGPT** format.

### The ShareGPT Format
Unsloth and HuggingFace training scripts standardize on the `ShareGPT` conversational format:
```json
{
  "conversations": [
    {"from": "human", "value": "How do I fix input lag in RetroArch?"},
    {"from": "gpt", "value": "To reduce input lag, turn on Run-Ahead..."}
  ],
  "source": "emuwiki"
}
```
This is what actually gets passed to the trainer.
