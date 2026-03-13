# Fine-Tuning & Export

Once the dataset is cleaned and formatted in ShareGPT form, it's time to teach our smaller model (`Gemma 3 4B`) how to replicate the expertise of the 12B model.

This gives us the best of both worlds: the intelligence of a massive model, with the inference speed and low VRAM requirement of a tiny model.

## Parameter-Efficient Fine-Tuning (PEFT)

Directly fine-tuning all 4 billion parameters of the base model would require terabytes of VRAM and months of compute on a cluster. 

Instead, our `training/finetune.py` script uses **QLoRA** (Quantized Low-Rank Adaptation).
1. The base 4B model is loaded into VRAM, but completely frozen and quantized to 4-bit precision.
2. A tiny "Adapter" matrix (LoRA) is attached to the model. Only this tiny adapter is trained. 
3. This reduces VRAM requirements from ~80GB down to ~4GB.

### Unsloth Magic
Our script relies completely on the `unsloth` library.
Unsloth provides heavily optimized GPU kernels for the attention mechanisms and cross-entropy loss calculations during training. Combining Unsloth with our 4070 Ti allowed us to fine-tune the model in minutes, rather than hours.

The core training hyperparameters:
- **Rank (`r`)**: 16. (Determines how "expressive" the new knowledge can be. 8-16 is ideal for Q&A tasks without risking catastrophic forgetting).
- **Target Modules**: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`. (We attach the LoRA adapter to almost all linear layers in the transformer for maximum adoption of the new tone and facts).

### Behavioral Persona Focus
Initial fine-tuning on raw facts resulted in high accuracy but "robotic" and inconsistent conversational flow. We've introduced a secondary training stage (via `training/finetune_persona.py`) that focuses exclusively on:
- **Tone & Style**: Responding as a friendly human expert rather than a search engine.
- **Out-of-Domain (OOD) Rejection**: Politely declining requests unrelated to retro gaming to prevent hallucination in areas where the RAG context is silent.
- **Natural Transitions**: Avoiding "robotic" phrasing like "According to the provided text..." while still utilizing the RAG context invisibly.

## Exporting for Inference (`training/merge_export.py`)

After training finishes, the output is just a folder of "Adapter Weights". It is useless on its own.

Our export script merges the newly trained Adapter Matrix directly back into the frozen base model mathematically, fusing them into a single, permanent model.

It then exports this fused model into the **GGUF** format (`Q4_K_M` quantized).
GGUF is the C++ engine format used by `llama.cpp` and **Ollama**. Ultimately, this single `gemma3-retro-4b-Q4_K_M.gguf` file is the culmination of the entire project: you can load it into Ollama and chat with your custom Retro Gaming Assistant.
