# Retro Game FAQ Assistant

This repository contains a complete, end-to-end pipeline for generating a fine-tuned language model designed to answer questions, troubleshoot, and provide setup guides for retro games and emulators. 

This project goes from raw, unstructured data on the public web all the way to a quantized, Ollama-ready inference model.

## Architecture & Workflow

The pipeline is broken down into four distinct phases, each serving a specific purpose:

1. **Data Crawling (`crawlers/`)**
2. **Text Processing (`processing/`)**
3. **Synthetic Data Generation (`generator/`)**
4. **Fine-Tuning & Export (`training/`)**

Here is how data flows through the system:
```text
[Emulation Wikis, Subreddits, Libretro Docs]
       │
       ▼ (Scrapy & Pushshift API)
[Raw JSONL Text Dumps]
       │
       ▼ (LangChain Text Splitters)
[Chunked Text Context Windows]
       │
       ▼ (Ollama API - Gemma 3 12B)
[Raw Synthetic Q&A JSON Arrays]
       │
       ▼ (Python Filter)
[ShareGPT Formatted Training Dataset]
       │
       ▼ (Unsloth + PyTorch Nightly)
[LoRA Adapter for Gemma 3 4B]
       │
       ▼ (Unsloth Base Merge)
[GGUF Model File] -> Loadable in Ollama
```

## Documentation Map

To understand the core components of the project in depth, please refer to the following guides:

- **[Environment Setup](setup.md)**: Why we chose specific versions of PyTorch and CUDA.
- **[Data Pipeline](data_pipeline.md)**: How we scraped wikis and Reddit without getting blocked.
- **[Synthetic Generation](generation.md)**: Using a large local model (12B) to teach a small model (4B).
- [**Model Training**](training.md): How Unsloth and QLoRA optimize VRAM usage and training time.
- [**GGUF Export & Ollama**](export_and_ollama.md): Patching llama.cpp for Gemma-3 and serving via Ollama.
