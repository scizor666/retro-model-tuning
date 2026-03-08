# Retro Model Tuning

This repository contains the data pipeline, synthetic data generation, and fine-tuning scripts for the **Gemma-3 Retro Assistant**.

## Project Structure
- `crawlers/`: Scrapy spiders for retro gaming sources.
- `docs/`: Extensive documentation on the training and export process.
- `generator/`: Scripts for synthetic Q&A generation using a larger teacher model.
- `training/`: Fine-tuning and model export (GGUF) implementation using Unsloth.
- `processing/`: Data cleanup and chunking utilities.

## Setup
Refer to `setup_system.sh` and the documentation in `docs/` for environment setup and training details.
