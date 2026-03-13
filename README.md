# Retro Model Tuning

This repository contains the data pipeline, synthetic data generation, and fine-tuning scripts for the **Gemma-3 Retro Assistant**.

## Project Structure
- `crawlers/`: Scrapy spiders for retro gaming sources.
- `docs/`: Extensive documentation on the training and export process.
- `generator/`: Scripts for synthetic Q&A generation using a larger teacher model.
- `training/`: Fine-tuning and model export (GGUF) implementation using Unsloth.
- `processing/`: Data cleanup and chunking utilities.
- `deploy/`: Android-ready RAG metadata and binary index exports.

## Core Capabilities
- **Behavioral Persona Tuning**: Specialized LoRA training to ensure natural, expert-toned responses instead of robotic "according to the text" phrasing.
- **RAG Export Pipeline**: Tooling to prepare high-precision vector indices (MediaPipe compatible) for on-device retrieval.
- **Multi-turn Evaluation**: A robust evaluation harness that mirrors the Android environment to ensure accuracy and SLA compliance.

## Setup
Refer to `setup_system.sh` and the documentation in `docs/` for environment setup and training details.
