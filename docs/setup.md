# Environment Setup

Building a local fine-tuning environment can be notoriously difficult due to dependency mismatches between CUDA, PyTorch, and optimization libraries. Here is how and why our environment is configured the way it is.

## Core Choices

- **OS:** Ubuntu 24. Linux is the native environment for AI training. Windows (even with WSL) introduces VRAM overhead and compatibility quirks with low-level CUDA kernels.
- **GPU:** NVIDIA RTX 4070 Ti (12GB VRAM). High compute capability but limited memory, making memory-efficient techniques (like 4-bit quantization and Unsloth) mandatory.
- **Python:** 3.11.9 (managed via `pyenv`). While Ubuntu 24 defaults to Python 3.13, key libraries in the PyTorch ecosystem (like `torchaudio` and `torchao`) often lag in support for the newest Python releases. 3.11 gives us maximum compatibility.

## The Dependency Triangle: PyTorch, Unsloth, XFormers

Our training stack replies heavily on **Unsloth**, a library that rewrites HuggingFace modeling code with custom Triton kernels to train 2x faster and use 70% less VRAM.

However, Unsloth is extremely particular about its environment:
1. It requires `torch.int1` support from newer torch versions (`torch>2.4`).
2. It requires `xformers` to precisely match the PyTorch version it was compiled against.
3. It requires specific constraints on `transformers`.

### Our Resolution
Through trial and error, we landed on the bleeding-edge "Nightly" stack, because stable PyTorch 2.5.1 does not natively expose the integer types that `torchao` and Unsloth require for extreme quantization. 

We used the following install workflow:
```bash
# 1. Install PyTorch 2.6 Nightly with CUDA 12.1 or 12.4
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121

# 2. Install Unsloth Nightly directly from GitHub
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# 3. Install remaining ecosystem
pip install trl peft accelerate bitsandbytes torchao transformers
```
*(Note: we explicitly uninstalled `xformers` as Unsloth provides its own optimized attention kernels, avoiding compiling nightmares).*

## Disk Space Optimization
Language models are massive (10-30GB each). To prevent the root Ubuntu partition (`/dev/sda1`) from filling up during caching, we exported the HuggingFace cache directory to a larger NTFs/secondary drive at runtime:
```python
os.environ["HF_HOME"] = "/mnt/ntfs/hf_cache"
```
