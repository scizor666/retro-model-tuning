import torch
from unsloth import FastLanguageModel
import os
import subprocess
import shutil

# CONFIGURATION
MODEL_NAME = "loras/gemma-3-retro-assistant"
EXPORT_DIR = "/mnt/ntfs/gemma3-retro-export"
LLAMA_CPP_PATH = "/mnt/ntfs/llama.cpp"
LLAMA_BIN_PATH = os.path.join(LLAMA_CPP_PATH, "build/bin")

# Add our patched llama.cpp to PATH so Unsloth uses it
os.environ["PATH"] = LLAMA_BIN_PATH + ":" + os.environ["PATH"]

print(f"Loading model and LoRA from {MODEL_NAME}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = 2048,
    load_in_4bit = True,
)

print("Exporting to GGUF (Q4_K_M) using Unsloth's built-in exporter...")
# save_pretrained_gguf handles merging and dequantization internally
model.save_pretrained_gguf(
    EXPORT_DIR,
    tokenizer,
    quantization_method = "q4_k_m",
)

print(f"Export finished. Checking {EXPORT_DIR}...")
gguf_file = None
for f in os.listdir(EXPORT_DIR):
    if f.endswith(".gguf"):
        gguf_file = os.path.join(EXPORT_DIR, f)
        break

if gguf_file:
    print(f"Found GGUF: {gguf_file}")
    
    # Create Modelfile
    modelfile_path = "/mnt/ntfs/Modelfile-retro-v4"
    with open(modelfile_path, "w") as f:
        f.write(f"FROM {gguf_file}\n")
        f.write('SYSTEM "You are a helpful assistant specializing in retro gaming. You provide accurate tips, history, and facts about classic games and consoles."\n')
        f.write('PARAMETER temperature 0.7\n')
        f.write('PARAMETER stop "<|file_separator|>"\n')
        f.write('PARAMETER stop "<|im_start|>"\n')
        f.write('PARAMETER stop "<|im_end|>"\n')

    print(f"Modelfile created at {modelfile_path}")
    
    print("Importing into Ollama...")
    subprocess.run(["ollama", "create", "retro-assistant", "-f", modelfile_path], check=True)
    print("Done!")
else:
    print("Error: No GGUF file found in export directory.")
