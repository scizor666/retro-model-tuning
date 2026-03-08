import sys

def test_imports():
    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"PyTorch import failed: {e}")
        return False

    try:
        import unsloth
        print(f"Unsloth Version: {unsloth.__version__}")
    except ImportError as e:
        print(f"Unsloth import failed: {e}")
        return False
        
    try:
        import trl
        import peft
        import bitsandbytes
        print("TRL, PEFT, and bitsandbytes imported successfully.")
    except ImportError as e:
        print(f"Training library import failed: {e}")
        return False

    try:
        import scrapy
        import bs4
        from playwright.sync_api import sync_playwright
        import pandas as pd
        import jsonlines
        print("Data pipeline libraries imported successfully.")
    except ImportError as e:
        print(f"Data pipeline import failed: {e}")
        return False

    return True

if __name__ == "__main__":
    print("Testing environment...")
    success = test_imports()
    if success:
        print("\nSUCCESS: All dependencies are installed and functioning correctly.")
        sys.exit(0)
    else:
        print("\nFAILURE: One or more dependencies failed to load.")
        sys.exit(1)
