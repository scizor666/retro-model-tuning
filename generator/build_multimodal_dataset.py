import jsonlines
import os
import shutil

# Make dummy images for test
os.makedirs('data/images', exist_ok=True)
from PIL import Image
Image.new('RGB', (100, 100), color='red').save('data/images/snes.jpg')
Image.new('RGB', (100, 100), color='blue').save('data/images/n64.jpg')

data = [
    {
        "instruction": "Identify the console in the image.",
        "image_path": "data/images/snes.jpg",
        "expected_response": "That is a Super Nintendo Entertainment System.",
        "category": "vision"
    },
    {
        "instruction": "What console is this?",
        "image_path": "data/images/n64.jpg",
        "expected_response": "That is a Nintendo 64.",
        "category": "vision"
    }
]

with jsonlines.open('data/multimodal_eval.jsonl', 'w') as writer:
    writer.write_all(data)
print("Created multimodal eval dataset with 2 items.")
