import os
import json
import numpy as np
import struct
from tqdm import tqdm

def export_for_android():
    vecs_path = "data/rag_vectors.npy"
    meta_path = "data/rag_metadata.json"
    output_dir = "data/android_export"
    
    os.makedirs(output_dir, exist_ok=True)
    
    output_vec_bin = os.path.join(output_dir, "rag_index.bin")
    output_meta_bin = os.path.join(output_dir, "rag_metadata.bin")
    
    if not os.path.exists(vecs_path) or not os.path.exists(meta_path):
        print("Source RAG files not found. Run build_rag_db.py first.")
        return

    print("Loading and normalizing vectors...")
    vectors = np.load(vecs_path)
    num_rows, num_cols = vectors.shape
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors_normalized = (vectors / norms).astype(np.float32)

    print(f"Exporting vectors to {output_vec_bin}...")
    with open(output_vec_bin, "wb") as f:
        # Header: rows (int32), cols (int32)
        f.write(struct.pack("<II", num_rows, num_cols))
        f.write(vectors_normalized.tobytes())

    print("Loading and converting metadata to binary format...")
    with open(meta_path, "r") as f:
        metadata = json.load(f)

    if len(metadata) != num_rows:
        print(f"Warning: Metadata count ({len(metadata)}) mismatch with vectors ({num_rows})")

    content_bytes = []
    offsets = []
    current_offset = 0
    
    print("Packing metadata...")
    for item in tqdm(metadata):
        # We only need 'content' for the context injection. 
        # Title/Source can be included if needed, but 'content' is the primary one.
        text = item.get("content", "")
        # Optionally prepend title/source for better context
        # text = f"Source: {item.get('source','')}\nTitle: {item.get('title','')}\n{text}"
        
        encoded = text.encode("utf-8")
        length = len(encoded)
        offsets.append((current_offset, length))
        content_bytes.append(encoded)
        current_offset += length

    print(f"Exporting metadata to {output_meta_bin}...")
    with open(output_meta_bin, "wb") as f:
        # Header: num_rows (int32)
        f.write(struct.pack("<I", num_rows))
        # Body 1: Offset Table [offset(int32), length(int32)]
        for off, leng in offsets:
            f.write(struct.pack("<II", off, leng))
        # Body 2: Raw data
        for data in content_bytes:
            f.write(data)

    print("Done! Android assets ready in data/android_export/")
    print(f"Vector binary size: {os.path.getsize(output_vec_bin) / 1024 / 1024:.2f} MB")
    print(f"Metadata binary size: {os.path.getsize(output_meta_bin) / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    export_for_android()
