"""
export_rag_for_android.py

Converts the existing numpy RAG index into two Android-friendly asset files:
  - rag_index.bin  : raw IEEE 754 float32 little-endian, preceded by two
                     uint32 header words: (num_rows, num_cols)
  - rag_texts.json : JSON array of {"content": ..., "title": ..., "source": ...}

Vectors are L2-normalised on export so that RagManager.kt can use a plain
dot-product for cosine similarity — identical to evaluate.py's approach.

Usage:
    cd /home/alex/projects/retro-model-tuning
    python generator/export_rag_for_android.py
"""

import json
import os
import struct

import numpy as np

VECTORS_PATH = "data/rag_vectors.npy"
METADATA_PATH = "data/rag_metadata.json"
OUTPUT_DIR = "data/android_export"
OUT_BIN = os.path.join(OUTPUT_DIR, "rag_index.bin")
OUT_JSON = os.path.join(OUTPUT_DIR, "rag_texts.json")


def main():
    print(f"Loading vectors from {VECTORS_PATH} ...")
    vectors = np.load(VECTORS_PATH).astype(np.float32)
    print(f"  Shape: {vectors.shape}")

    # L2-normalise – identical to evaluate.py load_vector_db()
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors_norm = vectors / norms
    print(f"  Normalised. Min norm: {np.linalg.norm(vectors_norm, axis=1).min():.4f}")

    print(f"Loading metadata from {METADATA_PATH} ...")
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    assert len(metadata) == vectors_norm.shape[0], (
        f"Row mismatch: {len(metadata)} metadata entries vs {vectors_norm.shape[0]} vectors"
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- Write rag_index.bin ----
    # Header: [num_rows (uint32 LE), num_cols (uint32 LE)]
    # Body:   float32 matrix, row-major, little-endian
    num_rows, num_cols = vectors_norm.shape
    print(f"Writing {OUT_BIN} ({num_rows} rows x {num_cols} cols) ...")
    with open(OUT_BIN, "wb") as f:
        f.write(struct.pack("<II", num_rows, num_cols))
        f.write(vectors_norm.astype("<f4").tobytes())  # little-endian float32
    bin_mb = os.path.getsize(OUT_BIN) / (1024 * 1024)
    print(f"  Wrote {bin_mb:.2f} MB")

    # ---- Write rag_texts.json ----
    # Only keep fields RagManager.kt actually needs; drop url to save space
    stripped = [
        {
            "content": m.get("content", ""),
            "title": m.get("title", ""),
            "source": m.get("source", ""),
        }
        for m in metadata
    ]
    print(f"Writing {OUT_JSON} ...")
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(stripped, f, ensure_ascii=False, separators=(",", ":"))
    json_kb = os.path.getsize(OUT_JSON) / 1024
    print(f"  Wrote {json_kb:.1f} KB")

    # ---- Sanity-check round-trip ----
    print("Sanity-checking binary round-trip ...")
    with open(OUT_BIN, "rb") as f:
        r, c = struct.unpack("<II", f.read(8))
        loaded = np.frombuffer(f.read(), dtype="<f4").reshape(r, c)
    assert loaded.shape == vectors_norm.shape, "Shape mismatch after reload!"
    max_diff = np.abs(loaded - vectors_norm).max()
    print(f"  Max float diff after reload: {max_diff:.2e}  (should be ~0)")

    print()
    print("Done! Copy these files into the Android assets folder:")
    print(f"  {os.path.abspath(OUT_BIN)}")
    print(f"  {os.path.abspath(OUT_JSON)}")
    print("  /home/alex/projects/retro-model-tuning/models/text_embedder.tflite")


if __name__ == "__main__":
    main()
