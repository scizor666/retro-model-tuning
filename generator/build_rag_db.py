import os
import jsonlines
import sqlite3
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import text
import numpy as np

def init_db(db_path):
    import sqlite_vss
    # Connect to the database
    conn = sqlite3.connect(db_path)
    
    # Load VSS extension correctly
    conn.enable_load_extension(True)
    sqlite_vss.load(conn)
    conn.enable_load_extension(False)
    
    return conn

def chunk_text(text, max_words=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i:i + max_words]))
    return chunks

def build_vector_db():
    import glob
    input_files = glob.glob("data/**/*.jsonl", recursive=True)
    db_path = "data/rag_index.db"
    model_path = "models/text_embedder.tflite"

    if os.path.exists(db_path):
        os.remove(db_path)

    print("Initializing SQLite DB...")
    conn = sqlite3.connect(db_path)
    
    # For newer Python/sqlite3, sometimes enable_load_extension is missing if sqlite3 was compiled without it.
    # An alternative is we simply use an in-memory/standard ChromaDB or wait...
    # Let's check if the method exists. If not, we might need a workaround or a different vector DB like FAISS or just basic dot product in python for testing cases, then handle Android separately.
    
    try:
        import sqlite_vss
        conn.enable_load_extension(True)
        sqlite_vss.load(conn)
        conn.enable_load_extension(False)
    except (AttributeError, ImportError):
        print("ERROR: Your Python's sqlite3 module was compiled without enable_load_extension() or sqlite_vss is missing.")
        print("Falling back to a raw JSON dump for Python, and we will handle Android DB generation differently, or we use a pure-Python vector search for the eval script.")
        # Alternatively, we just save the embeddings to a numpy array for the offline evaluation, and deal with Android ObjectBox/SQLite later.
        fallback_to_numpy(input_files, model_path)
        return

    cursor = conn.cursor()
    # Create the metadata table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            rowid INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT,
            url TEXT,
            title TEXT,
            content TEXT
        )
    ''')

    print("Loading text embedder...")
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = text.TextEmbedderOptions(base_options=base_options)
    with text.TextEmbedder.create_from_options(options) as embedder:
        
        # Determine embedding dimension by running a dummy text
        dummy_result = embedder.embed("Test")
        embedding_dim = len(dummy_result.embeddings[0].embedding)
        print(f"Embedding dimension detected as: {embedding_dim}")
        
        # Now create the VSS table
        cursor.execute(f'''
            CREATE VIRTUAL TABLE vss_documents USING vss0(
                embedding({embedding_dim})
            )
        ''')
        
        print(f"Reading from {len(input_files)} input files...")
        docs_processed = 0
        chunks_indexed = 0
        
        bad_namespaces = ["Special:", "Category:", "Help:", "Talk:", "User:", "Template:", "File:", "MediaWiki:"]
        
        for input_file in input_files:
            if "eval_dataset" in input_file or "multimodal_dataset" in input_file:
                continue
            if os.path.exists(input_file):
                with jsonlines.open(input_file) as reader:
                    for obj in reader:
                        source = obj.get("source", "")
                        url = obj.get("url", "")
                        
                        if any(ns in url for ns in bad_namespaces):
                            continue
                            
                        title = obj.get("title", "")
                        content = obj.get("text", "")
                        
                        if not content and "instruction" in obj and "response" in obj:
                            content = f"Q: {obj['instruction']}\nA: {obj['response']}"
                            
                        if len(content) < 50:
                            continue
                            
                        chunks = chunk_text(content, max_words=150)
                        
                        for chunk in chunks:
                            if len(chunk) < 20: 
                                continue
                                
                            # Compute embedding
                            embed_result = embedder.embed(chunk)
                            embed_vector = embed_result.embeddings[0].embedding
                            
                            vector_json = f"[{','.join(str(f) for f in embed_vector)}]"
                            
                            # Insert metadata
                            cursor.execute('''
                                INSERT INTO documents (source, url, title, content)
                                VALUES (?, ?, ?, ?)
                            ''', (source, url, title, chunk))
                            
                            last_id = cursor.lastrowid
                            
                            # Insert vector
                            cursor.execute('''
                                INSERT INTO vss_documents (rowid, embedding)
                                VALUES (?, ?)
                            ''', (last_id, vector_json))
                            
                            chunks_indexed += 1
                            
                        docs_processed += 1
                        
                        if docs_processed % 50 == 0:
                            print(f"Indexed {docs_processed} documents ({chunks_indexed} chunks)...")
                            conn.commit()
                        
    conn.commit()
    conn.close()
    print(f"Finished indexing. Processed {docs_processed} documents into {chunks_indexed} chunks.")
    print(f"Vector Database saved to {db_path}")

def fallback_to_numpy(input_files, model_path):
    import numpy as np
    import json
    
    output_meta = "data/rag_metadata.json"
    output_vecs = "data/rag_vectors.npy"
    
    print("Using simple Numpy fallback for evaluation script embeddings...")
    metadata = []
    vectors = []
    
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = text.TextEmbedderOptions(base_options=base_options)
    
    docs_processed = 0
    chunks_indexed = 0
    
    bad_namespaces = ["Special:", "Category:", "Help:", "Talk:", "User:", "Template:", "File:", "MediaWiki:"]
    
    with text.TextEmbedder.create_from_options(options) as embedder:
        for input_file in input_files:
            if "eval_dataset" in input_file or "multimodal_dataset" in input_file:
                continue
            if os.path.exists(input_file):
                with jsonlines.open(input_file) as reader:
                    for obj in reader:
                        source = obj.get("source", "")
                        url = obj.get("url", "")
                        
                        if any(ns in url for ns in bad_namespaces):
                            continue
                            
                        title = obj.get("title", "")
                        content = obj.get("text", "")
                        
                        if not content and "instruction" in obj and "response" in obj:
                            content = f"Q: {obj['instruction']}\nA: {obj['response']}"
                            
                        if len(content) < 50:
                            continue
                            
                        chunks = chunk_text(content, max_words=150)
                        
                        for chunk in chunks:
                            if len(chunk) < 20: 
                                continue
                                
                            embed_result = embedder.embed(chunk)
                            embed_vector = embed_result.embeddings[0].embedding
                            
                            metadata.append({
                                "source": source,
                                "url": url,
                                "title": title,
                                "content": chunk
                            })
                            vectors.append(embed_vector)
                            chunks_indexed += 1
                            
                        docs_processed += 1
                        if docs_processed % 50 == 0:
                            print(f"Indexed {docs_processed} documents ({chunks_indexed} chunks) from {input_file}...")
                            
    np.save(output_vecs, np.array(vectors, dtype=np.float32))
    with open(output_meta, "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"Finished fallback indexing. Processed {docs_processed} documents into {chunks_indexed} chunks.")

if __name__ == "__main__":
    build_vector_db()
