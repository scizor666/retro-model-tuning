import os
import re
import jsonlines
import glob
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_file(filepath, output_filepath, chunk_size=1000, chunk_overlap=200):
    """
    Takes a raw JSONL file containing 'title' and 'text', chunks the text 
    using Langchain's RecursiveCharacterTextSplitter, and saves the chunks 
    to a new JSONL file.
    """
    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks_created = 0
    with jsonlines.open(filepath) as reader, jsonlines.open(output_filepath, mode='w') as writer:
        for obj in reader:
            title = obj.get('title', 'Unknown Title')
            text = obj.get('text', '')
            source = obj.get('source', 'unknown')
            url = obj.get('url', '')

            # Clean extreme whitespace
            text = re.sub(r'\n{3,}', '\n\n', text).strip()
            
            if len(text) < 50:
                continue

            # Split the document text into chunks
            chunks = text_splitter.split_text(text)
            
            for chunk in chunks:
                if len(chunk.strip()) < 50:
                    continue # Skip tiny fragments
                
                # Prepend the title for context
                context_chunk = f"Title: {title}\n\n{chunk.strip()}"
                
                writer.write({
                    'source': source,
                    'url': url,
                    'chunk': context_chunk
                })
                chunks_created += 1
                
    return chunks_created

if __name__ == "__main__":
    os.makedirs('data/chunked', exist_ok=True)
    raw_files = glob.glob('data/raw_*.jsonl')
    
    total_chunks = 0
    print(f"Found {len(raw_files)} raw data files to process.")
    for f in raw_files:
        filename = os.path.basename(f)
        out_f = os.path.join('data', 'chunked', f"chunked_{filename}")
        print(f"Chunking {filename}...")
        
        # Reddit threads are naturally short, so we don't need heavy chunking
        # Wikis are long, need standard 1k chunking
        if 'reddit' in filename:
            chunks = chunk_file(f, out_f, chunk_size=2000, chunk_overlap=0)
        else:
            chunks = chunk_file(f, out_f, chunk_size=1000, chunk_overlap=150)
            
        print(f"  -> Created {chunks} chunks.")
        total_chunks += chunks
        
    print(f"\nTotal chunks created across all sources: {total_chunks}")
