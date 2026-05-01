import os
import json
import re
import numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi

# Fallback embedding import
try:
    from sentence_transformers import SentenceTransformer
    HAS_SEMANTIC = True
except ImportError:
    HAS_SEMANTIC = False

DATA_DIR = "data"
OUTPUT_JSON = "code/corpus.json"
OUTPUT_EMBEDDINGS = "code/embeddings.npy"
CHUNK_TOKEN_LIMIT = 300 # Approx words
SEMANTIC_THRESHOLD = 2000

def simple_chunker(text, company, filename):
    """Splits by markdown headings, then by word count if too long."""
    chunks = []
    # Split by Markdown H1, H2, H3
    sections = re.split(r'\n(?=#+ )', text)
    
    for section in sections:
        words = section.strip().split()
        if not words:
            continue
            
        # Sub-chunk if section exceeds limit
        for i in range(0, len(words), CHUNK_TOKEN_LIMIT - 50): # 50 word overlap
            chunk_words = words[i:i + CHUNK_TOKEN_LIMIT]
            chunks.append({
                "id": f"{filename}_chunk_{len(chunks)}",
                "company": company,
                "text": " ".join(chunk_words)
            })
    return chunks

def build_corpus():
    corpus = []
    data_path = Path(DATA_DIR)
    
    if not data_path.exists():
        print(f"Error: {DATA_DIR} directory not found.")
        return []

    for filepath in data_path.rglob('*.*'):
        if filepath.suffix not in ['.md', '.txt', '.html']:
            continue
            
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
            
        # Infer company from directory or filename if possible, else "Unknown"
        company = "None"
        lower_path = str(filepath).lower()
        if "hackerrank" in lower_path: company = "HackerRank"
        elif "claude" in lower_path: company = "Claude"
        elif "visa" in lower_path: company = "Visa"
        
        corpus.extend(simple_chunker(text, company, filepath.stem))
        
    return corpus

def main():
    print("Parsing data directory...")
    corpus = build_corpus()
    chunk_count = len(corpus)
    print(f"Total chunks generated: {chunk_count}")
    
    # Save text corpus
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(corpus, f, indent=2)
    print(f"Saved {OUTPUT_JSON}")

    # Handle Embeddings
    if chunk_count > SEMANTIC_THRESHOLD:
        print(f"Chunk count ({chunk_count}) exceeds threshold ({SEMANTIC_THRESHOLD}). Dropping semantic search. Use BM25 only.")
        return

    if not HAS_SEMANTIC:
        print("sentence_transformers not installed. Skipping semantic embeddings.")
        return
        
    print("Loading all-MiniLM-L6-v2 model (CPU)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Computing embeddings... this may take a minute.")
    texts = [c["text"] for c in corpus]
    embeddings = model.encode(texts, show_progress_bar=True)
    
    np.save(OUTPUT_EMBEDDINGS, embeddings)
    print(f"Saved {OUTPUT_EMBEDDINGS}")

if __name__ == "__main__":
    main()