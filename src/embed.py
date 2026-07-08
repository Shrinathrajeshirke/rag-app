import json, sys
import os, faiss
import numpy as np
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY not found in environment variables or .env file.")
    sys.exit(1)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

INPUT_CHUNK_FILE = "data/chunks.json"
FAISS_INDEX_FILE = "data/faiss_index.bin"
METADATA_FILE = "data/metadata_lookup.json"

EMBEDDING_MODEL = "text-embedding-3-small"
VECTOR_DIMENSION = 1536

def load_chunks(file_path: str) -> list:
    """Loads chunks generated from the chunking phase."""
    if not os.path.exists(file_path):
        print(f"Error: Input file '{file_path}' not found.")
        sys.exit(1)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading chunks file: {e}")
        sys.exit(1)

def get_embedding(text: str) -> list:
    """Calls OpenAI API to fetch the embedding vector array."""
    try:
        response = client.embeddings.create(input=[text], model=EMBEDDING_MODEL)
        return response.data[0].embedding
    except Exception as e:
        print(f"API Error fetching embedding: {e}")
        return []
    
def build_vector_store():
    print("Starting vector embedding and index generation process..")
    chunks = load_chunks(INPUT_CHUNK_FILE)

    index = faiss.IndexFlatL2(VECTOR_DIMENSION)

    vector_list = []
    metadata_lookup = {}

    print(f"Generating embeddings for {len(chunks)} chunks...")

    for idx, chunk in enumerate(chunks):
        print(f"Processing chunk {idx+1}/{len(chunks)}..")
        text_to_embed = chunk.get("text", "")

        if not text_to_embed.strip():
            continue

        vector = get_embedding(text_to_embed)

        if not vector:
            print(f"Warning: Skipping chunk {idx} due to embedding failure.")
            continue

        vector_list.append(vector)

        metadata_lookup[str(idx)] = {
            "chunk_id": chunk.get("chunk_id",""),
            "page_number": chunk.get("page_number", ""),
            "text":text_to_embed
        }

    if not vector_list:
        print("Error: No vectors were generated. Vector store build failed.")
        return
        
    print("Converting vector array to NumPy float32 format for FAISS...")
    # FAISS strictly requires float32 NumPy matrices
    np_vectors = np.array(vector_list).astype('float32')
    
    print("Adding vectors to FAISS index...")
    index.add(np_vectors)
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(FAISS_INDEX_FILE), exist_ok=True)
    
    # Save the binary FAISS index
    print(f"Saving FAISS index to {FAISS_INDEX_FILE}...")
    faiss.write_index(index, FAISS_INDEX_FILE)
    
    # Save the lookup database mapping keys back to metadata strings
    print(f"Saving metadata lookup mapping to {METADATA_FILE}...")
    try:
        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(metadata_lookup, f, indent=4, ensure_ascii=False)
    except IOError as e:
        print(f"Error saving metadata file: {e}")
        sys.exit(1)
        
    print(f"\nSuccess! Built vector store containing {index.ntotal} items.")

if __name__ == "__main__":
    build_vector_store()
