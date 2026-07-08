import json
import os
import sys
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY not found.")
    sys.exit(1)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# File paths
FAISS_INDEX_FILE = "data/faiss_index.bin"
METADATA_FILE = "data/metadata_lookup.json"

# Configurations
EMBEDDING_MODEL = "text-embedding-3-small"
GENERATION_MODEL = "gpt-4o"

def load_vector_store():
    """Loads the FAISS index and metadata lookup database from disk."""
    if not os.path.exists(FAISS_INDEX_FILE) or not os.path.exists(METADATA_FILE):
        print("Error: Vector store artifacts not found.")
        sys.exit(1)
        
    try:
        index = faiss.read_index(FAISS_INDEX_FILE)
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            metadata_lookup = json.load(f)
        return index, metadata_lookup
    except Exception as e:
        print(f"Error loading vector store components: {e}")
        sys.exit(1)

def get_query_embedding(query_text: str) -> list:
    """Embeds the user's query string using OpenAI."""
    try:
        response = client.embeddings.create(input=[query_text], model=EMBEDDING_MODEL)
        return response.data[0].embedding
    except Exception as e:
        print(f"API Error generating query embedding: {e}")
        return []

def retrieve_context(query_text: str, index, metadata_lookup, top_k: int = 3) -> list:
    """Searches FAISS for the top_k most similar chunks and returns their metadata."""
    query_vector = get_query_embedding(query_text)
    if not query_vector:
        return []
        
    # FAISS expects a 2D float32 array: shape (1, 1536)
    np_query = np.array([query_vector]).astype('float32')
    
    # index.search returns: distances (L2 scores), indices (row matches)
    distances, indices = index.search(np_query, top_k)
    
    retrieved_chunks = []
    # indices[0] contains the matched row IDs
    for match_idx in indices[0]:
        # FAISS will return -1 if it doesn't find enough matches to fill top_k
        if match_idx == -1:
            continue
            
        str_idx = str(match_idx)
        if str_idx in metadata_lookup:
            retrieved_chunks.append(metadata_lookup[str_idx])
            
    return retrieved_chunks

def generate_grounded_answer(query_text: str, retrieved_chunks: list) -> str:
    """Asks GPT-4o to answer the query strictly grounded in the retrieved context."""
    
    # Format the chunks into a clear text block for the LLM
    context_str = ""
    for i, chunk in enumerate(retrieved_chunks):
        context_str += f"--- Context Block {i+1} (Page {chunk['page_number']}) ---\n"
        context_str += f"{chunk['text']}\n\n"
        
    # System prompt enforcing strict grounding boundaries and fallback behavior
    system_prompt = (
        "You are a helpful, precise AI assistant answering questions based strictly on the provided context.\n"
        "Rules:\n"
        "1. Answer the question using ONLY the facts explicitly mentioned in the context.\n"
        "2. Do NOT use any external knowledge or assume/extrapolate facts not listed.\n"
        "3. If the context does not contain enough information to answer the question, you must respond exactly with:\n"
        "   'I cannot find that information in the document.'\n"
        "4. Keep your answer factual, direct, and concise."
    )
    
    user_prompt = f"Context:\n{context_str}\nQuestion: {query_text}\nAnswer:"
    
    try:
        response = client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0 # Low temperature ensures deterministic, factual output
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"API Error during generation: {e}"

def run_rag_pipeline(query: str):
    """the entire retrieval-generation workflow."""
    index, metadata_lookup = load_vector_store()
    
    print(f"\nUser Query: '{query}'")
    print("Retrieving matching text chunks from FAISS...")
    chunks = retrieve_context(query, index, metadata_lookup, top_k=3)
    
    print(f"Retrieved {len(chunks)} relevant chunks. Generating grounded response...")
    answer = generate_grounded_answer(query, chunks)
    
    print("\n" + "="*40 + "\nRESPONSE:\n" + "="*40)
    print(answer)
    print("="*40)
    
    # Return chunks for downstream evaluation tracking
    return answer, chunks

if __name__ == "__main__":
    # Test queries to verify pipeline behavior and fallback rules
    print("Testing pipeline with a valid document-based question:")
    run_rag_pipeline("What are the basics of renewable energy?")
    
    print("\nTesting pipeline fallback rule with an out-of-context question:")
    run_rag_pipeline("Who won the FIFA World Cup in 2022?")