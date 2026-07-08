import time
import os
import sys
from query import load_vector_store, generate_grounded_answer
from advanced_retrieval import initialize_bm25, advanced_retrieve

def run_advanced_demo():
    # Load foundational layers
    index, metadata_lookup = load_vector_store()
    
    # Build your sparse lookup index
    print("Initializing BM25 lexical index index from chunks...")
    bm25_index = initialize_bm25(metadata_lookup)
    
    print("\n" + "#"*50)
    print("  WELCOME TO YOUR ADVANCED HYBRID RAG DEMO  ")
    print("#"*50)
    print("Type your questions below. Type 'exit' or 'quit' to close.\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                break
                
            start_time = time.time()
            
            # Execute Advanced retrieval
            chunks = advanced_retrieve(
                query_text=user_input, 
                index=index, 
                metadata_lookup=metadata_lookup, 
                bm25_index=bm25_index,
                top_n_hybrid=10, # Gather 10 vector and 10 keyword targets
                final_top_k=3    # Rerank down to the 3 best
            )
            
            latency = time.time() - start_time
            answer = generate_grounded_answer(user_input, chunks)
            
            print(f"\nAI: {answer}")
            print(f"[Engine Speed: {latency:.4f}s]")
            
            # Print citations with their rerank relevancy metrics
            if "I cannot find that information" not in answer and chunks:
                citations = []
                for c in chunks:
                    citations.append(f"Page {c['page_number']} (Score: {c['rerank_score']:.2f})")
                print(f"[Sources: {', '.join(citations)}]\n")
            else:
                print("\n")
                
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    run_advanced_demo()