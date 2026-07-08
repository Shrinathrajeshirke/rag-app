import time
import json
import os
import sys
from query import load_vector_store, retrieve_context, generate_grounded_answer

# Define a small Golden Set for evaluating our "Renewable Energy Basics" document
GOLDEN_SET = [
    {
        "query": "What are the basics of renewable energy?",
        "expected_page": 1  # Adjust based on your actual PDF layout
    },
    {
        "query": "Are there any charts or data about solar power growth?",
        "expected_page": 3  # Testing if GPT-4o vision description got caught
    },
    {
        "query": "What are the main limitations or downsides of wind energy?",
        "expected_page": 4
    }
]

def run_evaluation():
    """Runs a benchmark evaluation over the Golden Set to track hit-rate and speed."""
    print("\n" + "="*50)
    print("RUNNING RAG PIPELINE EVALUATION (GOLDEN SET)")
    print("="*50)
    
    index, metadata_lookup = load_vector_store()
    
    total_queries = len(GOLDEN_SET)
    hits = 0
    total_latency = 0.0
    
    for idx, test_case in enumerate(GOLDEN_SET):
        query = test_case["query"]
        expected = str(test_case["expected_page"])
        
        print(f"\n[{idx+1}/{total_queries}] Query: '{query}'")
        
        # Track retrieval latency
        start_time = time.time()
        retrieved_chunks = retrieve_context(query, index, metadata_lookup, top_k=3)
        latency = time.time() - start_time
        total_latency += latency
        
        # Check hit-rate: Did any of our top_k chunks come from the expected page?
        retrieved_pages = [str(chunk["page_number"]) for chunk in retrieved_chunks]
        is_hit = expected in retrieved_pages
        
        if is_hit:
            hits += 1
            status = "HIT"
        else:
            status = "MISS"
            
        print(f"  -> Latency: {latency:.4f}s")
        print(f"  -> Expected Page: {expected} | Retrieved Pages: {retrieved_pages} -> [{status}]")
        
        # Run generation to confirm it doesn't hallucinate
        answer = generate_grounded_answer(query, retrieved_chunks)
        print(f"  -> Answer Preview: {answer[:80]}...")

    # Compute high-level metrics
    hit_rate = (hits / total_queries) * 100
    avg_latency = total_latency / total_queries
    
    print("\n" + "="*50)
    print("EVALUATION METRICS SUMMARY")
    print("="*50)
    print(f"Total Test Queries : {total_queries}")
    print(f"Successful Hits    : {hits}")
    print(f"Retrieval Hit-Rate : {hit_rate:.2f}%")
    print(f"Avg Search Latency : {avg_latency:.4f} seconds")
    print("="*50 + "\n")

def run_interactive_demo():
    """Launches a multi-turn terminal chat interface for user interaction."""
    index, metadata_lookup = load_vector_store()
    
    print("\n" + "#"*50)
    print("  WELCOME TO YOUR MULTI-TURN RAG DEMO INTERFACE  ")
    print("#"*50)
    print("Type your questions below. Type 'exit' or 'quit' to close.\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("Closing session. Great job on completing your RAG project!")
                break
                
            # Execute standard pipeline lookup
            chunks = retrieve_context(user_input, index, metadata_lookup, top_k=3)
            answer = generate_grounded_answer(user_input, chunks)
            
            print(f"\nAI: {answer}")
            
            # Print citations cleanly
            if "I cannot find that information" not in answer and chunks:
                pages = sorted(list(set([str(c['page_number']) for c in chunks])))
                print(f"[Sources: Page(s) {', '.join(pages)}]\n")
            else:
                print("\n")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    # If you want to change the flow, you can toggle these calls
    run_evaluation()
    run_interactive_demo()