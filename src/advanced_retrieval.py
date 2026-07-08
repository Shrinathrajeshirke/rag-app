import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

## Initialize the local reranker model
print("Loading local Cross-Encoder Reranker (bge-reranker-base)...")
reranker = CrossEncoder("BAAI/bge-reranker-base")

def initialize_bm25(metadata_lookup: dict) -> BM25Okapi:
    """Tokenize all chunks in the metadata lookup and initialize BM25."""
    corpus = []
    sorted_keys = sorted(metadata_lookup.keys(), key=int)

    for key in sorted_keys:
        text = metadata_lookup[key]["text"]
        corpus.append(text.lower().split())

    return BM25Okapi(corpus)

def reciprocal_rank_fusion(vector_matches: list, bm25_matches: list, k: int = 60) -> list:
    """
    combines two ranked lists using Reciprocal Rank Fusion (RRF).
    """
    rrf_scores = {}

    # Calculate RRF score for vector matches
    for rank, key in enumerate(vector_matches):
        if key not in rrf_scores:
            rrf_scores[key] = 0.0
        rrf_scores[key] += 1.0/(k+rank+1)

    # Calculate RRF score for BM25 matches
    for rank, key in enumerate(bm25_matches):
        if key not in rrf_scores:
            rrf_scores[key] = 0.0
        rrf_scores[key]+=1.0/(k+(rank+1))
    
    sorted_keys = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [key for key, score in sorted_keys]

def advanced_retrieve(query_text: str, index, metadata_lookup, bm25_index, top_n_hybrid: int = 10, final_top_k: int = 3) -> list:
    """
    Executes Hybrid search (FAISS + BM25) -> Merges via RRF -> Reranks via Cross-Encoder.
    """
    # 1. Get FAISS matches
    from query import get_query_embedding
    query_vector = get_query_embedding(query_text)
    if not query_vector:
        return []
    
    np_query = np.array([query_vector]).astype('float32')
    _, faiss_indices = index.search(np_query, top_n_hybrid)
    vector_matches = [str(idx) for idx in faiss_indices[0] if idx != -1]

    # 2. Get BM25 matches
    tokenized_query = query_text.lower().split()
    bm25_scores = bm25_index.get_scores(tokenized_query)
    bm25_indices = np.argsort(bm25_scores)[::-1][:top_n_hybrid]
    bm25_matches = [str(idx) for idx in bm25_indices if bm25_scores[idx]>0]

    # 3. Merge lists using Reciprocal Rank Fusion
    hybrid_merged_keys= reciprocal_rank_fusion(vector_matches, bm25_matches)

    if not hybrid_merged_keys:
        return []
    
    candidates = [metadata_lookup[key] for key in hybrid_merged_keys]

    # 4. deep reranking using Cross-Encoder
    pairs = [[query_text, chunk["text"]] for chunk in candidates]

    rerank_scores= reranker.predict(pairs)

    for idx, score in enumerate(rerank_scores):
        candidates[idx]["rerank_score"] = float(score)

    reranked_candidates = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

    return reranked_candidates[:final_top_k]
    