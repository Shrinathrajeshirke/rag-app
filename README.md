# Beginner-to-Advanced Hybrid RAG Project

An end-to-end Retrieval-Augmented Generation (RAG) pipeline built in Python. This project processes document text and embedded charts, indexes them using dense and sparse retrieval methods, and generates strictly grounded answers using OpenAI's `gpt-4o`.

## Features

- **Multimodal Ingestion:** Extracts text via `pdfplumber`/`PyMuPDF` and uses `gpt-4o` to generate visual descriptions for embedded charts/images.
- **Sliding-Window Chunking:** Custom token/word-count splitting with overlap to preserve semantic context boundaries.
- **Hybrid Search & Fusion:** Parallel retrieval combining dense vectors (**FAISS** via `text-embedding-3-small`) and lexical keywords (**BM25**), merged cleanly using **Reciprocal Rank Fusion (RRF)**.
- **Cross-Encoder Reranking:** Re-scores and optimizes the final context pool using a local `bge-reranker-base` model before passing it to the LLM.
- **Grounded Generation:** Guardrailed prompt engineering forcing zero-hallucination and deterministic fallback behavior.
- **Automated Evaluation:** Metrics framework to track pipeline **Hit-Rate %** and **Average Latency** against a custom Golden QA dataset.

---

## Project Structure

```text
rag_project/
├── data/
│   ├── sample.pdf             # Input source document
│   ├── extracted.json         # Raw text + vision parsing output (Step 1)
│   ├── chunks.json            # Sliding window processed chunks (Step 2)
│   ├── faiss_index.bin        # Dense vector matrix database (Step 3)
│   └── metadata_lookup.json   # FAISS row index mapping string keys (Step 3)
├── src/
│   ├── ingest.py              # PDF extraction and vision parsing engine
│   ├── chunk.py               # Custom word sliding window chunker
│   ├── embed.py               # FAISS vector database builder
│   ├── advanced_retrieval.py  # BM25 + RRF + Cross-Encoder pipeline
│   ├── query.py               # Base grounding prompt layout & OpenAI gateway
│   └── app.py                 # Integrated Evaluation & Interactive CLI Demo
├── .env                       # Local environment variables (API Keys)
├── requirements.txt           # Python library dependencies
└── README.md                  # Project documentation