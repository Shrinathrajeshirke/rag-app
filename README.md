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

## How it works
 
```
PDF  →  ingest.py  →  chunk.py  →  embed.py  →  query.py / app.py
        (extract)     (split)     (vectorize)   (retrieve + answer)
```
 
1. **`ingest.py`** — Extracts text from each page (`pdfplumber`) and describes embedded images using GPT-4o vision (`PyMuPDF` for image extraction). Saves `data/extracted.json`.
2. **`chunk.py`** — Splits each page's text (+ image descriptions) into overlapping ~300-word chunks. Saves `data/chunks.json`.
3. **`embed.py`** — Converts each chunk into a vector using OpenAI's `text-embedding-3-small`, stores them in a FAISS index. Saves `data/faiss_index.bin` and `data/metadata_lookup.json`.
4. **`query.py`** — Core retrieval + generation logic: embeds a question, finds the closest chunks in FAISS, and asks GPT-4o to answer using only that context (falls back to "I cannot find that information" if nothing relevant is found).
5. **`app.py`** — Runs an evaluation against a small golden Q&A set (hit-rate, latency), then launches an interactive terminal chat.

### Advanced mode 
 
- **`advanced_retrieval.py`** — Upgrades retrieval with hybrid search: combines FAISS (semantic) and BM25 (keyword) results via Reciprocal Rank Fusion, then reranks the merged candidates with a local cross-encoder (`BAAI/bge-reranker-base`) for higher precision.
- **`advanced_app.py`** — Interactive chat using the hybrid retrieval pipeline instead of plain FAISS search.
## Setup
 
```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
 
pip install openai python-dotenv pdfplumber pymupdf faiss-cpu numpy
pip install rank_bm25 sentence-transformers   # only needed for advanced_* scripts
```
 
Create a `.env` file in the project root:
 
```
OPENAI_API_KEY=sk-your-key-here
```
 
Place your source PDF at `data/sample.pdf` (or update `PDF_PATH` in `ingest.py`).
 
## Usage
 
Run the pipeline steps in order:
 
```bash
python3 ingest.py     # PDF -> data/extracted.json
python3 chunk.py      # -> data/chunks.json
python3 embed.py      # -> data/faiss_index.bin, data/metadata_lookup.json
python3 app.py         # runs evaluation, then opens interactive chat
```
 
Or try the hybrid/reranked version:
 
```bash
python3 advanced_app.py
```
 
Type questions at the prompt; type `exit` or `quit` to leave.
 
## Notes
 
- Answers are strictly grounded in the retrieved chunks — the model is instructed not to use outside knowledge, and to say so when it can't find an answer in the document.
- `app.py`'s `GOLDEN_SET` (in-file) is a small hand-written test set mapping questions to the page they should be retrieved from; used to track retrieval hit-rate as you tune chunk size, top_k, or retrieval method.
- Re-run `chunk.py` and `embed.py` any time you change chunk size/overlap or swap in a new PDF.