import json
import os
import sys
from typing import List, Dict

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
INPUT_FILE = "data/extracted.json"
OUTPUT_FILE = "data/chunks.json"

def load_extracted_data(file_path: str) -> List[Dict]:
    """Loads the page-by-page extracted data from the ingestion step."""
    if not os.path.exists(file_path):
        print(f"Error: Input file '{file_path} not found.")
        sys.exit(1)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, list):
                print("Error: Input data format is incorrect. Expected list of pages.")
                sys.exit(1)
            return data
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON from '{file_path}': {e}")
        sys.exit(1)

def split_text_into_chunks(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    if not text or not text.strip():
        return []
    
    words = text.split()

    if len(words) <= chunk_size:
        return [" ".join(words)]
    
    chunks = []
    step = chunk_size - chunk_overlap

    if step<= 0:
        print("Warning: Overlap is greater than or equal to chunk size. Defaulting overlap to 0.")
        step = chunk_size
    
    for i in range(0, len(words), step):
        chunk_words = words[i: i+chunk_size]

        if len(chunk_words) < chunk_size and len(chunks) > 0 and len(chunk_words) < (chunk_size/4):
            break
        chunks.append(" ".join(chunk_words))

    return chunks

def process_chunks() -> None:
    "Main execution function to read extracted data, chunk it and save results."
    print("Starting semantic chunking process...")

    # Load data from the ingestion phase
    pages_data = load_extracted_data(INPUT_FILE)

    processed_chunks = []
    chunk_id_counter = 0

    for item in pages_data:
        page_num = item.get("page_number", "unknown")
        raw_text = item.get("text","")
        vision_desc_list = item.get("image_description", [])
        vision_desc = "\n".join(vision_desc_list) if isinstance(vision_desc_list, list) else str(vision_desc_list)

        combined_text = f"{raw_text}\n{vision_desc}".strip()

        if not combined_text:
            continue

        text_chunks = split_text_into_chunks(combined_text, CHUNK_SIZE, CHUNK_OVERLAP)

        for chunk_text in text_chunks:
            processed_chunks.append({
                "chunk_id": f"chunk_{chunk_id_counter:04d}",
                "page_number": page_num,
                "text": chunk_text
            })
            chunk_id_counter += 1

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(processed_chunks, f, indent=4, ensure_ascii=False)
        print(f"Successfully generated {len(processed_chunks)} chunks and saved to '{OUTPUT_FILE}'.")
    except IOError as e:
        print(f"Error saving chunks to '{OUTPUT_FILE}': {e}")

if __name__ == "__main__":
    process_chunks()