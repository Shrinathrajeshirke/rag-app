import os
import json
import base64
import fitz
import pdfplumber
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI

client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))

PDF_PATH = "data/sample.pdf"
OUTPUT_PATH = "data/extracted.json"

def extract_text_per_page(pdf_path):
    """Extract raw text from each page using pdfplumber."""
    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages_text.append(text.strip())
    return pages_text

def describe_image(image_bytes):
    "using GPT-4o to convert image into a text description."
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    response = client.chat.completions.create(
        model = "gpt-4o",
        messages=[
            {"role": "user",
             "content": [
                 {"type":"text",
                  "text": ("""
                        Describe this image in 2-4 sentences, as if wrting 
                        cations for someone who can't see it. If it's a chart 
                        or graph, state the type of chart, what is measures,
                        and the key numbers/trend shown.
                        """
                  ),
                  },
                  {
                      "type": "image_url",
                      "image_url": {"url": f"data:image/png;base64,{b64}"},
                  },
             ],
             }
        ],
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()

def extract_images_per_page(pdf_path):
    """
    Extract embedded images per page using PyMuPDF, then describe
    with GPT-4o
    """
    doc = fitz.open(pdf_path)
    images_by_page = {}

    for page_index in range(len(doc)):
        page = doc[page_index]
        image_list = page.get_images(full=True)
        descriptions = []

        for img in image_list:
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            print(f" Describing image on page {page_index+1}...")
            try:
                description = describe_image(image_bytes)
                descriptions.append(description)
            except Exception as e:
                print(f" Warning: failed to describe image: {e}")
        if descriptions:
            images_by_page[page_index] = description

    doc.close()
    return images_by_page

def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set.")
    
    print(f"Extracting text from {PDF_PATH}...")
    pages_text = extract_text_per_page(PDF_PATH)
    print(f"  Found {len(pages_text)} pages of text.")

    print(f"Extracting and describing images from {PDF_PATH}...")
    images_by_page = extract_images_per_page(PDF_PATH)
    total_images = sum(len(v) for v in images_by_page.values())
    print(f"  Found and described {total_images} images.")

    #combine text + image descriptions into one document per page
    documents = []

    for i,text in enumerate(pages_text):
        image_descs = images_by_page.get(i,[])
        combined = text
        for j, desc in enumerate(image_descs):
            combined += f"\n\n[Image {j+1} on this page: {desc}]"
        
        documents.append({
            "page_number": i+1,
            "text": text,
            "image_description": image_descs,
            "combined_text": combined.strip(),
        })

    with open(OUTPUT_PATH, "w") as f:
        json.dump(documents, f, indent=2)

    print(f"\nDone. Saved {len(documents)} page-documents to {OUTPUT_PATH}")

if __name__=="__main__":
    main()
    