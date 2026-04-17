import fitz  # PyMuPDF
import os
import requests
import hashlib
import time
import sys
import base64
import io
from PIL import Image
from bs4 import BeautifulSoup

# -------------------------------
# CONFIG
# -------------------------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://10.68.0.143:9100/api/generate")
# MODEL_NAME = "gemma4:31b"
MODEL_NAME = "llava:34b"

if "/api/v1/" in OLLAMA_URL:
    print("⚠️ Fixing Ollama endpoint (/api/v1 → /api)")
    OLLAMA_URL = OLLAMA_URL.replace("/api/v1/", "/api/")

image_hash_cache = {}

# -------------------------------
# UTILS
# -------------------------------
def get_image_hash(base64_str):
    try:
        img_data = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(img_data))
        # Hash the raw pixel data instead of the base64 string
        return hashlib.md5(img.tobytes()).hexdigest()
    except Exception:
        return hashlib.md5(base64_str.encode("utf-8")).hexdigest()

def clean_description(desc):
    bad_phrases = [
        "this image shows", "the image depicts", "it appears to",
        "without additional context", "it is difficult", "seems to"
    ]
    desc = desc.lower()
    for phrase in bad_phrases:
        desc = desc.replace(phrase, "")
    return desc.strip().capitalize()

# -------------------------------
# OLLAMA CALL
# -------------------------------
def describe_image_with_ollama(base64_img, context_text=""):
    img_hash = get_image_hash(base64_img)

    if img_hash in image_hash_cache:
        return image_hash_cache[img_hash]

    prompt = f"""
You are an expert technical analyst examing an image from an ORAN / OAI / FlexRIC technical document.

### Context from surrounding text:
{context_text}

### Task:
Analyze the image in deep detail, heavily factoring in the provided text context. Extract an exhaustive technical breakdown covering:
- Entities & Modules: Identify ALL explicit components, classes, network nodes, or system modules shown.
- Relationships & Topology: Explain the exact relationships, interfaces, and structural topology between these entities.
- Workflows & Data Flow: Map out the step-by-step sequences, control flows, and data paths (inputs/outputs) occurring.
- What & Why (Architecture): Detail WHAT process or architecture is presented, and WHY it is designed this way (design rationale, objectives).
- Deep Specifics & Logic: Extract any specific parameters, metrics, protocol names, API methods, algorithms, or code logic visible.
- Constraints & Edge Cases: Identify any implied or explicit bottlenecks, thresholds, performance limits, or failure modes shown.

Rules:
- Output a cohesive, highly dense engineering summary using professional terminology.
- Synthesize what is purely seen in the image with the nuances from the surrounding context.
- Never use generic introductory phrases (e.g., "This image shows", "We can observe"). Be direct.
"""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "images": [base64_img],
        "stream": False
    }

    for attempt in range(3):
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=90)
            response.raise_for_status()
            desc = clean_description(response.json().get("response", ""))

            image_hash_cache[img_hash] = desc
            return desc
        except Exception as e:
            print(f"⚠️ Ollama error (attempt {attempt+1}): {e}")
            time.sleep(2)

    return "ORAN-related system diagram."

# -------------------------------
# MAIN EXTRACTION
# -------------------------------
def pdf_to_html_simple(pdf_path, output_path):
    print(f"📄 Opening {pdf_path}...")
    doc = fitz.open(pdf_path)

    # 1. Extract HTML per page
    html_content = "<!DOCTYPE html><html><head><meta charset='utf-8'><title>PDF Extract</title></head><body>"
    for page_num, page in enumerate(doc):
        print(f"Extracting page {page_num+1}...")
        
        # Get raw HTML
        html_content += page.get_text("html")
        
        # Extract tables
        tables = page.find_tables()
        for i, table in enumerate(tables):
            extracted = table.extract()
            if extracted:
                html_content += "<table border='1'>"
                for row_idx, row in enumerate(extracted):
                    html_content += "<tr>"
                    for cell in row:
                        cell_text = str(cell) if cell is not None else ""
                        # Replace newlines with breaks
                        cell_text = cell_text.replace("\n", "<br>")
                        # Strip None strings or clean up
                        if cell_text == "None":
                            cell_text = ""
                        
                        if row_idx == 0:
                            html_content += f"<th>{cell_text}</th>"
                        else:
                            html_content += f"<td>{cell_text}</td>"
                    html_content += "</tr>"
                html_content += "</table><br>"

    html_content += "</body></html>"

    # 2. Parse HTML to modify img tags AND remove styles
    print("🔍 Parsing HTML and analyzing images...")
    soup = BeautifulSoup(html_content, "html.parser")

    # Remove all style attributes
    for tag in soup.find_all(True):
        if "style" in tag.attrs:
            del tag.attrs["style"]

    images = soup.find_all("img")
    
    for i, img in enumerate(images):
        print(f"🖼️  Processing image {i+1} / {len(images)}...")
        src = img.get("src", "")
        if src.startswith("data:image/"):
            try:
                # Extract the base64 part of the URI
                base64_data = src.split("base64,")[1]
                
                # Extract surrounding text context (ongoing chunk)
                prev_strings = [str(s).strip() for s in img.find_all_previous(string=True, limit=30) if str(s).strip()]
                next_strings = [str(s).strip() for s in img.find_all_next(string=True, limit=30) if str(s).strip()]
                
                context_text = f"... {' '.join(reversed(prev_strings))} [IMAGE] {' '.join(next_strings)} ..."
                
                # Get the description from the model
                description = describe_image_with_ollama(base64_data, context_text)
                
                # Add it as a new attribute
                img["data-description"] = description
            except IndexError:
                continue

    # 3. Save modified HTML
    print(f"💾 Saving to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as out:
        out.write(str(soup))
        
    print("✅ Done!")

# -------------------------------
# ENTRY
# -------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pdf_to_html.py <input.pdf> [output.html]")
        sys.exit(1)

    INPUT_FILE = sys.argv[1]
    
    if len(sys.argv) > 2:
        OUTPUT_FILE = sys.argv[2]
    else:
        OUTPUT_FILE = INPUT_FILE.replace(".pdf", ".html")
        
    pdf_to_html_simple(INPUT_FILE, OUTPUT_FILE)
