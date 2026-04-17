import fitz  # PyMuPDF
import os
import requests
import hashlib
import time
import sys
import base64
import io
import re
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
def is_page_number(text):
    t = text.strip().lower()
    t = re.sub(r'[^a-z0-9]', '', t)
    t = t.replace('page', '')
    return t.isdigit()

def pdf_to_html_simple(pdf_path, output_path):
    print(f"📄 Opening {pdf_path}...")
    doc = fitz.open(pdf_path)

    # Pass 1: Dynamic Margin & Header/Footer Detection
    print("🔍 Performing Pass 1: Dynamic Margin & Header/Footer Detection...")
    num_pages = len(doc)
    text_counts = {}
    img_counts = {}
    
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        rect = page.rect
        for b in blocks:
            # Check if block is in the top 15% or bottom 15%
            is_extremity = b["bbox"][1] < rect.height * 0.15 or b["bbox"][3] > rect.height * 0.85
            if is_extremity:
                if b["type"] == 0:  # Text
                    text = ""
                    for l in b.get("lines", []):
                        for s in l.get("spans", []):
                            text += s.get("text", "")
                    text = text.strip()
                    if text and not is_page_number(text):
                        text_counts[text] = text_counts.get(text, 0) + 1
                elif b["type"] == 1:  # Image
                    if "image" in b:
                        img_hash = hashlib.md5(b["image"]).hexdigest()
                        img_counts[img_hash] = img_counts.get(img_hash, 0) + 1

    header_footer_texts = {t for t, c in text_counts.items() if c > num_pages * 0.3}
    header_footer_images = {h for h, c in img_counts.items() if c > num_pages * 0.3}

    # Pass 2: Extraction
    html_content = "<!DOCTYPE html><html><head><meta charset='utf-8'><title>PDF Extract</title></head><body>\n"
    for page_num, page in enumerate(doc):
        print(f"Extracting page {page_num+1}...")
        
        rect = page.rect
        blocks = page.get_text("dict")["blocks"]
        tables = page.find_tables()
        table_bboxes = [fitz.Rect(t.bbox) for t in tables]
        
        valid_blocks = []
        for b in blocks:
             is_hf = False
             
             # Header/Footer filtering
             is_extremity = b["bbox"][1] < rect.height * 0.15 or b["bbox"][3] > rect.height * 0.85
             if is_extremity:
                  if b["type"] == 0:
                      text = "".join([s.get("text", "") for l in b.get("lines", []) for s in l.get("spans", [])]).strip()
                      if text in header_footer_texts or is_page_number(text):
                          is_hf = True
                  elif b["type"] == 1:
                      if "image" in b:
                          img_hash = hashlib.md5(b["image"]).hexdigest()
                          if img_hash in header_footer_images:
                              is_hf = True
             if is_hf:
                  continue
                  
             # Table text duplication filtering
             # If a block overlaps with ANY extracted table, we discard it to prevent phantom duplicates!
             b_rect = fitz.Rect(b["bbox"])
             is_in_table = False
             for t_box in table_bboxes:
                  if b_rect.intersects(t_box):
                       is_in_table = True
                       break
             if is_in_table:
                  continue
             
             valid_blocks.append(b)

        page_elements = [] # Items with {'y0': y0, 'html': html_string}
        
        for i, b in enumerate(valid_blocks):
             y0 = b["bbox"][1]
             if b["type"] == 0:
                 text_content = ""
                 for l in b.get("lines", []):
                     line_text = "".join([s.get("text", "") for s in l.get("spans", [])])
                     text_content += line_text + "<br>"
                 page_elements.append({'y0': y0, 'html': f"<p>{text_content}</p>\n"})
             elif b["type"] == 1:
                 if "image" in b:
                     # Gather surrounding context natively from sibling blocks
                     prev_texts = []
                     for pb in valid_blocks[max(0, i-5):i]:
                         if pb["type"] == 0:
                             prev_texts.append("".join([s.get("text", "") for l in pb.get("lines", []) for s in l.get("spans", [])]).strip())
                             
                     next_texts = []
                     for nb in valid_blocks[i+1:min(len(valid_blocks), i+6)]:
                         if nb["type"] == 0:
                             next_texts.append("".join([s.get("text", "") for l in nb.get("lines", []) for s in l.get("spans", [])]).strip())
                             
                     context_text = f"... {' '.join(prev_texts)} [IMAGE] {' '.join(next_texts)} ..."
                     
                     print(f"🖼️  Processing image...")
                     b64 = base64.b64encode(b["image"]).decode("utf-8")
                     ext = b.get("ext", "png")
                     desc = describe_image_with_ollama(b64, context_text)
                     page_elements.append({'y0': y0, 'html': f'<img src="data:image/{ext};base64,{b64}" data-description="{desc}" />\n'})
                     
        # Process the structured tables
        for table in tables:
             extracted = table.extract()
             if not extracted: continue
             
             y0 = table.bbox[1]
             t_html = "<table border='1'>\n"
             for row_idx, row in enumerate(extracted):
                 t_html += "<tr>"
                 for cell in row:
                     c_text = str(cell).replace("\n", "<br>") if cell is not None and str(cell) != "None" else ""
                     if row_idx == 0: t_html += f"<th>{c_text}</th>"
                     else: t_html += f"<td>{c_text}</td>"
                 t_html += "</tr>\n"
             t_html += "</table><br>\n"
             page_elements.append({'y0': y0, 'html': t_html})
             
        # Interleave everything vertically (top-to-bottom on the y-axis)
        page_elements.sort(key=lambda x: x['y0'])
        for el in page_elements:
             html_content += el['html']

    html_content += "</body></html>\n"

    # 3. Save modified HTML
    print(f"💾 Saving to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as out:
        out.write(html_content)
        
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
