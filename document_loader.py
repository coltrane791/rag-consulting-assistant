from docx import Document
import os
import textwrap
import glob
from typing import List, Dict

# === Single-doc functions ===
def load_docx_text(file_path):
    """Extracts and returns all paragraph text from a Word (.docx) file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    doc = Document(file_path)
    paragraphs = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
    full_text = "\n".join(paragraphs)
    return full_text

def chunk_text(text, max_words=300):
    """Splits long text into chunks of approximately `max_words` words."""
    words = text.split()
    chunks = []

    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        chunks.append(chunk)

    return chunks

def load_and_chunk_docx(file_path, max_words=300):
    """Loads a Word document and returns a list of text chunks."""
    raw_text = load_docx_text(file_path)
    print("Total words:", len(raw_text.split()))
    return chunk_text(raw_text, max_words=max_words)

# === Multi-doc functions ===
def load_docx_text_with_source(file_path) -> str:
    """Return full text from a .docx (same as before, but kept separate for clarity)."""
    return load_docx_text(file_path)

def chunk_text_with_meta(text: str, source: str, max_words=300) -> List[Dict]:
    """Chunk text and attach metadata: source filename and chunk index."""
    words = text.split()
    chunks = []
    idx = 0
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i+max_words])
        chunks.append({"text": chunk, "source": os.path.basename(source), "chunk_id": idx})
        idx += 1
    return chunks

def load_and_chunk_folder(folder_path: str, max_words=300) -> List[Dict]:
    """
    Load all .docx files in folder, return list of dicts:
      { "text": str, "source": filename, "chunk_id": int }
    """
    results: List[Dict] = []
    for fp in glob.glob(os.path.join(folder_path, "*.docx")):
        txt = load_docx_text_with_source(fp)
        results.extend(chunk_text_with_meta(txt, fp, max_words=max_words))
    return results

# Optional: Test this module directly
if __name__ == "__main__":
    test_path = "input/input.docx"  # adjust if needed
    chunks = load_and_chunk_docx(test_path)

    print(f"✅ Loaded {len(chunks)} chunk(s). Showing preview:")
    for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
        print(f"\n--- Chunk {i+1} ---\n{textwrap.shorten(chunk, width=300)}")

