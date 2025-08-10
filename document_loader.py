# document_loader.py
import os
import glob
from typing import List, Dict, Tuple, Optional

from docx import Document
import fitz  # PyMuPDF

# ==============================
# Sentence-aware chunking utils
# ==============================

_ABBREVIATIONS = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "vs.", "etc.", "e.g.", "i.e.", "u.s.", "u.k.",
    "jan.", "feb.", "mar.", "apr.", "jun.", "jul.", "aug.", "sep.", "sept.", "oct.", "nov.", "dec."
}

def _simple_sentence_split(text: str) -> List[str]:
    """
    Regex-light sentence splitter:
    - Splits on ., ?, ! followed by whitespace/newline
    - Tries to avoid breaking on common abbreviations
    - Trims whitespace; drops empty sentences
    """
    # Normalize newlines -> spaces to reduce false splits
    clean = " ".join(text.split())
    sentences: List[str] = []

    start = 0
    i = 0
    while i < len(clean):
        ch = clean[i]
        if ch in ".!?":
            # look ahead to next space (end of sentence candidate)
            j = i + 1
            # allow multiple punctuation like "?!"
            while j < len(clean) and clean[j] in ".!?":
                j += 1
            # require a space or end of string after punctuation to split
            if j == len(clean) or (j < len(clean) and clean[j].isspace()):
                cand = clean[start:j].strip()
                lower_tail = cand.split()[-1].lower() if cand.split() else ""
                # Avoid splitting for common abbreviations
                if lower_tail in _ABBREVIATIONS:
                    i = j
                    continue
                if cand:
                    sentences.append(cand)
                # advance to after whitespace
                k = j
                while k < len(clean) and clean[k].isspace():
                    k += 1
                start = k
                i = k
                continue
        i += 1

    # remainder
    tail = clean[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences

def _pack_sentences_with_overlap(
    sentences: List[str],
    max_words: int = 300,
    overlap_words: int = 60
) -> List[str]:
    """
    Build chunks from full sentences with word-based overlap.
    - Fill each chunk up to ~max_words using whole sentences.
    - Advance by (max_words - overlap_words) words, aligned to sentence boundaries.
    """
    if not sentences:
        return []

    # Precompute sentence lengths
    lengths = [len(s.split()) for s in sentences]
    chunks: List[str] = []
    stride = max(1, max_words - overlap_words)

    start_idx = 0
    n = len(sentences)
    while start_idx < n:
        # Build a chunk
        total = 0
        end_idx = start_idx
        while end_idx < n and (total + lengths[end_idx] <= max_words or total == 0):
            total += lengths[end_idx]
            end_idx += 1
        chunk = " ".join(sentences[start_idx:end_idx]).strip()
        if chunk:
            chunks.append(chunk)

        # Advance start_idx by ~stride words, aligned to sentence boundaries
        advanced = 0
        next_start = start_idx
        while next_start < end_idx and advanced < stride:
            advanced += lengths[next_start]
            next_start += 1
        if next_start == start_idx:  # safety
            next_start += 1
        start_idx = next_start

    return chunks

def chunk_text_sentence_aware(text: str, max_words: int = 300, overlap: int = 60) -> List[str]:
    sentences = _simple_sentence_split(text)
    return _pack_sentences_with_overlap(sentences, max_words=max_words, overlap_words=overlap)

# ==============================
# .docx ingestion
# ==============================

def load_docx_text(file_path: str) -> str:
    """Extracts all non-empty paragraph text from a .docx file."""
    doc = Document(file_path)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)

def load_and_chunk_docx_with_meta(file_path: str, max_words: int = 300, overlap: int = 60) -> List[Dict]:
    """
    Return chunks with metadata for a .docx.
    NOTE: python-docx doesn't expose page numbers; we leave 'page' as None.
    """
    text = load_docx_text(file_path)
    base = os.path.basename(file_path)
    chunks = chunk_text_sentence_aware(text, max_words=max_words, overlap=overlap)

    out: List[Dict] = []
    for idx, ch in enumerate(chunks):
        out.append({
            "text": ch,
            "source": base,
            "chunk_id": idx,
            "page": None  # no real page info for docx
        })
    return out

# ==============================
# .pdf ingestion (with page numbers)
# ==============================

def load_and_chunk_pdf_with_meta(file_path: str, max_words: int = 300, overlap: int = 60) -> List[Dict]:
    """
    Extract text per-page from a PDF and chunk each page sentence-aware with overlap.
    Records the 1-based page number in metadata.
    """
    base = os.path.basename(file_path)
    out: List[Dict] = []
    with fitz.open(file_path) as doc:
        for pno in range(len(doc)):
            page = doc[pno]
            text = page.get_text("text").strip()
            if not text:
                continue
            page_chunks = chunk_text_sentence_aware(text, max_words=max_words, overlap=overlap)
            for idx, ch in enumerate(page_chunks):
                out.append({
                    "text": ch,
                    "source": base,
                    "chunk_id": idx,   # index within this page
                    "page": pno + 1    # human-friendly page number
                })
    return out

# ==============================
# Folder ingestion (single folder, .docx + .pdf)
# ==============================

def load_and_chunk_folder(folder_path: str, max_words: int = 300, overlap: int = 60) -> List[Dict]:
    """
    Load all .docx and .pdf files in folder.
    Returns list of dicts: {text, source, chunk_id, page (None for docx, int for pdf)}
    """
    results: List[Dict] = []

    # .docx
    for fp in sorted(glob.glob(os.path.join(folder_path, "*.docx"))):
        results.extend(load_and_chunk_docx_with_meta(fp, max_words=max_words, overlap=overlap))

    # .pdf
    for fp in sorted(glob.glob(os.path.join(folder_path, "*.pdf"))):
        results.extend(load_and_chunk_pdf_with_meta(fp, max_words=max_words, overlap=overlap))

    return results
