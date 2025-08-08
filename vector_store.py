# vector_store.py
import os, json, time, hashlib
import faiss

MANIFEST = "manifest.json"
INDEX = "index.faiss"
CHUNKS = "chunks.json"

def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def save_index(index, chunks, index_dir, *, model_name, source_path, max_words):
    os.makedirs(index_dir, exist_ok=True)

    # Save FAISS index
    faiss.write_index(index, os.path.join(index_dir, INDEX))

    # Save chunks
    with open(os.path.join(index_dir, CHUNKS), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    # Save manifest
    manifest = {
        "model_name": model_name,
        "source_path": os.path.abspath(source_path),
        "source_hash": file_sha256(source_path),
        "max_words": max_words,
        "num_vectors": index.ntotal,
        "saved_at": int(time.time()),
    }
    with open(os.path.join(index_dir, MANIFEST), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

def load_index(index_dir):
    idx_path = os.path.join(index_dir, INDEX)
    ch_path  = os.path.join(index_dir, CHUNKS)
    mf_path  = os.path.join(index_dir, MANIFEST)

    if not (os.path.exists(idx_path) and os.path.exists(ch_path) and os.path.exists(mf_path)):
        raise FileNotFoundError("Index files not found (index/chunks/manifest missing).")

    index = faiss.read_index(idx_path)
    with open(ch_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    with open(mf_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    return index, chunks, manifest
