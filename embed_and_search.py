# embed_and_search.py
import os
from dotenv import load_dotenv
import openai
import faiss
import numpy as np
from typing import List, Dict

EMBED_MODEL = "text-embedding-3-small"

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_embedding(text, model=EMBED_MODEL):
    resp = openai.embeddings.create(input=[text], model=model)
    return resp.data[0].embedding

def build_vector_store(chunks_meta: List[Dict]):
    """
    chunks_meta: list of {"text":..., "source":..., "chunk_id":...}
    Returns (faiss_index, dim)
    """
    embeddings = [get_embedding(c["text"]) for c in chunks_meta]
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings, dtype="float32"))
    return index

def search_index(index, chunks_meta: List[Dict], query: str, top_k=3) -> List[Dict]:
    q = np.array(get_embedding(query), dtype="float32").reshape(1, -1)
    distances, indices = index.search(q, top_k)
    out = []
    for i in indices[0]:
        out.append(chunks_meta[i])
    return out
