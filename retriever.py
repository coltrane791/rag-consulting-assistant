# retriever.py
import os
from typing import List, Dict, Tuple
import numpy as np
from dotenv import load_dotenv
import openai

from embed_and_search import get_embedding  # uses your EMBED_MODEL

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def expand_queries(question: str, n: int = 3, model: str = "gpt-4o-mini") -> List[str]:
    """
    Generate N paraphrases/expansions of the user's question to broaden recall.
    Returns a de-duplicated list (original + expansions).
    """
    if n <= 0:
        return [question]

    prompt = (
        "Rewrite the user's query into diverse, semantically distinct variants that could retrieve "
        "different but relevant passages from a document corpus. Keep each variant brief.\n\n"
        f"User query: {question}\n\n"
        f"Produce {n} variants, one per line, with no numbering."
    )
    resp = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You produce short, diverse query paraphrases."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.6
    )
    lines = [l.strip("-• ").strip() for l in resp.choices[0].message.content.splitlines() if l.strip()]
    variants = [question] + lines[:n]

    # De-duplicate conservatively (case-insensitive)
    seen, out = set(), []
    for v in variants:
        key = v.lower()
        if key not in seen:
            seen.add(key)
            out.append(v)
    return out

def search_faiss_for_query(index, chunks_meta: List[Dict], query: str, k: int) -> Tuple[np.ndarray, np.ndarray]:
    q_vec = np.array(get_embedding(query), dtype="float32").reshape(1, -1)
    distances, indices = index.search(q_vec, k)
    return distances[0], indices[0]

def collect_candidates_multiquery(index, chunks_meta: List[Dict], question: str, expand_n: int, per_query_k: int):
    """
    Run FAISS retrieval for original question + N expansions. Return a dict:
      candidate_id -> {"best_distance": float, "hits": int}
    """
    variants = expand_queries(question, n=expand_n)
    candidate_stats = {}
    for q in variants:
        dists, idxs = search_faiss_for_query(index, chunks_meta, q, per_query_k)
        for dist, idx in zip(dists, idxs):
            if idx < 0:
                continue
            s = candidate_stats.get(idx, {"best_distance": float("inf"), "hits": 0})
            if dist < s["best_distance"]:
                s["best_distance"] = float(dist)
            s["hits"] += 1
            candidate_stats[idx] = s
    return candidate_stats, variants

def _format_snippet_label(c):
    if c.get("page"):
        return f"{c['source']} p.{c['page']}"
    return f"{c['source']} #{c['chunk_id']}"

def llm_rerank(question: str, candidates: List[Dict], model: str = "gpt-4o-mini") -> List[Tuple[int, float]]:
    """
    Ask an LLM to score each candidate chunk for relevance (0-100).
    Returns list of (candidate_index, score) sorted DESC by score.
    """
    # Build a compact prompt with numbered snippets.
    blocks = []
    for i, c in enumerate(candidates, start=1):
        label = _format_snippet_label(c)
        text = c["text"]
        blocks.append(f"{i}. [{label}] {text}")
    snippets_block = "\n\n".join(blocks)

    prompt = (
        "You are ranking snippets by their relevance to the user's question.\n"
        "Score each snippet 0-100 (integer), where 100 = highly relevant, 0 = irrelevant.\n"
        "Return a JSON array of objects: [{\"idx\": <number>, \"score\": <int>}], using the snippet numbers.\n\n"
        f"Question: {question}\n\n"
        f"Snippets:\n{snippets_block}\n\n"
        "Output JSON only."
    )

    resp = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return only valid JSON. Be strict."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )
    content = resp.choices[0].message.content.strip()

    # Very light JSON parsing without importing json to keep this file minimal:
    # (If you prefer, import json and do json.loads(content))
    import json
    try:
        arr = json.loads(content)
        # Convert to list of (zero_based_index, score)
        out = []
        for obj in arr:
            i = int(obj["idx"]) - 1
            sc = float(obj["score"])
            out.append((i, sc))
        # Sort DESC by score
        out.sort(key=lambda x: x[1], reverse=True)
        return out
    except Exception:
        # If something goes wrong, fall back to input order with flat score.
        return [(i, 50.0) for i in range(len(candidates))]

def retrieve_with_expansion_and_rerank(
    index,
    chunks_meta: List[Dict],
    question: str,
    *,
    top_k: int = 3,
    expand_n: int = 3,
    per_query_k: int = 8,
    use_llm_rerank: bool = True,
    rerank_model: str = "gpt-4o-mini"
) -> List[Dict]:
    """
    Multi-query expansion:
      - Generate N paraphrases
      - Retrieve per-query top-K
      - Union candidates (dedupe)
      - Rerank either with LLM scores or by best FAISS distance
      - Return final top_k chunks (dicts from chunks_meta)
    """
    candidate_stats, _variants = collect_candidates_multiquery(index, chunks_meta, question, expand_n, per_query_k)
    if not candidate_stats:
        return []

    # Build candidate list preserving original order by best_distance
    candidates = []
    by_idx_sorted = sorted(candidate_stats.items(), key=lambda kv: kv[1]["best_distance"])
    for idx, stats in by_idx_sorted:
        candidates.append(chunks_meta[idx])

    if use_llm_rerank:
        ranking = llm_rerank(question, candidates, model=rerank_model)
        # ranking is list of (zero_based_in_candidates, score)
        ordered = [candidates[i] for (i, _score) in ranking][:top_k]
        return ordered

    # Non-LLM fallback: sort by best FAISS distance (ascending)
    top = [chunks_meta[idx] for idx, _ in by_idx_sorted[:top_k]]
    return top
