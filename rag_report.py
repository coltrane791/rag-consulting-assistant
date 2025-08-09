# rag_report.py
import os
import argparse
from datetime import datetime
from dotenv import load_dotenv
import openai
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

from document_loader import load_and_chunk_folder
from embed_and_search import build_vector_store, search_index, EMBED_MODEL
from vector_store import save_index, load_index, file_sha256

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def build_prompt(chunks, question, allow_general_knowledge: bool):
    """
    Build the LLM prompt. In strict mode, the model must use only excerpts.
    In hybrid mode, it may also use its general knowledge (but should prioritize excerpts).
    """
    excerpts_block = "\n\n".join([f"[{c['source']} #{c['chunk_id']}] {c['text']}" for c in chunks])

    if allow_general_knowledge:
        instruction = (
            "You are an expert analyst. Using both the provided document excerpts and your own general knowledge, "
            "compare, contrast, and synthesize to answer the user's question. If there is any conflict, prioritize "
            "the document excerpts. If you add anything not in the excerpts, treat it as general background."
        )
    else:
        instruction = (
            "You are an expert analyst. Using only the information in the provided document excerpts, "
            "answer the user's question concisely but with sufficient detail. If information is insufficient, say so."
        )

    return (
        f"{instruction}\n\n"
        f"Document excerpts (each labeled with source):\n{excerpts_block}\n\n"
        f"User question: {question}\n\n"
        "Answer:"
    )

def ask_llm(prompt, *, model="gpt-4", temperature=0.4):
    resp = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a careful, factual consultant."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature
    )
    return resp.choices[0].message.content.strip()

def write_report_docx(question, answer, chunks, out_path, *, mode_label: str):
    doc = Document()

    # Title
    title = doc.add_paragraph("Consulting Analysis Report")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.runs[0].font.size = Pt(20)

    date_p = doc.add_paragraph(datetime.now().strftime("%B %d, %Y"))
    date_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    date_p.runs[0].font.size = Pt(11)

    mode_p = doc.add_paragraph(f"Mode: {mode_label}")
    mode_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    mode_p.runs[0].font.size = Pt(10)

    doc.add_paragraph("")

    # Question
    h1 = doc.add_paragraph("Question"); h1.runs[0].bold = True
    doc.add_paragraph(question)

    # Answer
    h2 = doc.add_paragraph("Answer"); h2.runs[0].bold = True
    for para in answer.split("\n"):
        if para.strip():
            doc.add_paragraph(para.strip())

    # Sources appendix
    doc.add_paragraph("")
    h3 = doc.add_paragraph("Sources (Excerpts Used)"); h3.runs[0].bold = True
    for i, ch in enumerate(chunks, start=1):
        p = doc.add_paragraph(f"Excerpt {i} — {ch['source']} #{ch['chunk_id']}:")
        p.runs[0].bold = True
        doc.add_paragraph(ch["text"])

    doc.save(out_path)

def compute_corpus_sources(folder: str):
    """Return [{path, hash}] for all .docx in folder (single-folder mode)."""
    paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".docx")]
    return [{"path": os.path.abspath(p), "hash": file_sha256(p)} for p in sorted(paths)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG report generator")
    parser.add_argument("-g", "--use-general-knowledge", action="store_true",
                        help="Allow the model to use its general knowledge in addition to excerpts (hybrid mode).")
    parser.add_argument("--top-k", type=int, default=3, help="Number of excerpts to retrieve (default: 3).")
    parser.add_argument("--max-words", type=int, default=300, help="Chunk size in words (default: 300).")
    parser.add_argument("--model", type=str, default="gpt-4", help="OpenAI chat model (default: gpt-4).")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Override temperature. If not set, uses 0.4 (strict) or 0.6 (hybrid).")
    args = parser.parse_args()

    corpus_dir = "input"
    index_dir  = "index_store"
    os.makedirs("output", exist_ok=True)

    # Load vs build
    use_existing = input("Use existing index if available? [Y/n]: ").strip().lower()
    load_requested = (use_existing in ("", "y", "yes"))

    index = None
    chunks_meta = None

    if load_requested:
        try:
            index, chunks_meta, manifest = load_index(index_dir)
            # Model + corpus checks
            current_sources = compute_corpus_sources(corpus_dir)
            same_model = (manifest.get("model_name") == EMBED_MODEL)
            same_set = (manifest.get("sources") == current_sources)
            if not same_model:
                print(f"⚠️ Index model {manifest.get('model_name')} != current {EMBED_MODEL}. Rebuilding.")
                index = None
            elif not same_set:
                print("⚠️ Corpus has changed (file added/removed/modified). Rebuilding.")
                index = None
            else:
                print(f"✅ Loaded index ({manifest['num_vectors']} vectors) for {len(current_sources)} doc(s).")
        except Exception as e:
            print(f"ℹ️ Could not load existing index ({e}). Will build a new one.")

    if index is None:
        from document_loader import load_and_chunk_folder  # local import to avoid circulars
        print("🔄 Loading & chunking all documents in input/ ...")
        chunks_meta = load_and_chunk_folder(corpus_dir, max_words=args.max_words)

        from embed_and_search import build_vector_store, EMBED_MODEL
        print("🔄 Embedding and indexing...")
        index = build_vector_store(chunks_meta)

        sources = compute_corpus_sources(corpus_dir)
        save_index(
            index=index,
            chunks_meta=chunks_meta,
            index_dir=index_dir,
            model_name=EMBED_MODEL,
            sources=sources,
            max_words=args.max_words
        )
        print(f"✅ Saved index to {index_dir} for {len(sources)} doc(s).")

    # Query
    question = input("❓ What would you like to ask about the corpus? ")

    # Ensure chunks_meta is present if we loaded index
    if chunks_meta is None:
        # load_index already returned chunks_meta; only happens if logic above changes
        _, chunks_meta, _ = load_index(index_dir)

    top_chunks = search_index(index, chunks_meta, question, top_k=args.top_k)

    # Prompt + LLM
    prompt = build_prompt(top_chunks, question, allow_general_knowledge=args.use_general_knowledge)

    # Temperature: default 0.4 (strict) vs 0.6 (hybrid), unless overridden
    if args.temperature is not None:
        temp = args.temperature
    else:
        temp = 0.6 if args.use_general_knowledge else 0.4

    print("📤 Querying LLM...")
    answer = ask_llm(prompt, model=args.model, temperature=temp)

    # Save report
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = os.path.join("output", f"report_{ts}.docx")
    mode_label = "Hybrid (RAG + General Knowledge)" if args.use_general_knowledge else "Strict RAG (Excerpts Only)"
    write_report_docx(question, answer, top_chunks, out_path, mode_label=mode_label)
    print(f"✅ Report saved to {out_path}")
