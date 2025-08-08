# rag_report.py
import os
from datetime import datetime
from dotenv import load_dotenv
import openai
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

from document_loader import load_and_chunk_docx
from embed_and_search import build_vector_store, search_index, EMBED_MODEL
from vector_store import save_index, load_index, file_sha256

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def build_prompt(chunks, question):
    context = "\n\n".join(chunks)
    return (
        "You are an expert analyst. Using only the information in the provided excerpts, "
        "answer the user's question concisely but with sufficient detail. If information is "
        "insufficient, say so explicitly.\n\n"
        f"Document excerpts:\n{context}\n\n"
        f"User question: {question}\n\n"
        "Answer:"
    )

def ask_llm(prompt, model="gpt-4", temperature=0.4):
    resp = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a careful, factual consultant."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature
    )
    return resp.choices[0].message.content.strip()

def write_report_docx(question, answer, chunks, out_path):
    doc = Document()

    title = doc.add_paragraph("Consulting Analysis Report")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.runs[0].font.size = Pt(20)

    date_p = doc.add_paragraph(datetime.now().strftime("%B %d, %Y"))
    date_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    date_p.runs[0].font.size = Pt(11)

    doc.add_paragraph("")
    h1 = doc.add_paragraph("Question"); h1.runs[0].bold = True
    doc.add_paragraph(question)

    h2 = doc.add_paragraph("Answer"); h2.runs[0].bold = True
    for para in answer.split("\n"):
        if para.strip():
            doc.add_paragraph(para.strip())

    doc.add_paragraph("")
    h3 = doc.add_paragraph("Sources (Excerpts Used)"); h3.runs[0].bold = True
    for i, ch in enumerate(chunks, start=1):
        doc.add_paragraph(f"Excerpt {i}:").runs[0].bold = True
        doc.add_paragraph(ch)

    doc.save(out_path)

if __name__ == "__main__":
    # Config
    source_doc = "input/input.docx"     # your document
    index_dir  = "index_store"          # where we persist FAISS + metadata
    os.makedirs("output", exist_ok=True)

    # Choose load vs build
    use_existing = input("Use existing index if available? [Y/n]: ").strip().lower()
    load_requested = (use_existing in ("", "y", "yes"))

    index = None
    chunks = None

    if load_requested:
        try:
            index, chunks, manifest = load_index(index_dir)

            # Model sanity check
            if manifest.get("model_name") != EMBED_MODEL:
                print(f"⚠️ Index built with {manifest.get('model_name')}, current model is {EMBED_MODEL}. Rebuilding.")
                index = None
            else:
                # Source doc hash check
                current_hash = file_sha256(source_doc)
                if manifest.get("source_hash") != current_hash:
                    print("⚠️ Source document has changed since index was built. Rebuilding.")
                    index = None
                else:
                    print(f"✅ Loaded index ({manifest['num_vectors']} vectors) from {index_dir}")
        except Exception as e:
            print(f"ℹ️ Could not load existing index ({e}). Will build a new one.")

    if index is None:
        # Build from source
        print("🔄 Loading & chunking document...")
        chunks = load_and_chunk_docx(source_doc, max_words=300)

        print("🔄 Embedding and indexing...")
        index, _ = build_vector_store(chunks)

        # Save for reuse
        save_index(
            index=index,
            chunks=chunks,
            index_dir=index_dir,
            model_name=EMBED_MODEL,
            source_path=source_doc,
            max_words=300
        )
        print(f"✅ Saved index to {index_dir}")

    # Ask + answer
    question = input("❓ What would you like to ask about the document? ")
    top_chunks = search_index(index, chunks, question, top_k=3)

    prompt = build_prompt(top_chunks, question)
    print("📤 Querying LLM...")
    answer = ask_llm(prompt)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = os.path.join("output", f"report_{ts}.docx")
    write_report_docx(question, answer, top_chunks, out_path)
    print(f"✅ Report saved to {out_path}")
