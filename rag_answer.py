# rag_answer.py

import os
from dotenv import load_dotenv
import openai
from embed_and_search import load_and_chunk_docx, build_vector_store, search_index

# Load API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Prompt template
def build_prompt(chunks, question):
    context = "\n\n".join(chunks)
    prompt = (
        "You are an expert assistant. Based only on the following document excerpts, "
        "answer the user's question.\n\n"
        f"Document excerpts:\n{context}\n\n"
        f"User question: {question}\n\n"
        "Answer:"
    )
    return prompt

# Call LLM
def ask_llm(prompt, model="gpt-4", temperature=0.7):
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature
    )
    return response.choices[0].message.content.strip()

# Main runner
if __name__ == "__main__":
    file_path = "input/input.docx"
    chunks = load_and_chunk_docx(file_path)

    print("🔄 Embedding and indexing...")
    index, _ = build_vector_store(chunks)

    question = input("❓ What would you like to ask about the document? ")
    top_chunks = search_index(index, chunks, question, top_k=3)

    prompt = build_prompt(top_chunks, question)
    print("\n📤 Sending to LLM...")
    answer = ask_llm(prompt)

    print("\n✅ LLM Answer:\n")
    print(answer)
