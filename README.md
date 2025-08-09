# RAG Consulting Assistant

This project implements a Retrieval-Augmented Generation (RAG) workflow for analyzing and summarizing documents using OpenAI models.  
It loads `.docx` files, chunks them, builds a FAISS index, and answers questions using retrieved context (Strict RAG) or a combination of context + model general knowledge (Hybrid mode).

---

## 🚀 Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo

2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
.venv\Scripts\activate          # Windows

3. Install dependencies
pip install -r requirements.txt

4. Set-up environment variables
Copy .env.example to .env
   cp .env.example .env
Edit .env and add your OpenAI API key.

Project Structure
.
├── document_loader.py       # Loads and chunks .docx files
├── embed_and_search.py      # Creates embeddings and retrieves relevant chunks
├── rag_report.py            # Runs the RAG workflow and generates report
├── index_store/             # (ignored) Stores FAISS index files
├── input/                   # (ignored) Place your .docx files here
├── output/                  # (ignored) Generated reports saved here
├── requirements.txt         # Python dependencies
├── .env.example             # Template for environment variables
├── .gitignore
└── README.md

Usage
Basic (Strict RAG): python rag_report.py
Hybrid Mode (RAG + general knowledge): python rag_report.py -g
Example with custom parameters: python rag_report.py -g --top-k 5 --max-words 250 --temperature 0.55 --model gpt-4

Notes
The input/, output/, index_store/, .venv/, and .env files are ignored in Git to protect privacy and avoid large file uploads.

To share environment settings, commit .env.example but never commit your real .env.


---

## 🔧 Command-Line Flags

| Short Flag | Long Flag           | Type / Value                   | Default                       | Purpose |
|------------|--------------------|---------------------------------|--------------------------------|---------|
| `-g`       | `--use-general-knowledge` | Boolean switch                  | False                          | If set, runs in Hybrid Mode (LLM may use general knowledge + excerpts). If omitted, runs in Strict RAG mode (excerpts only). |
| (none)     | `--cite`            | Boolean switch                  | False                          | If set, requires inline citations like `[Source p.X]` after each factual claim, using only retrieved excerpts. |
| (none)     | `--top-k`           | Integer                         | 3                              | Number of most relevant excerpts **kept** after FAISS retrieval (and optional re-ranking). |
| (none)     | `--max-words`       | Integer                         | 300                            | Max words per chunk during ingestion. Changing this triggers index rebuild. |
| (none)     | `--model`           | String                          | "gpt-4"                        | Which OpenAI chat model to use for answering. |
| (none)     | `--temperature`     | Float (0–1)                     | 0.4 (strict), 0.6 (hybrid), 0.35 (strict+cite), 0.5 (hybrid+cite) | Controls creativity/randomness in output. Higher = more creative, lower = more factual/stable. |
| (none)     | `--expand-queries`  | Integer                         | 0                              | Number of alternative phrasings of the user’s question to generate for expanded retrieval. |
| (none)     | `--per-query-k`     | Integer                         | 8                              | Number of candidate chunks to retrieve per query/variant before merging and re-ranking. |
| (none)     | `--no-llm-rerank`   | Boolean switch                  | False (rerank ON by default)   | If set, skips LLM re-ranking and uses FAISS similarity scores directly. |
| (none)     | `--rerank-model`    | String                          | "gpt-4o-mini"                  | Model used for the LLM re-ranking step. |

---

## 📊 Retrieval & Re-Ranking Workflow
─────────────────────────────────────────┐
│ User enters query │
└─────────────────────────────────────────┘
│
▼
(If multi-query expansion is on)
Create N alternative phrasings
│
▼
┌─────────────────────────────────┐
│ For each query variant (incl. │
│ the original), search FAISS for │
│ --per-query-k candidates │
└─────────────────────────────────┘
│
▼
All candidates collected together
(can be > --top-k total)
│
▼
┌─────────────────────────────┐
│ LLM Re-ranking (if enabled)│
│ - Score each candidate for │
│ relevance to original Q │
│ - Sort by score │
└─────────────────────────────┘
│
▼
Keep only top N chunks overall:
N = --top-k
│
▼
Feed final chunks to answer LLM

## 🚀 Example Usage

```bash
# Strict RAG mode, top 5 chunks, no rerank
python rag_report.py --top-k 5 --no-llm-rerank

# Hybrid mode (general knowledge + excerpts), with citations
python rag_report.py -g --cite

# Multi-query expansion (3 variants), keep top 5 chunks, rerank with GPT-4
python rag_report.py --expand-queries 3 --top-k 5 --rerank-model gpt-4

# Strict mode with smaller chunks and higher creativity
python rag_report.py --max-words 200 --temperature 0.7