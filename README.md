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