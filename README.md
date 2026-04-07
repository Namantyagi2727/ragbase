# RAGBase

**Offline document intelligence. No cloud. No API keys. Runs on your machine.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-1.x-1C3C3C?style=flat-square)](https://python.langchain.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.37-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Ollama](https://img.shields.io/badge/Ollama-local-black?style=flat-square)](https://ollama.com)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)

RAGBase is an open-source, fully offline RAG (Retrieval-Augmented Generation) system for querying your own documents. Upload PDFs, Word docs, spreadsheets, or images — ask questions in plain English — get answers with source citations. Everything stays on your machine.

---

## Demo

> **Add a screenshot or GIF here.**
> Run the app, take a screenshot of a question being answered, and drop it in a `docs/` folder:
> `![RAGBase demo](docs/demo.png)`

---

## How It Works

```
Documents (PDF / DOCX / TXT / CSV / Excel / Images)
        │
        ▼
   ingest.py ─────────────────────────────────────────────
   │  Tesseract OCR (images + scanned PDFs)               │
   │  RecursiveCharacterTextSplitter (configurable)       │
   │  HuggingFace Embeddings (all-MiniLM-L6-v2)          │──► ragbase_index/
   │  TF-IDF Vectorizer (sklearn)                         │     ├── index.faiss
   └──────────────────────────────────────────────────────┘     ├── tfidf_matrix.pkl
                                                                 └── chunks_meta.pkl
        ▼
   retriever.py ──────────────────────────────────────────
   │  HybridRetriever                                      │
   │  ├── FAISS semantic search                            │
   │  └── TF-IDF keyword search  →  merged + deduplicated │
   │  CrossEncoderReranker (ms-marco-MiniLM-L-6-v2)       │
   │  LCEL chain  (RunnableParallel + PromptTemplate)      │
   │  Ollama LLM  (llama3.2, fully local)                  │
   └──────────────────────────────────────────────────────┘
        │
        ▼
   app.py  ── Streamlit UI (dark theme, sidebar config)
```

---

## Features

| | |
|---|---|
| **Multi-format ingestion** | PDF (text + OCR fallback), DOCX, TXT, PNG/JPG, CSV, Excel |
| **Hybrid retrieval** | FAISS semantic search + TF-IDF keyword search, merged and deduplicated |
| **Cross-encoder reranking** | Optional re-scoring pass with `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **Tabular Q&A** | Auto-generates and executes pandas code for numeric questions over CSV/Excel |
| **Source citations** | Every answer shows which document chunks were used, with location |
| **Fully configurable** | Embedding model, LLM, chunk size, overlap, top-k, reranking — all from the UI |
| **Fully offline** | No internet required after setup. Nothing leaves your machine |

---

## Prerequisites

| Requirement | Install |
|---|---|
| Python 3.10+ | [python.org](https://python.org) |
| Tesseract OCR | `brew install tesseract` (macOS) · `apt install tesseract-ocr` (Ubuntu) |
| Ollama | [ollama.com](https://ollama.com) |

```bash
# Pull the default model (once Ollama is installed)
ollama pull llama3.2
```

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/ragbase.git
cd ragbase

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start Ollama (separate terminal)
ollama serve

# 5. Run
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501), upload your documents, click **Build Index**, then start asking questions.

---

## Configuration

Defaults live in `config.py`. All settings are also exposed in the UI sidebar at runtime.

| Parameter | Default | Description |
|---|---|---|
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | HuggingFace sentence-transformer for embeddings |
| `OLLAMA_MODEL` | `llama3.2` | Ollama model tag |
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between adjacent chunks |
| `TOP_K` | `5` | Chunks retrieved per query |
| `RERANKING_ENABLED` | `True` | Cross-encoder reranking pass |
| `INDEX_DIR` | `./ragbase_index` | FAISS + TF-IDF artifact directory |
| `DOCS_DIR` | `./uploaded_docs` | Staging directory for uploaded files |

### Swapping models

Any `sentence-transformers`-compatible model from HuggingFace works as the embedding model — just change `EMBED_MODEL` and rebuild the index.

Any Ollama model works as the LLM:
```bash
ollama pull mistral   # or gemma2:9b, phi3, llama3.1, etc.
```
Then select it in the UI and click **Apply & Reload**.

---

## Project Structure

```
ragbase/
├── app.py              # Streamlit UI
├── ingest.py           # Document loading, OCR, chunking, indexing
├── retriever.py        # HybridRetriever, reranker, LCEL chain
├── config.py           # All settings
├── requirements.txt
├── sample_docs/        # Demo documents
├── .streamlit/
│   └── config.toml     # Dark theme config
└── .gitignore
```

---

## Stack

- **UI** — [Streamlit](https://streamlit.io)
- **Orchestration** — [LangChain](https://python.langchain.com) (LCEL chains)
- **Embeddings** — [sentence-transformers](https://sbert.net) via `langchain-huggingface`
- **Vector store** — [FAISS](https://github.com/facebookresearch/faiss)
- **Keyword search** — [scikit-learn](https://scikit-learn.org) TF-IDF
- **Reranking** — [sentence-transformers](https://sbert.net) CrossEncoder
- **LLM** — [Ollama](https://ollama.com) (local inference)
- **PDF parsing** — [pdfplumber](https://github.com/jsvine/pdfplumber)
- **OCR** — [Tesseract](https://github.com/tesseract-ocr/tesseract) via pytesseract

---

## License

MIT — see [LICENSE](LICENSE).
