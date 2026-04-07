"""
config.py — Central configuration for RAGBase.
All tuneable parameters live here; import from other modules.
"""

# Embedding model (Hugging Face sentence-transformers)
EMBED_MODEL: str = "all-MiniLM-L6-v2"

# Ollama LLM settings
OLLAMA_MODEL: str = "llama3.2"
OLLAMA_BASE_URL: str = "http://localhost:11434"

# Chunking (RecursiveCharacterTextSplitter)
CHUNK_SIZE: int = 512        # characters per chunk
CHUNK_OVERLAP: int = 64      # overlap between adjacent chunks

# Retrieval
TOP_K: int = 5               # number of chunks to retrieve per query

# Reranking (cross-encoder/ms-marco-MiniLM-L-6-v2)
RERANKING_ENABLED: bool = True

# Tabular ingestion
ROW_CHUNK_SIZE: int = 100    # rows per block for CSV/Excel

# Storage paths (gitignored)
INDEX_DIR: str = "./ragbase_index"   # FAISS + TF-IDF artifacts
DOCS_DIR: str = "./uploaded_docs"    # staging area for uploaded files

# Keywords that trigger tabular QA mode
NUMERIC_OPS: list = [
    "sum", "total", "average", "mean", "count",
    "max", "min", "median", "std", "how many",
]
