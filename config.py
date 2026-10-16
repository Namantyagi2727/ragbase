"""
config.py — Central configuration for RAGBase.
All tuneable parameters live here; import from other modules.

Values can be overridden with environment variables (shown in brackets),
which is how deployment and CI configure the app without code changes.
"""

import os


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# Embedding model (Hugging Face sentence-transformers)
EMBED_MODEL: str = _env_str("RAGBASE_EMBED_MODEL", "all-MiniLM-L6-v2")

# LLM backend [RAGBASE_LLM_BACKEND]: "ollama" (direct, fully local, default) or
# "prism" (through the Prism gateway). There is no automatic fallback between them.
LLM_BACKEND: str = _env_str("RAGBASE_LLM_BACKEND", "ollama")
LLM_TIMEOUT_S: float = _env_float("RAGBASE_LLM_TIMEOUT_S", 120.0)
LLM_TEMPERATURE: float = 0.1
LLM_NUM_CTX: int = 4096

# Prism gateway (RAGBASE_LLM_BACKEND=prism). Credentials come only from the
# environment; nothing here is a secret.
PRISM_BASE_URL: str = _env_str("PRISM_BASE_URL", "http://localhost:8000")
PRISM_MODEL: str = _env_str("PRISM_MODEL", "fast")            # Prism virtual model
PRISM_TIMEOUT_S: float = _env_float("PRISM_TIMEOUT_S", 120.0)
# Providers whose responses are accepted. Anything else — including Prism's mock
# providers and responses that do not identify their provider — is rejected.
PRISM_ALLOWED_PROVIDERS: frozenset = frozenset(
    p.strip() for p in _env_str("PRISM_ALLOWED_PROVIDERS", "ollama").split(",") if p.strip()
)
# Providers that run on the local machine (used to label inference mode).
LOCAL_PROVIDERS: frozenset = frozenset({"ollama"})

# Ollama LLM settings
OLLAMA_MODEL: str = _env_str("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL: str = _env_str("OLLAMA_BASE_URL", "http://localhost:11434")

# Chunking (RecursiveCharacterTextSplitter)
CHUNK_SIZE: int = 512        # characters per chunk
CHUNK_OVERLAP: int = 64      # overlap between adjacent chunks

# Retrieval
TOP_K: int = 5               # number of chunks passed to the LLM per query
CANDIDATE_POOL: int = 20     # candidates fetched from each retriever before fusion
RRF_K: int = 60              # reciprocal-rank-fusion damping constant

# Reranking (cross-encoder/ms-marco-MiniLM-L-6-v2)
RERANKING_ENABLED: bool = True
RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Agentic mode (LangGraph)
AGENT_MAX_REVISIONS: int = _env_int("RAGBASE_AGENT_MAX_REVISIONS", 1)   # refinement rounds after verification
AGENT_TIME_BUDGET_S: float = _env_float("RAGBASE_AGENT_TIME_BUDGET_S", 180.0)
AGENT_MAX_SUBQUERIES: int = 3
AGENT_K_PER_QUERY: int = 4
AGENT_MAX_EVIDENCE: int = 12
AGENT_MAX_COMPARE_DOCS: int = 5
# Optional separate model for the verifier (e.g. a larger one); empty = same as OLLAMA_MODEL
AGENT_VERIFIER_MODEL: str = _env_str("RAGBASE_AGENT_VERIFIER_MODEL", "")

# Tabular ingestion
ROW_CHUNK_SIZE: int = 100    # rows per block for CSV/Excel

# Tabular Q&A [RAGBASE_TABULAR_QA]: validated, allowlisted dataframe
# operations only — model output is never executed as code.
TABULAR_QA_ENABLED: bool = _env_bool("RAGBASE_TABULAR_QA", True)

# Public showcase [RAGBASE_PUBLIC_SHOWCASE]: read-only demo with the curated
# synthetic corpus and recorded runs; no uploads, no live generation. Enforced
# in code (ui/showcase.py never imports the ingestion or inference modules).
PUBLIC_SHOWCASE: bool = _env_bool("RAGBASE_PUBLIC_SHOWCASE", False)

# Storage paths (gitignored)
INDEX_DIR: str = _env_str("RAGBASE_INDEX_DIR", "./ragbase_index")   # FAISS + TF-IDF artifacts
DOCS_DIR: str = _env_str("RAGBASE_DOCS_DIR", "./uploaded_docs")     # staging area for uploads

# Upload / ingestion limits (resource-exhaustion guards)
MAX_UPLOAD_MB: int = _env_int("RAGBASE_MAX_UPLOAD_MB", 50)
MAX_UPLOAD_FILES: int = _env_int("RAGBASE_MAX_UPLOAD_FILES", 50)   # per upload in the UI
MAX_PDF_PAGES: int = _env_int("RAGBASE_MAX_PDF_PAGES", 500)
MAX_OCR_PAGES: int = _env_int("RAGBASE_MAX_OCR_PAGES", 50)
MAX_IMAGE_PIXELS: int = 50_000_000

SUPPORTED_EXTENSIONS: frozenset = frozenset({
    "pdf", "docx", "txt", "png", "jpg", "jpeg", "tiff", "bmp", "csv", "xlsx", "xls",
})

# Words that suggest an aggregate question; only route to tabular Q&A when
# the question also names a loaded table or one of its columns.
NUMERIC_OPS: list = [
    "sum", "total", "average", "mean", "count",
    "max", "min", "median", "std", "how many",
]
