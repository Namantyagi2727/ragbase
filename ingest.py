"""
ingest.py — Document ingestion pipeline for RAGBase.

Supports: PDF (native text + OCR fallback), DOCX, TXT,
          images (OCR), CSV, Excel.

Outputs to INDEX_DIR:
  index.faiss / index.pkl  — LangChain FAISS vectorstore
  tfidf_vectorizer.pkl     — sklearn TF-IDF vectorizer
  tfidf_matrix.pkl         — pre-computed TF-IDF matrix
  chunks_meta.pkl          — list of {chunk_id, text, source, loc} dicts
  manifest.json            — embedding model, chunking settings, file list

Every chunk carries a deterministic `chunk_id` derived from its source,
location and text, so the same documents with the same chunking settings
always produce the same IDs.
"""

import hashlib
import json
import logging
import os
import pickle
import re
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import pytesseract
from docx import Document as DocxDocument
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pdfplumber import open as open_pdf
from PIL import Image, ImageFilter
from sklearn.feature_extraction.text import TfidfVectorizer

import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Refuse decompression-bomb images instead of only warning.
Image.MAX_IMAGE_PIXELS = config.MAX_IMAGE_PIXELS

MANIFEST_FILE = "manifest.json"
INDEX_FORMAT_VERSION = 1


# ---------------------------------------------------------------------------
# Chunk identity and file-name safety
# ---------------------------------------------------------------------------

def assign_chunk_ids(records: Iterable[tuple[str, str, str]]) -> list[str]:
    """Deterministic IDs for (source, loc, text) records, in order.

    ID = first 16 hex chars of sha256(source, loc, text). Identical chunks
    within the same source/location get "-2", "-3", ... suffixes by order of
    appearance, so the mapping is stable for a given corpus and chunking.
    """
    ids: list[str] = []
    counts: dict[str, int] = {}
    for source, loc, text in records:
        base = hashlib.sha256(f"{source}\x1f{loc}\x1f{text}".encode()).hexdigest()[:16]
        counts[base] = counts.get(base, 0) + 1
        ids.append(base if counts[base] == 1 else f"{base}-{counts[base]}")
    return ids


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._() -]+")


def safe_filename(name: str) -> Optional[str]:
    """Reduce an untrusted upload name to a safe basename, or None if unusable.

    Strips directories (path traversal), control characters and unusual
    symbols; rejects hidden files and unsupported extensions.
    """
    base = os.path.basename(name.replace("\\", "/"))
    base = unicodedata.normalize("NFKC", base).strip()
    if not base or base.startswith("."):
        return None
    base = _SAFE_NAME_RE.sub("_", base).strip(" .")
    stem, dot, ext = base.rpartition(".")
    if not dot or not stem or ext.lower() not in config.SUPPORTED_EXTENSIONS:
        return None
    if len(base) > 128:
        base = f"{stem[: 127 - len(ext) - 1]}.{ext}"
    return base


# ---------------------------------------------------------------------------
# OCR helper
# ---------------------------------------------------------------------------

def ocr_image(img: Image.Image) -> str:
    """Grayscale → median filter → binarize → Tesseract OCR."""
    gray = img.convert("L")
    blur = gray.filter(ImageFilter.MedianFilter(size=3))
    bw = blur.point(lambda x: 0 if x < 200 else 255, "1")
    return pytesseract.image_to_string(bw)


# ---------------------------------------------------------------------------
# File loaders  →  List[(text, source_filename, location_hint)]
# ---------------------------------------------------------------------------

def load_doc(path: str, warnings: Optional[list[str]] = None) -> list[tuple[str, str, str]]:
    """Load a single file and return raw (text, source, loc) tuples.

    Limit violations (page caps, OCR caps) are appended to `warnings`.
    """
    fn = os.path.basename(path)
    ext = fn.lower().rsplit(".", 1)[-1]
    warnings = warnings if warnings is not None else []

    if ext == "pdf":
        entries = []
        ocr_pages = 0
        with open_pdf(path) as pdf:
            if len(pdf.pages) > config.MAX_PDF_PAGES:
                warnings.append(
                    f"{fn}: {len(pdf.pages)} pages; only the first {config.MAX_PDF_PAGES} were indexed"
                )
            for i, page in enumerate(pdf.pages[: config.MAX_PDF_PAGES], start=1):
                text = page.extract_text() or ""
                if not text.strip():
                    if ocr_pages >= config.MAX_OCR_PAGES:
                        continue
                    ocr_pages += 1
                    pil = page.to_image(resolution=300).original
                    text = ocr_image(pil)
                text = re.sub(r"-\n", "", text)
                text = re.sub(r"\n+", " ", text).strip()
                entries.append((text, fn, f"p.{i}"))
        if ocr_pages >= config.MAX_OCR_PAGES:
            warnings.append(f"{fn}: OCR limit of {config.MAX_OCR_PAGES} scanned pages reached")
        return entries

    if ext == "docx":
        doc = DocxDocument(path)
        text = "\n".join(p.text for p in doc.paragraphs)
        return [(text, fn, "docx")]

    if ext == "txt":
        with open(path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        return [(text, fn, "txt")]

    if ext in ("png", "jpg", "jpeg", "tiff", "bmp"):
        img = Image.open(path)
        text = ocr_image(img)
        text = re.sub(r"-\n", "", text)
        text = re.sub(r"\n+", " ", text).strip()
        return [(text, fn, "ocr")]

    if ext == "csv":
        df = pd.read_csv(path, low_memory=False).infer_objects()
        return _ingest_tabular(df, fn)

    if ext in ("xlsx", "xls"):
        xls = pd.ExcelFile(path)
        entries = []
        for sheet in xls.sheet_names:
            df = xls.parse(sheet).infer_objects()
            entries.extend(_ingest_tabular(df, fn, sheet))
        return entries

    log.warning("Unsupported file type: %s", fn)
    return []


def _ingest_tabular(df: pd.DataFrame, filename: str, sheet: str | None = None) -> list:
    """Generate rich metadata chunk + row-block chunks for a DataFrame."""
    rows, cols = df.shape
    col_names = df.columns.tolist()
    dtypes = {c: str(dt) for c, dt in df.dtypes.items()}

    num_df = df.select_dtypes(include=["number"])
    stats = num_df.describe().to_dict() if not num_df.empty else {}

    cat_cols = df.select_dtypes(exclude=["number", "datetime", "timedelta"]).columns
    uniques = {c: int(df[c].nunique()) for c in cat_cols}

    meta_lines = [
        f"File: {filename}" + (f" (sheet: {sheet})" if sheet else ""),
        f"Rows: {rows}, Columns: {cols}",
        f"Column names: {col_names}",
        f"Dtypes: {dtypes}",
    ]
    if stats:
        meta_lines.append("Numeric summary stats:")
        for col, col_stats in stats.items():
            meta_lines.append(
                f"  {col}: count={col_stats['count']:.0f}, mean={col_stats['mean']:.3f}, "
                f"std={col_stats['std']:.3f}, min={col_stats['min']}, max={col_stats['max']}"
            )
    if uniques:
        meta_lines.append("Categorical unique counts:")
        for c, u in uniques.items():
            meta_lines.append(f"  {c}: unique={u}")
    if rows > 0:
        meta_lines.append("Sample first 5 rows:")
        for r in df.head(5).itertuples(index=False):
            meta_lines.append("  " + ", ".join(map(str, r)))

    chunks = [("\n".join(meta_lines), filename, "metadata")]

    records = df.values.tolist()
    for start in range(0, len(records), config.ROW_CHUNK_SIZE):
        block = records[start: start + config.ROW_CHUNK_SIZE]
        text_block = "\n".join(", ".join(map(str, r)) for r in block)
        loc = (
            f"{(sheet + '.') if sheet else ''}rows."
            f"{start + 1}-{min(start + config.ROW_CHUNK_SIZE, rows)}"
        )
        chunks.append((text_block, filename, loc))

    return chunks




# ---------------------------------------------------------------------------
# Main ingestion entry point
# ---------------------------------------------------------------------------

@dataclass
class IngestReport:
    chunks: int = 0
    files_indexed: list[str] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)   # (file, reason)
    warnings: list[str] = field(default_factory=list)


def list_ingestible_files(ingest_dir: str) -> list[str]:
    return [
        f for f in sorted(os.listdir(ingest_dir))
        if not f.startswith(".") and f.lower().rsplit(".", 1)[-1] in config.SUPPORTED_EXTENSIONS
    ]


def write_manifest(index_dir: str, **fields) -> dict:
    manifest = {
        "format_version": INDEX_FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **fields,
    }
    with open(os.path.join(index_dir, MANIFEST_FILE), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def read_manifest(index_dir: str) -> dict:
    """Index manifest, or {} for indexes built before manifests existed."""
    path = os.path.join(index_dir, MANIFEST_FILE)
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def ingest_documents(
    ingest_dir: str = config.DOCS_DIR,
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
    embed_model: str = config.EMBED_MODEL,
    index_dir: str = config.INDEX_DIR,
    progress_fn: Optional[Callable[[float, str], None]] = None,
    embeddings: Optional[Embeddings] = None,
    max_files: Optional[int] = None,
) -> IngestReport:
    """
    Ingest all documents in `ingest_dir`, build FAISS + TF-IDF indexes and a
    manifest in `index_dir`. A file that fails to load is skipped and
    reported; it does not abort the build.

    progress_fn(fraction: float, message: str) is called if provided.
    `embeddings` overrides the Hugging Face model (used by tests).
    `max_files` caps the number of files read (None = no cap; batch/CLI use).
    """
    report = IngestReport()
    os.makedirs(index_dir, exist_ok=True)

    files = list_ingestible_files(ingest_dir)
    if max_files is not None and len(files) > max_files:
        report.warnings.append(f"{len(files)} files found; only the first {max_files} were indexed")
        files = files[:max_files]
    if not files:
        log.warning("No supported documents found in %s", ingest_dir)
        return report

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    pieces: list[tuple[str, str, str]] = []   # (source, loc, text) in corpus order
    max_bytes = config.MAX_UPLOAD_MB * 1024 * 1024
    for idx, fn in enumerate(files):
        path = os.path.join(ingest_dir, fn)
        if progress_fn:
            progress_fn(0.7 * idx / len(files), f"Loading {fn}…")
        if os.path.getsize(path) > max_bytes:
            report.skipped.append((fn, f"larger than {config.MAX_UPLOAD_MB} MB"))
            continue
        log.info("Processing %s", fn)
        try:
            raw_entries = load_doc(path, warnings=report.warnings)
        except Exception as e:  # one bad file must not abort the whole build
            log.warning("Skipping %s: %s", fn, type(e).__name__)
            report.skipped.append((fn, f"could not be read ({type(e).__name__})"))
            continue
        produced = 0
        for text, source, loc in raw_entries:
            if not text.strip():
                continue
            for chunk in splitter.split_text(text):
                pieces.append((source, loc, chunk))
                produced += 1
        if produced:
            report.files_indexed.append(fn)
        else:
            report.skipped.append((fn, "no extractable text"))

    if not pieces:
        log.error("No text could be extracted from any document.")
        return report

    chunk_ids = assign_chunk_ids(pieces)
    chunks_meta = [
        {"chunk_id": cid, "text": text, "source": source, "loc": loc}
        for cid, (source, loc, text) in zip(chunk_ids, pieces, strict=True)
    ]
    lc_docs = [
        Document(
            page_content=c["text"],
            metadata={"chunk_id": c["chunk_id"], "source": c["source"], "loc": c["loc"]},
        )
        for c in chunks_meta
    ]
    log.info("Total chunks: %d", len(lc_docs))

    # Build FAISS vectorstore via LangChain; docstore IDs are the chunk IDs.
    if progress_fn:
        progress_fn(0.75, "Building FAISS index…")
    embeddings = embeddings or HuggingFaceEmbeddings(model_name=embed_model)
    vectorstore = FAISS.from_documents(lc_docs, embeddings, ids=chunk_ids)
    vectorstore.save_local(index_dir)

    # Build TF-IDF index
    if progress_fn:
        progress_fn(0.9, "Building TF-IDF index…")
    vectorizer = TfidfVectorizer(max_features=50_000, sublinear_tf=True)
    tfidf_matrix = vectorizer.fit_transform([c["text"] for c in chunks_meta])

    with open(os.path.join(index_dir, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(index_dir, "tfidf_matrix.pkl"), "wb") as f:
        pickle.dump(tfidf_matrix, f)
    with open(os.path.join(index_dir, "chunks_meta.pkl"), "wb") as f:
        pickle.dump(chunks_meta, f)

    write_manifest(
        index_dir,
        embed_model=embed_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunk_count=len(chunks_meta),
        files=report.files_indexed,
    )

    report.chunks = len(chunks_meta)
    if progress_fn:
        progress_fn(1.0, f"Done — {report.chunks} chunks indexed.")
    log.info("Indexing complete: %d chunks.", report.chunks)
    return report


def main(
    ingest_dir: str = config.DOCS_DIR,
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
    embed_model: str = config.EMBED_MODEL,
    index_dir: str = config.INDEX_DIR,
    progress_fn: Optional[Callable[[float, str], None]] = None,
) -> int:
    """Backward-compatible entry point: returns the number of chunks indexed."""
    return ingest_documents(
        ingest_dir, chunk_size, chunk_overlap, embed_model, index_dir, progress_fn
    ).chunks


if __name__ == "__main__":
    n = main()
    print(f"Indexed {n} chunks.")
