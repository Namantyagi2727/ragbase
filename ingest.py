"""
ingest.py — Document ingestion pipeline for RAGBase.

Supports: PDF (native text + OCR fallback), DOCX, TXT,
          images (OCR), CSV, Excel.

Outputs to INDEX_DIR:
  index.faiss / index.pkl  — LangChain FAISS vectorstore
  tfidf_vectorizer.pkl     — sklearn TF-IDF vectorizer
  tfidf_matrix.pkl         — pre-computed TF-IDF matrix
  chunks_meta.pkl          — list of {text, source, loc} dicts
"""

import os
import re
import pickle
import logging
from typing import Callable, Optional

import numpy as np
import pandas as pd
import pytesseract
from PIL import Image, ImageFilter
from pdfplumber import open as open_pdf
from docx import Document as DocxDocument
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


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

def load_doc(path: str) -> list[tuple[str, str, str]]:
    """Load a single file and return raw (text, source, loc) tuples."""
    fn = os.path.basename(path)
    ext = fn.lower().rsplit(".", 1)[-1]

    if ext == "pdf":
        entries = []
        with open_pdf(path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                if not text.strip():
                    pil = page.to_image(resolution=300).original
                    text = ocr_image(pil)
                text = re.sub(r"-\n", "", text)
                text = re.sub(r"\n+", " ", text).strip()
                entries.append((text, fn, f"p.{i}"))
        return entries

    if ext == "docx":
        doc = DocxDocument(path)
        text = "\n".join(p.text for p in doc.paragraphs)
        return [(text, fn, "docx")]

    if ext == "txt":
        text = open(path, encoding="utf-8").read()
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

def main(
    ingest_dir: str = config.DOCS_DIR,
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
    embed_model: str = config.EMBED_MODEL,
    index_dir: str = config.INDEX_DIR,
    progress_fn: Optional[Callable[[float, str], None]] = None,
) -> int:
    """
    Ingest all documents in `ingest_dir`, build FAISS + TF-IDF indexes,
    save to `index_dir`. Returns the total number of chunks indexed.

    progress_fn(fraction: float, message: str) is called if provided.
    """
    os.makedirs(index_dir, exist_ok=True)

    # Collect files
    supported = {
        "pdf", "docx", "txt", "png", "jpg", "jpeg",
        "tiff", "bmp", "csv", "xlsx", "xls",
    }
    files = [
        f for f in sorted(os.listdir(ingest_dir))
        if f.lower().rsplit(".", 1)[-1] in supported and not f.startswith(".")
    ]
    if not files:
        log.warning("No supported documents found in %s", ingest_dir)
        return 0

    # Load and chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    lc_docs: list[Document] = []
    chunks_meta: list[dict] = []

    for idx, fn in enumerate(files):
        path = os.path.join(ingest_dir, fn)
        if progress_fn:
            progress_fn(idx / len(files), f"Loading {fn}…")
        log.info("Processing %s", fn)

        raw_entries = load_doc(path)
        for text, source, loc in raw_entries:
            if not text.strip():
                continue
            # Split each raw block with RecursiveCharacterTextSplitter
            sub_docs = splitter.create_documents(
                [text],
                metadatas=[{"source": source, "loc": loc}],
            )
            lc_docs.extend(sub_docs)
            for d in sub_docs:
                chunks_meta.append({
                    "text": d.page_content,
                    "source": source,
                    "loc": loc,
                })

    if not lc_docs:
        log.error("No text could be extracted from any document.")
        return 0

    log.info("Total chunks: %d", len(lc_docs))

    # Build FAISS vectorstore via LangChain
    if progress_fn:
        progress_fn(0.7, "Building FAISS index…")
    embeddings = HuggingFaceEmbeddings(model_name=embed_model)
    vectorstore = FAISS.from_documents(lc_docs, embeddings)
    vectorstore.save_local(index_dir)
    log.info("FAISS index saved to %s", index_dir)

    # Build TF-IDF index
    if progress_fn:
        progress_fn(0.85, "Building TF-IDF index…")
    texts = [c["text"] for c in chunks_meta]
    vectorizer = TfidfVectorizer(
        max_features=50_000,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)

    with open(os.path.join(index_dir, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(index_dir, "tfidf_matrix.pkl"), "wb") as f:
        pickle.dump(tfidf_matrix, f)
    with open(os.path.join(index_dir, "chunks_meta.pkl"), "wb") as f:
        pickle.dump(chunks_meta, f)

    if progress_fn:
        progress_fn(1.0, f"Done — {len(lc_docs)} chunks indexed.")
    log.info("Indexing complete: %d chunks.", len(lc_docs))
    return len(lc_docs)


if __name__ == "__main__":
    n = main()
    print(f"Indexed {n} chunks.")
