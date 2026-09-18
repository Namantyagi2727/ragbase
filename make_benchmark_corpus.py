"""
make_benchmark_corpus.py — Builds a realistic multi-format document corpus
for benchmarking RAGBase's ingestion and retrieval performance.

Downloads a set of public-domain novels (Project Gutenberg), splits them
into document-sized blocks of real prose, and writes them out as a mix of
.txt / .pdf / .docx / .csv files — mirroring the formats ingest.py supports.

Requires (not in requirements.txt — dev-only):
    pip install reportlab

Usage:
    python make_benchmark_corpus.py [--out ./benchmark_corpus] [--target 1200]
"""
import argparse
import csv
import glob
import os
import random
import re
import urllib.request

from docx import Document as DocxDocument
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

GUTENBERG_IDS = [
    1342, 84, 11, 2701, 1661, 98, 174, 43, 76, 1080,
    345, 2600, 5200, 1952, 4300, 158, 120, 219, 1400, 30254,
]
GUTENBERG_START = re.compile(r"\*\*\*\s*START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE | re.DOTALL)
GUTENBERG_END = re.compile(r"\*\*\*\s*END OF THE PROJECT GUTENBERG EBOOK.*", re.IGNORECASE | re.DOTALL)

WORDS_PER_DOC_MIN = 180
WORDS_PER_DOC_MAX = 420


def download_books(cache_dir: str) -> list[str]:
    os.makedirs(cache_dir, exist_ok=True)
    paths = []
    for gid in GUTENBERG_IDS:
        path = os.path.join(cache_dir, f"{gid}.txt")
        if not os.path.exists(path):
            url = f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt"
            try:
                urllib.request.urlretrieve(url, path)
            except Exception as e:
                print(f"  skip {gid}: {e}")
                continue
        paths.append(path)
    return paths


def load_clean_text(path: str) -> str:
    with open(path, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    text = GUTENBERG_START.split(text, maxsplit=1)[-1]
    text = GUTENBERG_END.split(text, maxsplit=1)[0]
    return text


def chunk_words(words, lo, hi, rng):
    i = 0
    while i < len(words):
        n = rng.randint(lo, hi)
        yield words[i:i + n]
        i += n


def write_txt(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def write_pdf(path, text):
    c = canvas.Canvas(path, pagesize=LETTER)
    width, height = LETTER
    margin = 0.75 * inch
    x, y = margin, height - margin
    c.setFont("Helvetica", 10)
    max_width = width - 2 * margin
    for para in text.split("\n\n"):
        line = ""
        for w in para.split():
            trial = (line + " " + w).strip()
            if c.stringWidth(trial, "Helvetica", 10) > max_width:
                c.drawString(x, y, line)
                y -= 14
                line = w
                if y < margin:
                    c.showPage()
                    c.setFont("Helvetica", 10)
                    y = height - margin
            else:
                line = trial
        if line:
            c.drawString(x, y, line)
            y -= 14
        y -= 8
        if y < margin:
            c.showPage()
            c.setFont("Helvetica", 10)
            y = height - margin
    c.save()


def write_docx(path, text):
    doc = DocxDocument()
    for para in text.split("\n\n"):
        doc.add_paragraph(para)
    doc.save(path)


def write_csv(path, rng):
    depts = ["Sales", "Engineering", "Support", "Marketing", "Finance", "Ops"]
    regions = ["NA", "EMEA", "APAC", "LATAM"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["employee_id", "department", "region", "quarter_revenue", "headcount"])
        for i in range(rng.randint(40, 120)):
            w.writerow([1000 + i, rng.choice(depts), rng.choice(regions),
                        round(rng.uniform(10_000, 500_000), 2), rng.randint(1, 50)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./benchmark_corpus")
    ap.add_argument("--target", type=int, default=1200)
    ap.add_argument("--cache-dir", default="./.gutenberg_cache")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)

    print(f"Downloading {len(GUTENBERG_IDS)} public-domain books...")
    books = download_books(args.cache_dir)

    all_words = []
    for b in books:
        all_words.extend(load_clean_text(b).split())
    print(f"Loaded {len(books)} books, {len(all_words)} words total")

    n_csv = 20
    docs_words = list(chunk_words(all_words, WORDS_PER_DOC_MIN, WORDS_PER_DOC_MAX, rng))
    docs_words = docs_words[: args.target - n_csv]

    n = len(docs_words)
    n_pdf, n_docx = int(n * 0.18), int(n * 0.08)
    n_txt = n - n_pdf - n_docx

    idx = 0
    manifest = []

    def body_for(words):
        text = " ".join(words)
        return "\n\n".join(text[j:j + 500] for j in range(0, len(text), 500))

    for _ in range(n_txt):
        body = body_for(docs_words[idx]); idx += 1
        fn = f"doc_{idx:05d}.txt"
        write_txt(os.path.join(args.out, fn), body)
        manifest.append((fn, "txt"))

    for _ in range(n_pdf):
        body = body_for(docs_words[idx]); idx += 1
        fn = f"doc_{idx:05d}.pdf"
        write_pdf(os.path.join(args.out, fn), body)
        manifest.append((fn, "pdf"))

    for _ in range(n_docx):
        body = body_for(docs_words[idx]); idx += 1
        fn = f"doc_{idx:05d}.docx"
        write_docx(os.path.join(args.out, fn), body)
        manifest.append((fn, "docx"))

    for i in range(n_csv):
        fn = f"data_{i:03d}.csv"
        write_csv(os.path.join(args.out, fn), rng)
        manifest.append((fn, "csv"))

    print(f"Wrote {len(manifest)} files to {args.out}")
    print(f"  txt={n_txt} pdf={n_pdf} docx={n_docx} csv={n_csv}")


if __name__ == "__main__":
    main()
