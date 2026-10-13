# RAGBase Benchmarks

Measured ingestion and query latency for RAGBase on a 1,200-document corpus.
Methodology, hardware, and raw numbers below; reproduce with
`make_benchmark_corpus.py` + `benchmark.py`.

## Summary

| Metric | Result |
|---|---|
| Corpus size | 1,200 mixed-format documents → 4,750 chunks |
| Ingestion time | 14.4s (83.2 files/sec) |
| Retrieval latency (hybrid + reranking, default config) | 17ms median, 20ms p95 |
| End-to-end latency (retrieval + local LLM generation) | 0.40s median, 0.73s p95, 0.89s max |

All measured on an Apple M4 Pro / 24GB RAM with `llama3.2` (3B) via Ollama.

## Hardware

- Apple M4 Pro, 12 cores (8 performance + 4 efficiency), 24GB unified memory
- macOS, Python 3.14
- No discrete GPU; embeddings and reranking run on CPU, LLM generation via
  Ollama uses Apple's Metal backend
- LLM: `llama3.2` (3B, Ollama). Embedding: `all-MiniLM-L6-v2`. Reranker:
  `cross-encoder/ms-marco-MiniLM-L-6-v2`

## Corpus

1,200 files, built by `make_benchmark_corpus.py` from 20 public-domain
novels (Project Gutenberg), split into 180–420 word blocks and written across
the formats RAGBase supports:

| Format | Count |
|---|---|
| .txt | 874 |
| .pdf (text layer, no OCR needed) | 212 |
| .docx | 94 |
| .csv (synthetic tabular data) | 20 |
| **Total** | **1,200** |

`ingest.py` chunks this into **4,750 chunks** via
`RecursiveCharacterTextSplitter` (512 chars, 64 overlap) in **14.4s**.

No scanned/image-only PDFs are included, so the Tesseract OCR path is not
exercised here; OCR is substantially slower per page and would lower
throughput for corpora containing scanned documents.

## Query set

30 natural-language questions against the ingested novels (e.g. "Who is
Elizabeth Bennet and how does she feel about Mr. Darcy?", "How does Dr.
Jekyll transform into Mr. Hyde?") — see `QUERIES` in `benchmark.py`. 15 of
these are also run through full generation.

## Results

All times in seconds, median / p95 / max.

| Stage | Latency |
|---|---|
| Hybrid retrieval only (FAISS + TF-IDF, no reranking) | 0.008 / 0.131 / 0.266 |
| Hybrid + reranking (**default config**, `RERANKING_ENABLED=True`) | 0.017 / 0.020 / 0.021 |
| Retrieval + generation (full pipeline, llama3.2 via Ollama) | 0.404 / 0.728 / 0.893 |

For reference, direct instrumentation against Ollama's `/api/generate` for one
query showed `prompt_eval_count=681` tokens in 21ms and `eval_count=71` tokens
in 753ms (~94 tok/s), consistent with llama3.2 3B throughput on Apple Silicon
via Metal.

## Performance note: reranker caching

The cross-encoder reranker is loaded once and cached per model name
(`_get_cross_encoder()` in `retriever.py`). Loading it per query would add a
flat ~1.3–1.5s to every retrieval call; with caching, reranking `top_k=5`
candidates takes single-digit milliseconds.

## What the three stages measure

1. **Hybrid search alone** (semantic + keyword, no reranking): the raw
   retrieval floor.
2. **Hybrid + reranking**: RAGBase's default configuration
   (`config.py`: `RERANKING_ENABLED = True`).
3. **Retrieval + generation**: what a user waits for end-to-end after asking
   a question.

## Reproducing this benchmark

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install reportlab              # only needed to build the synthetic corpus
ollama pull llama3.2 && ollama serve

python make_benchmark_corpus.py    # builds ./benchmark_corpus (~1,200 files)
python benchmark.py                # ingests it, times retrieval + generation
```

`benchmark_corpus/` and `benchmark_index/` are gitignored and regenerated
locally.

## Limitations

- Single machine, single run per condition.
- Corpus is public-domain prose rather than enterprise documents (contracts,
  invoices, scanned forms), so vocabulary distribution differs and the OCR
  path is not exercised.
- Generation latency scales with answer length; sample answers here average
  under 100 output tokens.
- Numbers are for `llama3.2` (3B). Larger Ollama models will generate more
  slowly; retrieval timing is unaffected by LLM choice.
