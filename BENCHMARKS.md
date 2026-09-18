# RAGBase Benchmarks

Measured performance claims for RAGBase, so they can be cited without
qualification. Methodology, hardware, and raw numbers below; reproduce with
`make_benchmark_corpus.py` + `benchmark.py`.

## Verdict

| Claim | Status |
|---|---|
| "1,000+ documents" | **Verified.** 1,200 mixed-format documents ingested to 4,750 chunks in 14.4s. |
| "Sub-2-second retrieval" (as originally shipped) | **Not reliably true.** Median 1.36s, but max hit 2.13s across just 30 queries — caused by a real bug (below), not model cost. |
| "Sub-2-second retrieval" (after fixing the bug) | **True, with large margin.** Median 17ms. |
| "Sub-2-second" including LLM generation (as shipped) | **False.** Median 2.50s, p95 2.88s. |
| "Sub-2-second" including LLM generation (after fix) | **True.** Median 0.40s, p95 0.73s, max 0.89s. |

The earlier repository review was correct that this was unsubstantiated. It's
now measured, and the honest version is: **RAGBase retrieves from a
1,200-document / 4,750-chunk corpus in ~17ms median, and answers a question
end-to-end (retrieval + local LLM generation) in ~0.4s median**, on an Apple
M4 Pro / 24GB RAM — but only after a real caching bug in the reranker was
fixed. As originally written, the code's default configuration was
borderline-to-failing on "sub-2-second," including when generation is folded
in.

## Hardware

- Apple M4 Pro, 12 cores (8 performance + 4 efficiency), 24GB unified memory
- macOS (Darwin 27.0.0), Python 3.14.6
- No discrete GPU; embeddings and reranking run on CPU, LLM generation via
  Ollama uses Apple's Metal backend
- LLM: `llama3.2` (3B, Ollama). Embedding: `all-MiniLM-L6-v2`. Reranker:
  `cross-encoder/ms-marco-MiniLM-L-6-v2`

## Corpus

1,200 files, built by `make_benchmark_corpus.py` from 20 public-domain
novels (Project Gutenberg — real prose, not lorem ipsum), split into
180–420 word blocks and written across the formats RAGBase actually
supports:

| Format | Count |
|---|---|
| .txt | 874 |
| .pdf (real text layer, no OCR needed) | 212 |
| .docx | 94 |
| .csv (synthetic tabular data) | 20 |
| **Total** | **1,200** |

`ingest.py` (unmodified) chunks this into **4,750 chunks** via
`RecursiveCharacterTextSplitter` (512 chars, 64 overlap) in **14.4s**
(83.2 files/sec).

Caveat: no scanned/image-only PDFs were included, so the OCR fallback path
in `ingest.py` (Tesseract) is not exercised by this benchmark — that path is
substantially slower per page and would pull down throughput for a corpus
containing scanned documents.

## Query set

30 natural-language questions against the ingested novels (e.g. "Who is
Elizabeth Bennet and how does she feel about Mr. Darcy?", "How does Dr.
Jekyll transform into Mr. Hyde?") — see `QUERIES` in `benchmark.py`. 15 of
these were also run through full generation (LLM calls are the slow part;
15 gives a stable median without a multi-minute run).

## The bug

`retriever.py`'s `CrossEncoderReranker._get_relevant_documents` called
`CrossEncoder(self.model_name)` — reloading the entire reranker model from
disk into memory — on **every single query**, instead of once. This produced
a flat ~1.3–1.5s tax on every retrieval call regardless of query complexity
or corpus size, which is most of what was pushing retrieval toward (and
past) the 2-second line. Fixed by caching the loaded model in a module-level
dict, keyed by model name (`_get_cross_encoder()` in `retriever.py`).
Reranking itself, once the model is loaded, takes single-digit milliseconds
for `top_k=5` candidates.

This fix is applied in this working tree but **not yet committed** — verify
`git diff retriever.py` and decide whether to ship it.

## Results

All times in seconds, 30 queries (15 for generation), median / p95 / max.

| Stage | Before fix | After fix |
|---|---|---|
| Hybrid retrieval only (FAISS + TF-IDF, no reranking) | 0.008 / 0.131 / 0.266 | *(unaffected by fix)* |
| Hybrid + reranking (**default config**, `RERANKING_ENABLED=True`) | 1.355 / 1.526 / 2.134 | 0.017 / 0.020 / 0.021 |
| Retrieval + generation (full pipeline, llama3.2 via Ollama) | 2.504 / 2.879 / 5.323 | 0.404 / 0.728 / 0.893 |

The 5.323s outlier in the "before fix" generation row is the first query
in the run (Ollama's first model load); subsequent queries settled to
~2.4–2.9s.

Sanity check on the post-fix generation numbers (they look fast for an LLM
call): direct instrumentation against Ollama's `/api/generate` for one
query showed `prompt_eval_count=681` tokens in 21ms and `eval_count=71`
tokens in 753ms (~94 tok/s) — consistent with expected llama3.2 3B
throughput on Apple Silicon via Metal, not an artifact of caching.

## What "retrieval" means here

The original claim didn't specify. This benchmark reports three numbers so
the reader can pick the honest one for their context:

1. **Hybrid search alone** (semantic + keyword, no reranking) — fastest,
   but not RAGBase's default configuration.
2. **Hybrid + reranking** — RAGBase's actual default (`config.py`:
   `RERANKING_ENABLED = True`). This is what "retrieval" should mean if the
   claim is about the shipped default.
3. **Retrieval + generation** — what the user actually waits for end-to-end.
   If the portfolio claim is about the user-facing experience of asking a
   question, this is the number that matters.

## Reproducing this benchmark

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install reportlab              # only needed to build the synthetic corpus
ollama pull llama3.2 && ollama serve

python make_benchmark_corpus.py    # builds ./benchmark_corpus (~1,200 files)
python benchmark.py                # ingests it, times retrieval + generation
```

Both `benchmark_corpus/` and `benchmark_index/` are gitignored — they're
regenerated locally, not shipped in the repo.

## Caveats / limits of this benchmark

- Single machine, single run per condition — no multi-machine or
  cold-vs-warm-cache statistical study beyond the min/median/p95/max spread
  reported above.
- Corpus is derived from public-domain prose, not real enterprise documents
  (contracts, invoices, scanned forms) — token/vocabulary distribution
  differs, and no OCR path is exercised.
- Generation latency scales with answer length; the sample answers here
  average well under 100 output tokens. Longer-form answers would take
  longer.
- These numbers are for `llama3.2` (3B). A larger Ollama model
  (`gemma3:12b`, `llama3:latest`) would be meaningfully slower to generate
  with, though retrieval timing is unaffected by LLM choice.
