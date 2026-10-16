# Historical latency benchmark (October 2026)

> **Historical record.** This page preserves a latency-only benchmark taken
> on 2026-10-13 (and earlier results from 2026-09-18), **before** the
> evaluation framework, the agent and the later retrieval and citation work.
> It measured two code versions that no longer exist on `main`:
>
> - **"Original"** — the pre-upgrade code at commit `909df31`.
> - **"Fixed"** — commit `6872aab` (2026-10-13), the hybrid-retrieval fusion
>   fix with the first evidence citations.
>
> Its figures were measured on a different corpus (1,200 public-domain
> documents), with different queries and conditions from the Phase 3
> evaluation, so they must not be combined with or compared directly to the
> current evaluation results. For answer quality, citations and the
> Standard vs Agentic comparison, see the
> [evaluation methodology](README.md#evaluation) and the
> [evaluation change log](docs/EVALUATION_CHANGELOG.md).
>
> The measurements below are unchanged from the original page. Two
> statements in it are out of date and are corrected inline: citations are
> now written `[1]`, `[2, 3]` (the `[E#]` form described below was replaced
> in Phase 2; legacy forms are still normalised), and the evaluation suite is
> complete rather than "in progress".

Measured ingestion and query latency on a 1,200-document corpus.
Reproduce with `make_benchmark_corpus.py` + `benchmark.py`; raw results are in
[`benchmarks/results/`](benchmarks/results/).

These benchmarks measure **speed only**. Retrieval and answer *quality* are
measured separately by the evaluation suite (*then in progress; now
complete — see the README's [Evaluation](README.md#evaluation) section*).

## Results (2026-10-13)

Same machine, same corpus, same session: the original retrieval code
(`main` @ `909df31`) and the fixed hybrid retrieval, two runs each.
Median / p95 / max in seconds.

| Stage | Original, run 1 | Original, run 2 | Fixed, run 1 | Fixed, run 2 |
|---|---|---|---|---|
| Hybrid retrieval, no reranking | 0.010 / 0.054 / 0.056 | 0.010 / 0.054 / 0.055 | 0.010 / 0.058 / 0.059 | 0.011 / 0.059 / 0.093 |
| Hybrid + reranking (**default**) | 0.069 / 0.077 / 0.172 | 0.077 / 0.091 / 0.092 | 0.089 / 0.107 / 0.120 | 0.083 / 0.101 / 0.105 |
| Retrieval + generation | 1.156 / 1.420 / 1.778 | 0.406 / 0.723 / 1.261 | 1.499 / 3.366 / 4.011 | 0.632 / 3.385 / 3.436 |

Ingestion: 1,200 files → **4,906 chunks** in 16.7–18.1s (66–72 files/sec), identical
chunk counts for both versions.

### What changed between the two versions

1. **Hybrid retrieval now actually fuses both retrievers.** The original
   `HybridRetriever` appended TF-IDF results *after* FAISS's top-k and then
   truncated to k, so keyword results never reached the output — "hybrid"
   retrieval behaved as FAISS-only. It now takes up to 20 candidates from
   each retriever and merges them with reciprocal-rank fusion.
2. **The reranker scores a larger pool.** Previously it re-ordered the same
   5 FAISS results; it now scores the top 20 fused candidates and keeps 5.
   This accounts for the extra ~10–20ms in the reranking row.
3. **Answers now carry evidence citations.** The standard prompt asks for
   `[E#]` citations tied to retrieved chunks (validated deterministically).
   *(Correction: the current format is `[1]`, `[2, 3]`; see the note above.)*
   Answers are fuller: in a diagnostic pass over the 15 generation queries,
   output length ranged from 15 to 245 tokens (median 51), and generation
   time tracked output length almost linearly. The higher p95 in the
   generation row comes from the longest answers, not from pipeline overhead.

### Reading these numbers

- Retrieval, including reranking, stays well under 0.2s at p95.
- End-to-end latency depends mostly on answer length. Median 0.6–1.5s across
  runs; p95 ~3.4s with citation-style answers versus 0.7–1.4s for the original
  prompt. Generation timings vary noticeably between runs (n=15 per run).
- The corpus generator re-downloads its source texts; today's corpus produced
  4,906 chunks for **both** versions versus 4,750 in September, so September
  numbers are not directly comparable with these.

## Hardware and configuration

- Apple M4 Pro, 12 cores, 24GB unified memory; macOS; Python 3.12
- Embeddings and reranking on Apple's MPS backend; LLM via Ollama (Metal)
- LLM `llama3.2` (3B) · embeddings `all-MiniLM-L6-v2` · reranker
  `cross-encoder/ms-marco-MiniLM-L-6-v2`
- `top_k=5`, `candidate_pool=20`, `rrf_k=60`, chunk size 512 / overlap 64
  (recorded in each result file's `config` block)

## Corpus

1,200 files built by `make_benchmark_corpus.py` (seed 42) from 20 public-domain
novels (Project Gutenberg), split into 180–420 word blocks:

| Format | Count |
|---|---|
| .txt | 874 |
| .pdf (text layer, no OCR needed) | 212 |
| .docx | 94 |
| .csv (synthetic tabular data) | 20 |
| **Total** | **1,200** |

No scanned/image-only PDFs are included, so the Tesseract OCR path is not
exercised; OCR is substantially slower per page.

## Query set

30 natural-language questions about the novels (see `QUERIES` in
`benchmark.py`); the first 15 also run through full generation. Several
questions are deliberately generic ("Describe the setting of the opening
chapter"), so abstentions are expected.

## Reproducing

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt -c constraints.txt
pip install reportlab              # only needed to build the synthetic corpus
ollama pull llama3.2 && ollama serve

python make_benchmark_corpus.py    # builds ./benchmark_corpus (~1,200 files)
python benchmark.py                # ingests it, times retrieval + generation
```

These scripts still run against the current code (checked 2026-10-16), so
re-running them measures **today's** pipeline, not the historical versions
above. To reproduce the historical numbers, check out `909df31` or `6872aab`
first. `make_benchmark_corpus.py` downloads its source texts, so the corpus
can differ slightly between builds (see "Reading these numbers").

`benchmark_corpus/` and `benchmark_index/` are gitignored and regenerated locally.

## Limitations

- Single machine; two runs per condition.
- Public-domain prose rather than enterprise documents; OCR not exercised.
- Generation latency scales with answer length and varies between runs.
- Numbers are for `llama3.2` (3B); larger models generate more slowly.

---

## Earlier results (2026-09-18) — FAISS-only behaviour

Preserved as originally published. These were measured **before** the hybrid
fusion fix: the code labelled "hybrid" returned FAISS results only (see
"What changed" above), and the reranker re-ordered 5 candidates.

| Stage | Median / p95 / max (s) |
|---|---|
| "Hybrid" retrieval only (effectively FAISS-only) | 0.008 / 0.131 / 0.266 |
| "Hybrid" + reranking (5 candidates) | 0.017 / 0.020 / 0.021 |
| Retrieval + generation (original prompt) | 0.404 / 0.728 / 0.893 |

Corpus at the time: 1,200 files → 4,750 chunks in 14.4s. Same hardware.
Python 3.14 environment.
