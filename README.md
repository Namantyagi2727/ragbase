# RAGBase

**Local-first document Q&A that compares Standard and Agentic RAG — with inspectable citations, a bounded LangGraph agent, and an evaluation that reports where each approach falls short.**

[![CI](https://github.com/Namantyagi2727/ragbase/actions/workflows/ci.yml/badge.svg)](https://github.com/Namantyagi2727/ragbase/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangGraph](https://img.shields.io/badge/LangGraph-agent-1C3C3C?style=flat-square)](https://langchain-ai.github.io/langgraph/)
[![Ollama](https://img.shields.io/badge/Ollama-local-black?style=flat-square)](https://ollama.com)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)

RAGBase answers questions about your own documents and shows its work: every
answer cites numbered passages you can open, and the optional agent mode
shows how it planned, searched and checked its claims. It runs entirely on
your machine with [Ollama](https://ollama.com) by default.

## Two ways to see it

| | Public showcase | Full local app |
|---|---|---|
| Where | **https://ragbase-showcase.streamlit.app/** | Your machine (`streamlit run app.py`) |
| Answers | **Recorded runs** of the real pipeline, replayed with provenance. **No live model inference** | Live answers from your local Ollama model |
| Documents | A curated synthetic corpus (fictional company) | Your own uploads, indexed per session |
| Questions | Ten curated questions; no free-text input | Anything you ask |

The showcase exists so anyone can inspect real outputs — citations, agent
traces, verification results and known failures — without installing
anything. To try your own documents and questions, run it locally.

![Public showcase: recorded Standard and Agentic answers side by side](docs/images/showcase-compare.png)

## Engineering highlights

- **Hybrid retrieval** — FAISS semantic search and TF-IDF keyword search
  fused with reciprocal-rank fusion, then cross-encoder reranking.
- **Inspectable citations** — answers cite numbered evidence (`[1]`, `[2, 3]`);
  IDs are validated against the retrieved passages, and cited passages are
  shown separately from those retrieved but not used.
- **Agentic RAG with LangGraph** — a bounded workflow that plans sub-queries,
  selects documents for comparisons, verifies claims against their
  citations and refines at most once, with a structured execution trace.
- **Honest verification wording** — a citation or a verifier verdict is
  presented as evidence about an answer, never as proof that it is correct.
- **Evaluation framework** — a human-reviewed benchmark comparing the
  original code, Standard RAG and Agentic RAG, with versioned labels and
  scoring ([results](#evaluation), [full evaluation](docs/EVALUATION.md)).
- **CI/CD** — linting, unit and integration tests on every pull request, a
  manually triggered live-inference workflow, and a pinned deployment for
  the public showcase.

## Features

| | |
|---|---|
| **Standard RAG** | One hybrid search, one model call, numbered citations checked for validity |
| **Agentic RAG** | LangGraph workflow: plan → retrieve → write → verify → (refine) → finish, bounded by time, revision and search limits |
| **Comparison questions** | Documents named in the question (or by the planner) are searched separately; otherwise a relevance search chooses them |
| **Claim verification** | Deterministic checks (citation IDs, numbers present in the cited text) plus per-claim model verdicts; results shown as checked, partly checked or not checked |
| **Multi-format ingestion** | PDF (with column-aware extraction for two-column pages), DOCX, TXT, CSV, Excel, and images via OCR |
| **Tabular Q&A** | Questions over CSV/Excel answered with allowlisted dataframe operations; model output is never executed as code |
| **Inference backends** | Direct local Ollama (default) or an optional [Prism](docs/PRISM.md) gateway with fail-closed provider verification |
| **Session-isolated uploads** | Each browser session indexes its uploads in its own temporary workspace |

![Local app: a live Standard RAG answer with citations, the mode selector and the inference panel](docs/images/local-standard.png)

## How it works

```mermaid
flowchart LR
    D["Documents"] --> I["Ingestion<br/>column-aware PDF · OCR · chunking"]
    I --> X[("FAISS + TF-IDF index")]
    Q["Question"] --> M{"Mode"}
    M -->|Standard| S["Hybrid search → rerank →<br/>one cited answer"]
    M -->|Agentic| A["LangGraph: plan → retrieve →<br/>write → verify → refine"]
    X --> S
    X --> A
    S --> O["Answer, citations,<br/>evidence"]
    A --> O2["Answer, citations,<br/>evidence, verification,<br/>execution trace"]
```

Detailed diagrams — ingestion, retrieval, the agent graph, verification
states, the Prism gateway and the local/showcase split — are in
**[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

<p>
  <img src="docs/images/showcase-agent-trace.png" alt="Agent execution summary for a comparison question" width="360">
  <img src="docs/images/showcase-documents.png" alt="Source document viewer showing a two-column PDF page read column by column" width="480">
</p>

*Left: the agent's execution summary for a comparison question (recorded run) — it searched exactly the two policy versions, and its citation checks flagged two figures that don't appear in the passages they cite. Right: the source-document viewer.*

## Quickstart (local, Python 3.12)

Prerequisites: **Python 3.12** (the tested version), **[Ollama](https://ollama.com)**, and optionally **Tesseract OCR** (`brew install tesseract` or `apt install tesseract-ocr`) — needed only for images and scanned PDF pages.

```bash
git clone https://github.com/Namantyagi2727/ragbase.git
cd ragbase
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt -c constraints.txt
ollama pull llama3.2               # the default model; Ollama must be running
streamlit run app.py
```

Open http://localhost:8501, upload documents, click **Build Index**, choose
**Standard RAG** or **Agentic RAG**, and ask a question. The first run
downloads the embedding and reranking models from Hugging Face.

`constraints.txt` pins every dependency version used in CI, so a fresh
install matches what was tested. To preview the public showcase locally
(only Streamlit needed): `pip install -r showcase/requirements.txt`, then
`streamlit run showcase/streamlit_app.py`.

### Development

```bash
pip install -r requirements-dev.txt -c constraints.txt
ruff check .
pytest tests/unit                  # no models needed
pytest tests/integration           # real embedding models; live Ollama/Prism tests skip if unavailable
```

## Configuration

Defaults live in [`config.py`](config.py); model, chunking and retrieval
settings can also be changed in the app's sidebar.

| Setting | Environment variable | Default |
|---|---|---|
| Ollama model | `OLLAMA_MODEL` | `llama3.2` |
| Ollama server | `OLLAMA_BASE_URL` | `http://localhost:11434` |
| Backend | `RAGBASE_LLM_BACKEND` | `ollama` (or `prism`, see [PRISM.md](docs/PRISM.md)) |
| Embedding model | `RAGBASE_EMBED_MODEL` | `all-MiniLM-L6-v2` |
| Agent refinement rounds / time budget | `RAGBASE_AGENT_MAX_REVISIONS` / `RAGBASE_AGENT_TIME_BUDGET_S` | `1` / `180` |
| Upload limits | `RAGBASE_MAX_UPLOAD_MB` / `RAGBASE_MAX_UPLOAD_FILES` | `50` / `50` |
| Question length limit | `RAGBASE_MAX_QUESTION_CHARS` | `2000` |
| Tabular Q&A | `RAGBASE_TABULAR_QA` | on |

Chunk size (512), overlap (64), top-k (5), the fused candidate pool (20) and
the reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) are set in `config.py`.

## Evaluation

RAGBase was evaluated on **42 human-reviewed questions** over a synthetic
eight-document corpus, with **three repeats** of each question, comparing
three systems on one local machine:

- **Shared by all three:** the question set and reviewed labels, the corpus,
  the generator model (`llama3.2` 3B via Ollama, same model digest), the
  embedding model and chunking (512 characters, 64 overlap).
- **Different:** the **original pre-upgrade code** (commit `909df31`) ran
  with its own ingestion (no column-aware PDF reading), its historical
  retriever (in practice semantic search only, with the reranker re-ordering
  those five results), free-form citations, and **unseeded** generation.
  **Standard** and **Agentic RAG** share the current ingestion and modern
  hybrid retriever (semantic and keyword search fused, reranked from 20
  candidates) and use **fixed seeds** per repeat, but differ in retrieval
  orchestration and evidence gathering: Standard RAG uses one search and
  five passages; the agent's complex route runs planned and per-document
  searches and can gather up to 12 passages. The agent also adds claim
  checks and refinement, so it makes more model calls.

Every score uses the final labels (v1.3) and scorer (v1.2). What was
measured, each separately:

- **Correctness (keyword proxy)**, per response: the answer contains every
  reviewed required fact and no known-wrong statement. This is not
  human-verified semantic accuracy.
- **Abstention**, per response: declining the unanswerable questions, and
  wrongly declining the answerable ones.
- **Citations**, per response: whether the answer's citations are valid
  (every cited ID refers to a retrieved passage, no invented references, and
  at least one citation unless it declines); separately, what share of the
  cited passages contain the expected evidence from the expected document. The original code's
  free-form citations make these not applicable to it.
- **Retrieval and cost**, per response: share of the expected passages
  among the first five each system retrieved (a fixed budget; the agent may
  gather more); model calls, tokens and latency.
- **Verifier challenge set** (per labelled claim), **tabular security suite**
  (per case), and a calibrated local **LLM judge** used as supporting
  evidence only.

Findings: under the tested conditions **Standard RAG was stronger overall**
and made fewer model calls than the agent, with lower latency. The
benchmark exposed two genuine agent defects — comparison questions searched
documents in alphabetical order, and partly hedged answers were treated as
refusals — which were fixed and re-measured in separately labelled runs.
Label and scoring changes are versioned and reported separately from
application changes ([change log](docs/EVALUATION_CHANGELOG.md)).

### Results

42 questions × 3 repeats = **126 responses per system**; final labels v1.3,
scorer v1.2; `llama3.2` 3B via Ollama on one Apple Silicon Mac. The original
code differs from the other two in retriever, ingestion, citation style and
seeding (see above), so its row reflects the whole historical system, not
just its answering strategy.

| Metric (unit · denominator per system) | Original RAGBase | Standard RAG | Agentic RAG |
|---|---|---|---|
| Correct, keyword proxy (per response · 126) | 0.71 (89/126) | **0.83** (104/126) | 0.79 (99/126) |
| Wrongly declined an answerable question (per response · 102 responses to the 34 fully answerable questions) | 0.09 (9/102) | 0.00 (0/102) | 0.04 (4/102) |
| Expected evidence among the first 5 passages retrieved — a fixed budget; the agent can gather more (mean per-response share · 111 responses to the 37 questions with expected evidence) | 0.86 | 0.97 | 0.98 |
| Model calls per response (mean · 126) | 1 | 1 | 3.4 |
| Latency per response, median / 95th percentile (wall clock · 126) | 1.02 / 1.66 s | 0.82 / 1.42 s | 1.77 / 6.86 s |

- Under these conditions **Standard RAG was stronger overall** and made
  fewer model calls than the agent, with lower latency. At the category
  level, Agentic RAG exceeded Standard RAG only on conflicting-information
  questions in this small benchmark; both modes remain weak on
  multi-document synthesis and false premises.
- **The evaluation found an agent defect**: comparison questions searched
  documents in alphabetical order. After the fix, in a separately labelled
  run scored with the same final scorer, Agentic RAG's keyword-proxy
  correctness rose from **0.66 (83/126) to 0.79 (99/126)**. The pre-fix
  results are preserved.
- **Keyword-proxy correctness** means the response contains every reviewed
  required fact and no known-wrong statement. It is not human-verified
  semantic accuracy, and these figures are specific to this benchmark —
  not a measure of production accuracy or hallucination rates. The claim
  verifier, tabular feature and LLM judge each have documented error rates.

Metric definitions, citation results, categories, the defect and its fix,
verifier, tabular and judge results, and limitations:
**[docs/EVALUATION.md](docs/EVALUATION.md)**.

### Run the benchmark yourself

With Ollama running and `llama3.2` pulled (and a full clone, since the
original baseline is extracted from commit `909df31`):

```bash
python -m evals run --out evals/results/run-a --modes original,standard,agentic --repeats 3 --seed 1000 --labels 1.3 --scoring 1.2
```

This writes the records, `summary.csv` and `report.md` to
`evals/results/run-a` (ignored by git); the full run took about ten minutes
on the Mac used above. A fresh run is an independent measurement: outputs can
differ across machines and Ollama versions. Comparing runs and other options
are in [docs/EVALUATION.md](docs/EVALUATION.md#reproducing).

## CI/CD and testing

**Automatic** — on every pull request and every push to `main`
([`ci.yml`](.github/workflows/ci.yml)), installed with `constraints.txt`:

| Job | What it runs | Models |
|---|---|---|
| Lint | `ruff check .` | none |
| Unit tests | `pytest -m "not integration and not live"` — retrieval, citations, the agent graph, the verifier, the evaluation metrics and harness, tabular security, both Streamlit apps and the showcase restrictions | none; LLM calls use scripted fake models |
| Integration | `pytest -m "integration and not live"` — retrieval and indexing with the real embedding and reranker models, session-isolated uploads | real embedding and reranker models; **no LLM** |

No real LLM runs in automatic CI.

**Manual** — the [live evaluation workflow](.github/workflows/eval-live.yml)
is triggered by hand from GitHub Actions. It installs Ollama on a CPU runner,
runs a 15-question subset (`ci_subset`) with real `llama3.2:3b` inference,
scores it with labels 1.3 and scorer 1.2, and uploads the results as an
artifact. It is labelled "development": it checks that the pipeline works end
to end and does not replace the full local benchmark. Its job summary shows
status, configuration and runtime only.

**Local only** — tests marked `live` need a running Ollama server (and, for
Prism, a configured gateway and key). They are excluded from CI and skip
automatically when the service is unavailable:

```bash
pytest -m live
```

**Deployment** — the public showcase runs from the `showcase-deploy` branch on
Streamlit Community Cloud and redeploys automatically when that branch
changes; see [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md#updating-the-deployment).

## Security and privacy

- **Local by default** — with the Ollama backend, documents and questions
  never leave your machine. Streamlit usage statistics are disabled.
- **Optional Prism gateway** — offered only when configured. Prism's
  guardrails can block prompts, and it caches responses for an hour, so
  document-derived text is held there; RAGBase rejects answers from
  providers outside an allowlist (fail closed). See [PRISM.md](docs/PRISM.md).
- **Untrusted content** — document text is treated as data in every prompt;
  model answers are rendered as escaped text (not Markdown), so injected
  content cannot load remote images or disguise links.
- **No code execution** — spreadsheet questions use allowlisted operations.
- **Uploads** — safe file names, size and count limits, per-session workspaces.
- **Public showcase** — read-only, recorded runs, no inference or tabular
  code loaded; details and residual risks in [DEPLOYMENT.md](docs/DEPLOYMENT.md).

## Known limitations

- Standard RAG outperformed Agentic RAG overall on the benchmark; the agent
  costs more model calls and time.
- Multi-document synthesis, false-premise questions and some two-step
  questions remain weak in both modes.
- The claim verifier is a small model: it rejects some supported claims and
  accepts some incorrect ones. Citations show provenance, not correctness.
- Tabular Q&A answers most but not all questions in its test suite.
- The evaluation uses one small local model, one machine and 42 synthetic
  questions; the keyword proxy and the LLM judge are both fallible.
- Not a production service: no authentication, multi-user deployment or
  persistent storage beyond the local index.

## Repository guide

```
app.py                  Streamlit entry point (local app or showcase)
ui/                     Local app, showcase, answer/citation/trace rendering, theme
agent/                  LangGraph workflow, document selection, claim verifier
retriever.py            Hybrid retrieval, reranking, Standard RAG chain
ingest.py               Loading, column-aware PDF extraction, OCR, chunking, indexing
citations.py            Evidence IDs, citation validation, abstention detection
tabular.py              Allowlisted tabular Q&A
llm/                    Ollama and Prism clients
evals/                  Benchmark corpus, reviewed labels, harness, metrics, reports
showcase/               Public showcase entry point, recorded runs, curated questions
scripts/                Showcase recorder, Prism helpers
tests/                  Unit and integration tests
docs/                   Architecture, evaluation, deployment, evaluation change log, Prism, demo, development log
```

| Document | Contents |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Diagrams of every component |
| [docs/EVALUATION.md](docs/EVALUATION.md) | Evaluation protocol, metric definitions, full results and limitations |
| [docs/DEMO.md](docs/DEMO.md) | A short guided walkthrough |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | Showcase deployment, measurements, security checklist |
| [docs/EVALUATION_CHANGELOG.md](docs/EVALUATION_CHANGELOG.md) | Label, scoring and output versions |
| [docs/PRISM.md](docs/PRISM.md) | Optional gateway backend and its privacy behaviour |
| [docs/DEVELOPMENT_LOG.md](docs/DEVELOPMENT_LOG.md) | Engineering log of each phase and decision |
| [BENCHMARKS.md](BENCHMARKS.md) | Historical latency benchmark (October 2026, earlier code) |

## Stack

Python 3.12 · [LangChain](https://python.langchain.com) · [LangGraph](https://langchain-ai.github.io/langgraph/) ·
[FAISS](https://github.com/facebookresearch/faiss) · [scikit-learn](https://scikit-learn.org) TF-IDF ·
[sentence-transformers](https://sbert.net) embeddings and cross-encoder ·
[Ollama](https://ollama.com) · [Streamlit](https://streamlit.io) ·
[pdfplumber](https://github.com/jsvine/pdfplumber) · [Tesseract](https://github.com/tesseract-ocr/tesseract) ·
pydantic · pytest · ruff · GitHub Actions

## License

MIT — see [LICENSE](LICENSE).
