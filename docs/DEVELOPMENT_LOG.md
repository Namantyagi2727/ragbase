# Development Log

Running record of the agentic upgrade: phase status, technical decisions,
test results and known limitations.

## Phase checklist

| Phase | Scope | Status |
|---|---|---|
| 0 | Repository audit and implementation plan | Done |
| 1A | Foundations: retrieval fusion, chunk IDs, shared citations, LLM adapter, security fixes, tests | Done — merged ([PR #1](https://github.com/Namantyagi2727/ragbase/pull/1)) |
| 1B | LangGraph agentic engine (planner, retrieval tools, synthesis, verifier) | Done — in review ([PR #2](https://github.com/Namantyagi2727/ragbase/pull/2)) |
| 2 | Optional Prism gateway integration | Not started |
| 3 | Evaluation dataset, deterministic metrics, LLM-as-a-judge, CI | Not started |
| 4 | Streamlit integration and public demo | Not started |
| 5 | Documentation and final review | Not started |

## Phase 0 — audit findings (summary)

- "Hybrid" retrieval was effectively FAISS-only: TF-IDF results were appended
  after FAISS's top-k and then truncated away.
- The LLM received chunk text without source metadata yet was asked to cite
  `[filename @ location]`, so citations were invented.
- Tabular Q&A executed LLM-generated Python with `exec()`; the `"read_"`
  line filter was not a security boundary.
- Source snippets and filenames were rendered as unescaped HTML; upload
  names were used as paths without sanitisation; no upload limits.
- No automated tests or CI.

## Phase 1A — foundations

### Baseline

Before any code change, 13 characterisation tests were written against the
original code: **9 passed** (behaviour to preserve: loaders, metadata,
row-block chunking, top-k cap, de-duplication) and **4 failed**, each
encoding a known defect — TF-IDF never reaching results, no source IDs in
the prompt, generated code executing (the test's sentinel file was
written), and prose "how many" questions routed to code execution.

### Changes

| Area | Change |
|---|---|
| Retrieval | Reciprocal-rank fusion of FAISS and TF-IDF over a 20-candidate pool per retriever (`reciprocal_rank_fusion()` is a pure, separately tested function). Reranker now scores the fused pool instead of re-ordering 5 FAISS hits. Source filtering and `sources()` added for the upcoming retrieval tools. |
| Chunk identity | Deterministic `chunk_id` = first 16 hex chars of sha256(source, location, text), ordered suffixes for duplicates. Stored in FAISS metadata, as FAISS docstore IDs and in `chunks_meta`. Indexes built before this change get identical IDs computed at load time. |
| Index manifest | `manifest.json` records embedding model, chunking settings, chunk count and files. The index's embedding model overrides a mismatched UI selection (previously a dimension error). |
| Citations | New `citations.py`, shared by both answer modes: retrieved chunks become evidence items `E1..En`; prompt shows tagged evidence; `validate_citations()` deterministically flags unknown IDs, free-form "[file @ page]" references, uncited claims (coverage metric) and abstentions. |
| Standard RAG | Same retrieve → single LLM call structure, now with evidence IDs, an explicit abstention phrase and an instruction to ignore instructions inside documents. Output includes the evidence list and citation report. Remains an LCEL chain. |
| LLM adapter | `llm/` package: `ChatModel` protocol, direct Ollama backend (HTTP, native JSON-schema output, typed timeout / unavailable / response errors), scripted `FakeChatModel` (flagged `is_mock`), and `complete_structured()` with pydantic validation and one bounded repair attempt — a failed parse returns `None`, never a default decision. |
| Tabular Q&A | `exec()` removed. The model proposes a `TabularQuery` (allowlisted operation, column, ≤5 typed filters, optional group-by, bounded limit); it is validated against the real schema and executed by fixed pandas code. Routing now requires a table name, or an aggregate word plus a real column name. Can be disabled with `RAGBASE_TABULAR_QA=false`. |
| Upload safety | `safe_filename()` (basename only, character allowlist, no dotfiles, extension allowlist, length cap); per-file size cap (50 MB, also set in Streamlit config); per-upload file cap in the UI; PDF page and OCR page caps; image decompression-bomb limit; one unreadable file no longer aborts a build. |
| UI | All HTML-rendered values escaped. Sources panel shows cited evidence with `E#` badges, then uncited retrieved chunks; warns on invalid or missing citations. Index-load failures are shown instead of silently ignored; chunk count restored after auto-load. Visual design unchanged. |
| Config | Environment-variable overrides for backend, model, paths, limits and the tabular toggle. |
| Dependencies | Removed unused `langchain` and `langchain-ollama`; declared `langchain-core`, `httpx`, `pydantic`. Dev: `pytest`, `ruff`. |

### Decisions

- **Prompt format.** Evidence is rendered as `<doc id="E1" source=… location=…>`
  blocks with the citation instruction repeated after the question. With
  `llama3.2` (3B), an earlier `[E1] source: … | location: …` header format
  caused the model to echo the header instead of answering; the tagged
  format fixed this in a side-by-side check on the same questions.
- **Ingestion file cap.** The upload file cap applies only to the UI. A first
  version applied it inside ingestion and silently truncated the
  1,200-document benchmark corpus; caught by the benchmark, fixed, and
  covered by a regression test.
- **Legacy indexes.** Content-derived IDs mean old indexes need no
  migration; IDs are computed on load and match what a rebuild produces.
- **Prism.** Unchanged in this phase, as agreed.

### Tests

| Suite | Count | Needs |
|---|---|---|
| Unit (fakes / mocked HTTP, no models) | 108 | nothing |
| Integration (real MiniLM embeddings + cross-encoder) | 4 | Hugging Face model download on first run |
| Live (real Ollama `llama3.2`) | 3 | local Ollama; skipped otherwise |

All 115 pass locally (counts include the quantity fix below); the live
tests passed on every repeated run. Lint
(`ruff`) clean. Includes regression tests for malicious tabular inputs
(code replies, dunder-import column names, unknown operations, extra fields,
injected filter values, unbounded limits) and an AST check that `exec`,
`eval`, `compile`, `__import__`, `DataFrame.query` and `DataFrame.eval` do
not appear in `tabular.py`, `retriever.py` or `app.py`.

### Benchmarks

Re-run on the same machine and corpus against the original code — see
[BENCHMARKS.md](../BENCHMARKS.md). Retrieval latency is essentially
unchanged (reranking +10–20ms for scoring 20 candidates instead of 5).
End-to-end p95 rose to ~3.4s because cited answers are longer; median
remains 0.6–1.5s.

### Fix: quantity questions counted rows instead of totalling

With `llama3.2`, "How many units were sold in the EU?" was planned as
`count(units)` (2 rows) instead of `sum(units)` (15).

- Prompt guidance distinguishing row counts from quantity totals was added,
  but on its own it did **not** fix the case: the model still chose `count`
  on 3 of 3 attempts.
- The fix that works is deterministic: `reconcile_operation()` rewrites
  `count` to `sum` when the question asks "how many / total / number of /
  how much" followed by a numeric column name. It is skipped for row-count
  phrasing ("how many rows have…"), identifier columns, non-numeric columns,
  and questions naming more than one quantity column. The rewrite is shown
  with the answer.
- With the real model: the EU question returns `sum` = 15 on 3 of 3
  attempts; "How many rows are from the EU?" stays `count`; other operations
  are unchanged. Covered by 11 unit tests and a live regression test.

### Known limitations after Phase 1A

- **Small-model tabular planning** is corrected for the count/sum case only;
  other misreadings of a question remain possible. The executed query is
  always shown with the answer. To be measured in Phase 3.
- **Answer quality is not yet measured.** Citation validity is checked
  deterministically; whether answers are *correct* awaits the evaluation
  suite.
- **Shared upload directory.** `uploaded_docs/` and the index are global and
  accumulate across builds; each build re-indexes every file in the
  directory. Session isolation is required before public uploads (Phase 4).
- **Pickle-based index files.** Safe only for indexes this app builds; the
  demo must never load index files from elsewhere.
- **UI fonts** are loaded from Google Fonts, so the interface itself makes
  one external request.
- **README** still describes the pre-upgrade system; updated in Phase 5.

## Phase 1B — LangGraph agentic engine

### Architecture

```
plan ──simple──▶ standard ───────────────────────────────▶ finalize
  └──complex──▶ retrieve ▶ synthesize ▶ verify ──accept──▶ finalize
                   ▲            ▲          │
                   └── refine ──┘◀─revise──┘   (≤ max_revisions, time left)
```

| Component | Kind | Role |
|---|---|---|
| Planner | Model decision (structured `QueryPlan`) | Chooses the simple or complex route, writes up to 3 sub-queries, targets documents, flags comparisons |
| Retrieval tools | Deterministic | `hybrid_search` (validated query, k, known-source filter) and `list_documents`; reuse the Phase 1A retriever and reranker. No filesystem, shell or network tools |
| Synthesizer | Model generation | Answers from numbered evidence, cites `[E#]`, states conflicts, abstains when unsupported |
| Verifier | Deterministic checks + model decision (structured `VerifierOutput`) | Citation validity; numbers in each cited claim present in its cited passages; per-claim verdicts from the model; recommends accept / revise with a follow-up search |
| Refinement | Deterministic control | At most `max_revisions` (default 1) rounds: optional targeted search, then re-synthesis with verifier feedback |

The planner and verifier make distinct model-driven decisions; retrieval is a
tool layer, not an extra persona. The simple route is the Standard RAG path
(one search, one answer call).

The engine lives in `agent/` and has no Streamlit dependency;
`python -m agent "question"` runs it against a built index and prints the
trace.

### Verification is separate from citation validity

`Verification` records independently:

- `citation_ok` — deterministic (`CitationReport.ok`): cited IDs exist, no invented references.
- `grounding` — `verified`, `partially_verified`, `unsupported`, `not_verified` or `skipped`.
- `llm_status` — `completed`, `failed` or `skipped`.

A verifier call that fails, returns invalid JSON, or returns no usable
verdicts gives `not_verified`; it is never reported as verified. Verdicts
that reference unknown evidence or non-existent claims are discarded and
reported. Final status:

| Status | Meaning |
|---|---|
| `completed` | Verified (or a valid abstention) |
| `completed_unverified` | Citations valid but claims not verified (simple route, or verifier failed) |
| `qualified` | Unresolved problems after the allowed revisions; answer carries a visible warning |
| `failed` / `timeout` | Backend error or time budget exhausted |

### Bounds and failure handling

Three independent limits: `max_revisions`, a per-run time budget (default
180 s; no revision starts with < 15 s left), and LangGraph's recursion limit.
Backend errors in planning or synthesis end the run as `failed`; verifier
errors degrade to `not_verified`. Planner output that fails validation falls
back to the simple route, and the fallback reason is stored in the result
and trace.

### Findings from live runs with `llama3.2` and fixes

| Observed | Fix |
|---|---|
| Verifier omitted claim verdicts (correctly reported as not verified, but useless) | Verdicts required (≥1) in the schema, which Ollama's constrained decoding enforces |
| Verifier named claims by number instead of text | Verdicts reference `claim_number`, mapped back to the answer's claims in code |
| Planner sent simple questions down the complex route and put file names inside queries | Planner examples, "most questions are simple" guidance, temperature 0, file-name prefixes stripped from queries |
| Years from source file names flagged as unsupported numbers | Numbers in a cited passage's source name count as available |
| A partial answer containing "the documents do not mention X" was treated as a full abstention, skipping verification | An answer is an abstention only if it cites nothing; absence statements are excluded from claim verification and reported |
| Two trace events shared one step number | Unique step numbering |

Planner routing on seven test questions was identical across three repeated
runs (temperature 0): factual lookups → simple; comparisons and multi-part
questions → complex.

### Live examples (fixture corpus, `llama3.2`, after fixes)

| Question type | Route | LLM calls | Searches | Status | Time |
|---|---|---|---|---|---|
| Single fact | simple | 2 | 1 | completed_unverified | 1.0 s |
| Compare two policy versions | complex (per-document searches) | 3 | 3 | completed | 5.4 s |
| Conflicting deadlines across versions | simple | 2 | 1 | completed_unverified | 0.9 s |
| Two-part question | complex | 3 | 1 | completed | 2.3 s |
| Unanswerable | simple | 2 | 1 | completed (abstained) | 0.6 s |

Standard RAG produced equivalent answers on the simple questions in about
the same time. On the comparison it combined both years' figures under a
single citation, whereas the agent cited each policy separately. This is
an observation from one run, not a measured improvement; Phase 3 will
compare the modes on a labelled dataset.

On a real two-column PDF (driver's manual), a multi-part question used the
refinement loop: the verifier's follow-up search found the passage with the
under-21 limit, but the 3B model's revised answer still contradicted itself,
and the run ended `qualified` with the problem claim listed.

### Tests

| Suite | Count |
|---|---|
| Unit (no models) | 157 |
| Integration (real embeddings + reranker) | 4 |
| Live (local Ollama) | 6 |

All 167 pass locally; CI runs lint and the unit suite on every pull request.

### Known limitations after Phase 1B

- **Small-model quality.** `llama3.2` (3B) drafts answers that can be
  internally inconsistent; the verifier catches some but not all problems.
  The verifier can be pointed at a larger model with
  `RAGBASE_AGENT_VERIFIER_MODEL` (not yet evaluated).
- **Absence statements are unverified.** A wrong "the documents do not
  state X" is not caught by claim verification.
- **Simple-route answers are never claim-verified** (by design, to keep
  simple questions to two model calls); they are reported as
  `completed_unverified`.
- **Multi-column PDFs** are extracted with interleaved columns by
  pdfplumber, which degrades retrieval on such documents.
- **No UI integration yet** — Agentic mode is available through code and
  the CLI; the Streamlit mode switch is Phase 4.
- **Cost of agentic mode:** 2 model calls on the simple route, typically 3–5
  on the complex route (at most 8 with one revision and every structured-output
  repair attempt used), versus 1 for Standard RAG.
