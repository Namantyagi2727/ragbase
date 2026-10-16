# Development Log

Running record of the agentic upgrade: phase status, technical decisions,
test results and known limitations.

## Phase checklist

| Phase | Scope | Status |
|---|---|---|
| 0 | Repository audit and implementation plan | Done |
| 1A | Foundations: retrieval fusion, chunk IDs, shared citations, LLM adapter, security fixes, tests | Done — merged ([PR #1](https://github.com/Namantyagi2727/ragbase/pull/1)) |
| 1B | LangGraph agentic engine (planner, retrieval tools, synthesis, verifier) | Done — merged ([PR #2](https://github.com/Namantyagi2727/ragbase/pull/2)) |
| 2 | Optional Prism gateway integration | Done — merged ([PR #3](https://github.com/Namantyagi2727/ragbase/pull/3)) |
| 3 | Evaluation dataset, deterministic metrics, LLM-as-a-judge, CI | Done — merged ([PR #4](https://github.com/Namantyagi2727/ragbase/pull/4)) |
| 4 | Streamlit integration and public demo | Done — merged ([PR #5](https://github.com/Namantyagi2727/ragbase/pull/5), closeout [PR #6](https://github.com/Namantyagi2727/ragbase/pull/6)); live at https://ragbase-showcase.streamlit.app/ |
| 5 | Documentation, portfolio and reproducibility | In progress ([PR #7](https://github.com/Namantyagi2727/ragbase/pull/7)) |

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
| `completed` | Every claim verified, or an abstention confirmed by a check / with no evidence retrieved |
| `completed_unverified` | Citations valid but verification incomplete or not run (simple route, verifier failure, missing verdicts, unconfirmed absence statements) |
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

### Verification-correctness patch (review follow-up)

Review found that one accepted verdict could mark a multi-claim answer
`verified`, that absence statements bypassed verification while the answer
could still be `verified`, and that any abstention phrase counted as a
correct outcome.

- **Complete coverage.** `verified` now requires every checkable claim to
  have exactly one valid verdict, and each "supported" verdict must include
  evidence the claim itself cites. Missing, duplicate, invalid or
  unattributable verdicts, uncited claims, and claims beyond the
  verification limit (12) give a new grounding value, `incomplete`.
  Uncited substantive claims also trigger the existing bounded revision.
- **Absence statements** ("the documents do not specify X") trigger the
  existing follow-up search (one round, no new loop). If they remain, the
  result stays `incomplete`, and the answer carries a note that they reflect
  the retrieved passages (and the targeted search, if it ran), not the
  whole collection.
- **Abstentions** are labelled `no_evidence`, `confirmed` (the verifier
  checked the retrieved evidence and found no answer), `evidence_may_answer`
  (triggers a revision; `qualified` if none remains), `not_checked` (simple
  route) or `failed`. Only `no_evidence` and `confirmed` count as
  `completed`. On the complex route the check uses the verifier's slot, so
  an abstention with retrieved evidence now costs one model call that
  previously was skipped.

17 scripted regression tests cover these rules, including a reproduction
of the driver's-manual failure (absence claim → targeted search → passage
found → re-verified).

**Live check after the patch** (`llama3.2`, two runs per question, fixture
corpus plus the driver's manual): no run exceeded one revision or five model
calls. Statuses: simple questions `completed_unverified` (by design); the
two-part question `verified` both runs; the comparison `qualified` both runs
— in the inspected run the answer wrongly said the 2023 policy had no hotel
rule (correctly caught as an unconfirmed absence statement), but the
verifier also marked a correct 2024 claim unsupported, a model false
negative.

**Verifier false positive found.** Across 8 further driver's-manual runs, one
answer was marked `verified` although it was wrong: it stated the under-21
limit is 0.08, citing a passage about drivers *over* 21. All deterministic
checks pass on that claim (valid citation, number present in the cited
text), and both `llama3.2` and `gemma3:12b` judged it supported on every
re-check — the passage's two-column extraction interleaves the "over 21"
text with the DWI lines. `verified` therefore means "passed the
deterministic checks and the model's claim check", not "correct".

A final deterministic correction: an extra malformed verdict (unknown
evidence ID or non-existent claim number) alongside otherwise complete
coverage previously still allowed `verified`; it now yields `incomplete`.

### Tests

| Suite | Count |
|---|---|
| Unit (no models) | 176 |
| Integration (real embeddings + reranker) | 4 |
| Live (local Ollama) | 6 |

All 186 pass locally; CI runs lint and the unit suite on every pull request.

### Known limitations after Phase 1B

- **Model verification can be wrong in both directions.** The verifier has
  produced a false positive (wrong answer marked `verified`) and false
  negatives (correct claims marked unsupported). Deterministic checks catch
  invented citations, missing coverage and numbers absent from cited text,
  but not semantic misreadings. Measuring verifier precision is Phase 3
  work. *(Follow-up: model quality.)*
- **Absence statements cannot be confirmed.** They are searched for once and
  otherwise reported as unconfirmed; the system never claims information
  is absent from the whole collection.
- **Simple-route answers are never claim-verified** (by design, to keep
  simple questions to two model calls); they are reported as
  `completed_unverified`.
- **Multi-column PDFs** are extracted with interleaved columns by
  pdfplumber, which degrades retrieval and contributed to the verifier false
  positive above. *(Follow-up: ingestion.)*
- **No UI integration yet** — Agentic mode is available through code and
  the CLI; the Streamlit mode switch is Phase 4.
- **Cost of agentic mode:** 2 model calls on the simple route, typically 3–5
  on the complex route (at most 8 with one revision and every structured-output
  repair attempt used), versus 1 for Standard RAG.

## Phase 2 — Prism gateway integration

Full details: [docs/PRISM.md](PRISM.md). Prism itself was not modified.

### Implementation

- `llm/openai_compat.py`: `PrismChatModel`, selected with
  `RAGBASE_LLM_BACKEND=prism`. Direct Ollama stays the default; there is no
  fallback between backends. Temperature is always sent; structured output
  uses explicit JSON instructions (Prism drops schema fields) with the
  existing validation and one repair attempt; the API key is read from the
  environment and redacted from errors and `repr`.
- **Provider verification, fail closed:** a response is accepted only if
  `prism_metadata.provider_used` is in `PRISM_ALLOWED_PROVIDERS` (default
  `ollama`). Other providers (including Prism's mocks), cache hits (origin not
  recorded by Prism) and missing metadata raise `ProviderRejectedError` and
  the text is discarded. Mock content/ID detection is a second check.
- Typed errors for guardrail blocks (with reason), auth, rate limit, missing
  route, provider failure, timeout and connection failure.
- Agent results record per-call backend, inference mode
  (`local_direct` / `prism_local` / `prism_remote`) and only the gateway
  fields actually reported (provider, fallback, cost, latency, tokens).
- The factory keeps the Ollama model name and the Prism virtual model
  separate, so an Ollama name is never sent to Prism.

### Findings

- **Guardrails block every evidence prompt.** RAGBase's evidence IDs (`E1`,
  `E2`, …) match Presidio's `US_DRIVER_LICENSE` recognizer at score 0.3, and
  Prism blocks on any match: 100% of standard-RAG prompts were blocked on the
  travel-policy fixtures, the driver's manual and the novel corpus. With the
  IDs rewritten to `E-1` style (measurement only), documents without personal
  data pass (travel policies 0/6 blocked), but documents mentioning people are
  still blocked (driver's manual 3/6, novels 29/30). Measured with Prism's
  own `scan_prompt` run in-process.
- **Context length.** Prism's upstream request carries no context option;
  Ollama's default was 4096 tokens on the test machine, the same as RAGBase's
  direct setting. Planted-fact probes behaved identically on both paths up
  to ~4,000 tokens; beyond the context size both truncate silently to
  ~2,050 tokens. RAGBase's default prompts are around 2,000 tokens at most.
- **Cache hits are always rejected** under the fail-closed rule, so repeating
  an identical request within Prism's one-hour cache window fails.

### Completion patch: real-gateway verification

**Citation format.** Evidence citations changed from `[E1]` to `[1]`, with the
id attribute last in each evidence tag. `E1` matched Presidio's
`US_DRIVER_LICENSE` recognizer; `E-1`, `doc1`, `EV1` and others were tagged
`PERSON` by spaCy in some contexts; `#1` scanned clean but cited worse with
`llama3.2` (coverage ~0.5). Numbered citations scanned clean across IDs 1–60
and every prompt pattern, and cited at least as well as before (standard RAG
coverage 0.86–0.87 vs 0.82–0.83, citation checks 20/20 both). Older styles in
model output are normalized. With numeric IDs the verifier left
`evidence_ids` empty, so verdicts now require them; agent outcomes then
matched the old format. Chunk IDs and indexes are unchanged.

**Structured output.** Against the real gateway every structured call was
blocked: spaCy tagged "JSON", "Respond" and raw JSON-schema text as `PERSON`.
The adapter now sends a plain-language field list; validation is unchanged.
This was missed by the earlier in-process scan, which ran on prompts before the
adapter's instruction was added — the measurement script now scans prompts
exactly as sent.

**Real stack.** Prism's app, Postgres, Redis and Jaeger ran in Docker with a
compose override (outside the Prism repo) pointing the app at Ollama on the
host; a test team and key were created through Prism's admin API, with the key
stored only in RAGBase's gitignored `.env`. Verified over HTTP:

| Check | Result |
|---|---|
| Ollama response through Prism, attributed (`provider_used=ollama`, `prism_local`, tokens, cost 0, gateway latency) | pass |
| Standard RAG with valid citations (3 questions in the demo + 1 test) | pass |
| Agentic multi-document questions (1 test + 2 demo questions; complex route, both policies retrieved, citations valid) | pass; demo answers ended `qualified` (model contradictions) |
| Identical repeated request → cache hit without provenance → rejected | pass |
| Prism `mock-fast` route (mock `openai`) → rejected | pass |
| Synthetic names → guardrail block (PERSON) | pass |
| Policy text without PII passes guardrails | pass |

**Latency** (3 Standard RAG questions, matched answer lengths): LLM call
0.51–1.06 s via Prism vs ~0.31 s direct; Prism-reported gateway latency
426–978 ms. Median end to end 0.99 s vs 0.35 s.

**Other findings.** Random letter-digit tokens in a question (hex IDs, SKUs)
can be blocked as `US_DRIVER_LICENSE`. Prompted JSON was valid on first
attempt in 15/16 calls vs 16/16 with native schema; one live verifier output
failed both attempts and was recorded as `not_verified`.

### Tests

| Suite | Count |
|---|---|
| Unit (incl. Prism adapter, mocked-Prism end-to-end, citation-format regressions) | 239 |
| Integration (real embeddings + reranker) | 4 |
| Live (local Ollama) | 6 |
| Live Prism (real gateway) | 7 — all passed against the real gateway; on re-runs the two fixed-prompt tests skip because Prism answers them from cache |

### Remaining limitations

- Documents mentioning people (and other detected PII or injection-like text)
  are blocked through Prism by design.
- Identical repeated requests within Prism's one-hour cache window are rejected.
- Context length through Prism depends on the Ollama server default.
- Prompted structured output is slightly less reliable than native decoding.
- Prism keeps full responses (document-derived) in Redis for an hour.

### Possible later Prism changes (not made)

Record the originating provider on cache entries; a per-key guardrail policy
for document-grounded clients; a context-length pass-through; native
structured output.

## Phase 3 — Evaluation framework (closed)

### Two-column PDF extraction

`ingest.page_text()` reads a page column by column only when the layout is
clearly two-column: an empty vertical band in the middle 30-70% of the page
(every line leaving a gap of at least 12 pt there), no more than 15% of lines
crossing it (full-width headings), and a run of at least three lines with
prose (4+ words per segment on average) on both sides. Table rows (several
cells on one side) and full-width lines stay whole; a vertical gap above 2.5x
normal line spacing ends a column block. Every other page keeps pdfplumber's
original output.

Tests were written first (4 of 6 failed before the change). On real files:
212 single-column PDFs produce identical text; on the two-column driver's
manual 66 of 85 pages switch, the penalties table page does not, and the
under-21 zero-tolerance passage reads in order. Known side effect: a
right-aligned running header moves to the start of the right column.
**Previously indexed two-column PDFs must be re-ingested** to benefit; their
chunk text, and therefore chunk IDs, will change.

### Synthetic corpus and labels

- `evals/build_corpus.py` writes 8 fictional documents (TXT, DOCX, single- and
  two-column PDF) and a CSV. A SHA-256 manifest is checked in CI; rebuilds made
  seconds apart were byte-identical after fixing DOCX zip timestamps (an
  earlier "identical" result had only held within the zip format's 2-second
  timestamp resolution).
- 42 RAG cases across 10 categories, each expected fact with an exact evidence
  quote; 3 required regression cases (under-21 / over-21 / second violation).
- 20 verifier challenge pairs with per-claim labels, including the Phase 1B
  false positive in clean and interleaved-extraction form.
- 9 tabular cases; expected values computed from the CSV by the executor.
- All labels are `proposed`. CI checks schema, sources, quotes, checksums and
  computed values — consistency, not correctness.
- Forbidden-fact patterns are evaluated per sentence with negation skipping;
  a probe test guards against false matches on correct contrast sentences.

### Label review corrections

Requested after the first review; no other labels changed (verified by diffing
the label files against the previous commit).

| Case | Change |
|---|---|
| VC04 | Claim replaced with a false *maximum* ("the maximum rate eligible for automatic approval is $200"); the original claim ("stays up to $200 are approved") was true when the limit is $250 |
| VC21 (new) | True within-range counterpart ("a $200 night is approved automatically"), paired with VC04 in a regression test |
| VC15 | Claims now carry two labels — support by the cited evidence and factual truth; answer labelled `correct_fact_wrong_citation`, counted separately from factually wrong answers |
| C03 | Reference answer reduced to the current director requirement, matching its evidence |
| R01, R03 | Evidence quotes extended to include the sentence naming the under-21 group |
| TB08, TB09 | Explicit passing outcomes and security checks; verified through sandboxed file state and patched execution boundaries |

Every evidence quote is now also checked to sit inside a single chunk of the
indexed corpus (default chunking), not just in the extracted text.

### Evaluation infrastructure

- `python -m evals run|report|verifier|tabular|judge` (see `evals/__main__.py`).
- One index per run from the synthetic corpus; fixed-budget retrieval recorded
  once per question and shared by all modes; per-repeat Ollama seeds;
  `config.json` with corpus checksum, commit, models, settings, environment.
- Original-code baseline (`909df31`) runs from a `git archive` in an isolated
  interpreter (`python -I`), importing no current modules and never the
  original executable tabular path; it is unseeded (its own `OllamaLLM`).
- Metrics keep citation validity, citation relevance, keyword-proxy correctness
  and faithfulness proxies separate; agentic results are split by actual
  planner route; verifier confusion keeps undecided cases separate.
- Optional judge (`gemma3:12b`): blind to mode, configurable items and repeats,
  `--estimate` projects runtime first.
- Official runs are refused while any selected label is unapproved.
- A development dry run (2 cases, not a result) caught Ollama being stopped;
  live runs now preflight the server and model.

### CI

- `ci.yml`: lint; unit tests (now including dataset, metric, tabular-security
  and mocked harness smoke tests); a separate integration job with real
  embedding and reranker models and no LLM.
- `eval-live.yml`: manual trigger only, CPU runner, bounded subset, 90-minute
  timeout, no secrets or paid inference, no judge; records elapsed time and
  peak memory. Not yet run: a manually triggered workflow can only be
  dispatched once the file is on the default branch.

### Initial official benchmark: surprising findings

The first official run (commit `a83197f`, results kept local) completed with
no errors. Its findings led to the corrective work below; the results remain
the pre-fix baseline and were not overwritten.

- **Comparison-mode bug in the agent.** For comparison questions the agent
  searched `list_documents()[:max_compare_docs]`: the first five documents in
  alphabetical order. With eight documents in the corpus the two travel
  policies were never searched, so every comparison-mode answer failed and
  most of the agent's false abstentions came from this. The Phase 1B tests
  used three documents and never reached the cap. The abstention check
  "confirmed" these abstentions because it only sees retrieved evidence.
- **Keyword-proxy false pass (FP01).** Every mode wrote "the policy reduced
  the allowance from $45 to $60", accepting the false premise; the forbidden
  pattern only matched the passive voice.
- **Judge leniency.** The judge passed incomplete answers whose claims were
  true, and passed a false abstention because the retrieved passages lacked
  the answer.
- **Verifier challenge counts** were summed over temperature-0 repeats, which
  are not independent cases.

### Corrective work (approved after the initial review)

- Comparison-mode document selection (`agent/sources.py`): documents named
  in the question or by the planner first; otherwise one bounded relevance
  discovery search; passages interleaved by source before the evidence cap;
  refinement reuses the selection; limitations recorded in the trace and in
  benchmark records. Regression tests use nine documents, irrelevant ones
  sorting first, and similarly named policy versions.
- Label revisions versioned separately from the approved labels
  (`evals/label_revisions.jsonl`): FP01 approved; T01, V05, C01 proposed for
  review. Scoring 1.1 adds optional facts and reproduces the 1.0 scores
  exactly under labels 1.0.
- Reports can rescore saved answers into a separate directory at a chosen
  label version; `python -m evals compare` scores the initial and post-fix
  results with one scoring version and shows the scorer's effect separately.
- Judge rubric 2 and 13 calibration answers (see
  `docs/EVALUATION_CHANGELOG.md`).
- A post-fix run of Standard and Agentic (same corpus, model digest,
  configuration and seeds) is stored separately; its results are pending
  review and are not claimed anywhere.

### Completion pass (approved after the post-fix review)

- Abstention classification: an answer that states facts plus "the evidence
  does not specify ..." (benchmark case M03) is a substantive answer, not a
  full abstention; uncited facts are still reported and fail the citation
  check; sentences mixing a fact with an absence statement go to claim
  verification. This changed agent behaviour, so a separately labelled
  controlled run (`final-2026-10-15`) was made with the same configuration
  and seeds. Standard RAG answers were byte-identical; 17 of 126 Agentic
  answers changed, all through the modified verification path.
- Labels 1.2 (T01, V05, C01 contextual facts optional, approved) and 1.3
  (explicit old → new changes for a narrow comparison-consistency check);
  scoring 1.2. Saved answers were rescored into separate directories; no
  earlier result file changed (checked by file hashes).
- Judge rubric 3 for partially answerable questions, with 22 calibration
  answers. It fixed the partial-answer false fails of rubric 2 but passed two
  known-incorrect calibration answers that rubric 2 failed; the rubric was not
  tuned further. Earlier judgments are kept.
- Known limitations are listed in `docs/EVALUATION_CHANGELOG.md`. Standard
  RAG still outperforms Agentic RAG overall on the reviewed benchmark.

## Phase 4 — Streamlit UI and public showcase (closed)

Decisions (approved): Streamlit Community Cloud as the first target for a
read-only public showcase with curated synthetic documents and genuinely
recorded runs; a fully functional local application with live Ollama and
optional Prism; no deployment without separate approval.

Checklist:

- [x] CI cleanup: provenance, duplicate Ollama server, explicit label/scoring
      versions, no scores in the public job summary, public artifact
      documented, actions updated to Node 24 versions
- [x] UI refactor preserving the Neural Terminal identity (stylesheet moved
      unchanged to `ui/theme.css`; `app.py` is a thin entry point)
- [x] Standard / Agentic mode selector with execution summaries
- [x] Citation and source inspection (cited vs retrieved-but-uncited, escaped)
- [x] Session-isolated uploads (local mode); provider and privacy panel
- [x] Methodology and evaluation section (no numbers)
- [x] Tests (scripted models, AppTest) and live local UI checks with Ollama — Checkpoint A
- [x] Public showcase mode with recorded executions and code-level restrictions
- [x] Resource profiling, pinned deployment dependencies, local preview — Checkpoint B
- [x] Deployment: https://ragbase-showcase.streamlit.app/ (approved; details in `docs/DEPLOYMENT.md`)

Notes:

- Live checks: Standard and Agentic answers through real Ollama in the local
  app (comparison question: documents searched were the two travel policies;
  the verifier flagged figures whose citations pointed at the wrong passage).
- Model answers are now rendered as escaped text rather than Markdown, so an
  injected document cannot make the browser fetch remote images.
- Streamlit's upload endpoint accepts files for any active session even
  without an upload widget; the showcase never reads them, and the size cap
  is a deployment decision.
- Showcase runs were recorded at commit `1aef28a` (`llama3.2:3b`, seed 1001)
  and are stored as generated; curator notes are separate.

### Deployment and closeout

- Merged into `main` as `f5b6d87` (PR #5, merge commit). Deployed to
  Streamlit Community Cloud from `showcase-deploy`: `main` plus two root
  config overrides, `maxUploadSize = 1` (`4d293a2`) and `headless = true`
  (`569162e`).
- The first deployment failed its health check. The build was correct
  (showcase dependencies only, Python 3.12.15), but with `main`'s
  `headless = false` Streamlit waited at its first-run email prompt and never
  started. Reproduced locally; fixed with the deployment-only `569162e`.
- Ten live acceptance checks passed as an anonymous visitor (recorded in
  `docs/DEPLOYMENT.md`): all ten comparisons, citations, traces, documents,
  methodology, disclosures and provenance, no inputs or uploads, no secrets
  or private documents, a successful redeploy, desktop and mobile layouts.
- Not measured: platform memory, CPU and exact cold-start time (owner-only
  view); not estimated. A live probe of the 1 MB upload cap was
  inconclusive; the cap was verified locally on the same Streamlit version.
- Tests at closeout: lint clean; 431 unit tests under Streamlit 1.37.1 and
  1.66.0; integration 11 passed, 7 skipped (live Prism not configured); CI
  green on `ui-demo` (`8e6a4b5`) and on `main` (`f5b6d87`).
- Optional future polish: the agent-trace header is cramped at narrow
  desktop widths (around 1,100 px).
- Future approved changes reach the showcase by merging `main` into
  `showcase-deploy` (merge commit, no rebase or force-push) and keeping the
  two overrides; see *Updating the deployment* in `docs/DEPLOYMENT.md`.

## Phase 5 — Documentation, portfolio and reproducibility (in progress)

Goal: present RAGBase accurately for GitHub, recruiters and interviews. No
changes to Prism, the agent, showcase recordings, approved labels or the
deployment without separate approval. Numerical results are published only
after the proposed presentation is approved.

Checklist:

- [x] Audit of README and docs (findings in PR #7)
- [x] Architecture document with Mermaid diagrams drawn from the code
      (`docs/ARCHITECTURE.md`; all diagrams render with Mermaid 11)
- [x] README rewrite (value proposition, recorded-showcase disclosure, real
      screenshots, features, architecture, install, evaluation methodology
      and CI overview, security and limitations, navigation) — no numerical
      results yet
- [x] Screenshots from the deployed showcase and the local app (live
      Ollama runs) in `docs/images/`; outdated screenshots removed
- [ ] Public evaluation results — review gate: numbers approved first
- [x] Demo walkthrough (`docs/DEMO.md`)
- [x] Reproducibility: `constraints.txt` (pinned, all platforms, Python
      3.12) used by the documented install and by CI; clean-environment
      install and full test run; accurate badges including CI; link check
- [x] Historical latency benchmark retitled and corrected; its scripts
      validated against the current code
- [ ] Resume, portfolio and LinkedIn text (kept outside the repository)

## Carried forward to later phases

### Required evaluation regression cases (Phase 3)

| Case | Source | Expected behaviour |
|---|---|---|
| Under-21 blood-alcohol limit misattributed | Live run, Phase 1B: answer stated "the limit for drivers under 21 is 0.08", citing a passage about drivers over 21; marked `verified` by `llama3.2`, and `gemma3:12b` agreed on re-check | Must not be reported as `verified`; the evaluation must measure verifier precision on this and similar misattributed claims |

### Ingestion follow-up (before quality benchmarks or demos)

Two-column PDFs are extracted by pdfplumber with interleaved columns, which
degraded retrieval and contributed to the false positive above. Investigate
column-aware extraction before any affected PDF is used in quality
benchmarks or a public demonstration.
