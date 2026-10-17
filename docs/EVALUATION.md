# Evaluation

How RAGBase was evaluated and what was found. These results are specific to
this benchmark and these conditions. They are not a general accuracy,
reliability or hallucination-rate claim.

## Protocol

| | |
|---|---|
| Questions | 42 human-reviewed cases in ten categories: 34 fully answerable, 3 partially answerable, 5 unanswerable |
| Corpus | Eight synthetic policy documents for a fictional company (including a two-column PDF page), version 1.0.0, checksummed |
| Repeats | 3 per question: **126 responses per system** |
| Scoring | Final labels **v1.3** and scorer **v1.2** for every figure here ([change log](EVALUATION_CHANGELOG.md)) |
| Generator | `llama3.2` 3B via Ollama, same model digest throughout, application-default temperature |
| Hardware | One local Apple Silicon Mac; latency is not a hosting benchmark |

### What the three systems shared, and what differed

| | Original RAGBase | Standard RAG | Agentic RAG |
|---|---|---|---|
| Code | Pre-upgrade commit `909df31`, run unchanged in an isolated interpreter | Current | Current |
| Questions, corpus, generator model | Shared | Shared | Shared |
| Embedding model, chunking (512 / 64) | Shared | Shared | Shared |
| Ingestion | Its own (no column-aware PDF reading) | Current (column-aware PDF reading) | Current |
| Retrieval components | Historical: semantic top 5 (keyword results never reached the output), reranker re-orders those 5 | Modern hybrid retriever: semantic + keyword search fused (20 candidates each), cross-encoder reranking | Same modern hybrid retriever as Standard |
| Retrieval orchestration | One search per question | One search per question | Simple route: one search. Complex route: planned sub-queries (up to 3) and, for comparisons, separate searches of selected documents (up to 5), plus at most one refinement search |
| Evidence per answer | 5 passages | 5 passages | Simple route: 5 passages. Complex route: 4 passages per sub-query search, or 2 per selected document for comparisons; deduplicated and capped at 12 passages |
| Citations | Free-form text references | Numbered evidence IDs | Numbered evidence IDs |
| Seeds | Unseeded | Fixed per repeat (1001–1003) | Fixed per repeat (1001–1003) |
| Model calls | One | One | Planner plus answer; on the complex route also claim verification and at most one refinement |

Because the original code differs in retriever, ingestion, citation style and
seeding, its results reflect that whole historical system. Standard and
Agentic RAG share the modern retrieval components, but differ in retrieval
orchestration, evidence gathering, model calls and claim-verification
behavior.

## Metric definitions

| Metric | Unit | Denominator per system |
|---|---|---|
| **Correct (keyword proxy)** — every reviewed required fact present and no known-wrong statement (including inconsistent before/after figures on comparison questions); for unanswerable questions, a refusal; for partially answerable ones, the available facts plus a statement that the rest is missing | Per response | 126 |
| **Wrongly declined** — a full refusal to a fully answerable question | Per response | 102 (34 questions × 3) |
| **Correctly declined** — a refusal to an unanswerable question | Per response | 15 (5 × 3) |
| **Expected evidence among the first 5 passages** — share of the question's expected evidence passages (matched by text and source document) found among the first five passages each system retrieved, in retrieval order. A fixed budget for comparability: the complex Agentic route can gather more passages (see above), which this metric does not count | Per response, then mean | 111 (37 questions with expected evidence × 3) |
| **Citations valid** — every cited ID refers to a retrieved passage, no invented references, and at least one citation unless the response declines | Per response | 126; not applicable to the original code |
| **Cited passages contain the expected evidence** — share of the passages a response cites that contain expected evidence from the expected document; averaged | Per cited passage within a response, then mean over responses | Responses that cited at least one passage on a question with expected evidence: 106 Standard, 103 Agentic; not applicable to the original code |
| **Expected facts' evidence cited** — share of the question's required facts whose evidence passage is among the cited passages; averaged | Per fact within a response, then mean over responses | 111; not applicable to the original code |
| **Model calls, retrieval calls, tokens** | Per response, mean | 126 (the original code's token counts were not recorded) |
| **Latency** — wall-clock time per response | Median and 95th percentile | 126 |
| **Paired outcome** — majority result over the three repeats | Per question | 42 |

"Not applicable" means the original code produces free-form references
rather than evidence IDs, so these citation metrics cannot be computed for
it; it does not mean a score of zero.

## Headline results

| Metric | Original | Standard | Agentic |
|---|---|---|---|
| Correct (keyword proxy) | 0.71 (89/126) | **0.83** (104/126) | 0.79 (99/126) |
| Wrongly declined answerable questions | 0.09 (9/102) | 0.00 (0/102) | 0.04 (4/102) |
| Correctly declined unanswerable questions | 1.00 (15/15) | 1.00 (15/15) | 1.00 (15/15) |
| Expected evidence among the first 5 passages | 0.86 | 0.97 | 0.98 |
| Citations valid | n/a | 0.96 (121/126) | 0.97 (122/126) |
| Cited passages contain the expected evidence | n/a | 0.92 | 0.87 |
| Expected facts' evidence cited | n/a | 0.80 | 0.78 |
| Model calls per response | 1 | 1 | 3.4 |
| Retrieval calls per response | 1 | 1 | 1.6 |
| Prompt / completion tokens per response | not recorded | 693 / 35 | 2,151 / 170 |
| Latency, median / 95th percentile | 1.02 / 1.66 s | 0.82 / 1.42 s | 1.77 / 6.86 s |

Per question (majority over three repeats), Agentic RAG was right where
Standard RAG was wrong on 1 question, wrong where Standard was right on 2,
and the two agreed on 39. Under these conditions Standard RAG was stronger
overall and made fewer model calls; the agent's additional calls did not
improve keyword-proxy correctness.

## By category (keyword proxy, per response)

| Category (questions) | Original | Standard | Agentic |
|---|---|---|---|
| Direct factual (8), citation precision (3), unanswerable (5) | 1.00 | 1.00 | 1.00 |
| Partially answerable (3) | 0.33 | 1.00 | 1.00 |
| Version comparison (6) | 1.00 | 0.94 | 0.72 |
| Conflicting information (4) | 0.83 | 0.75 | 0.92 |
| Two-step (2) | 0.00 | 0.83 | 0.67 |
| Regression (3) | 0.33 | 0.78 | 0.78 |
| Multi-document (5) | 0.47 | 0.40 | 0.27 |
| False premise (3) | 0.00 | 0.33 | 0.33 |

Categories are small (two to eight questions), so one question moves a
category score noticeably.

## A defect the evaluation found, and its fix

The first official run showed every Agentic comparison answer failing: the
comparison mode searched the first five documents **in alphabetical order**,
so with eight documents the two travel policies were never searched.
Document selection now uses documents named in the question (or by the
planner), or otherwise a relevance search. Each run below is scored with
the same final scorer; the pre-fix run is preserved unchanged.

| Agentic RAG | Initial run | After the source-selection fix | After a second fix |
|---|---|---|---|
| Correct (keyword proxy) | 0.66 (83/126) | 0.79 (99/126) | 0.79 (99/126) |
| Version comparison | 0.00 (0/18) | 0.72 (13/18) | 0.72 (13/18) |
| Wrongly declined answerable questions | 0.24 (24/102) | 0.06 (6/102) | 0.04 (4/102) |
| Comparison selections including the expected documents | — | 18 of 18 | 18 of 18 |

The second fix stopped answers containing facts plus "the evidence does not
specify…" from being treated as refusals; it did not change overall
correctness. Standard RAG's answers were byte-identical across all runs.

### Scorer changes, reported separately

Label and scorer revisions were applied to fixed sets of saved answers, one
change at a time, so their effect is visible separately from application
changes (keyword-proxy correctness):

| Scoring | Original | Standard | Agentic (initial) | Agentic (final) |
|---|---|---|---|---|
| Labels 1.0, scorer 1.1 (reproduces the original official scores) | 0.68 | 0.79 | 0.61 | 0.78 |
| + labels 1.1 (false-premise pattern catches active voice) | 0.68 | 0.76 | 0.59 | 0.75 |
| + labels 1.2 (contextual facts optional for three questions) | 0.71 | 0.83 | 0.63 | 0.79 |
| + scorer 1.2 (clause-level refusal detection) | 0.71 | 0.83 | 0.66 | 0.79 |
| + labels 1.3 (before/after consistency check) | 0.71 | 0.83 | 0.66 | 0.79 |

## Other components

| Component | Result | Unit |
|---|---|---|
| Claim verifier, challenge set | Wrongly accepted 1 of 14 unsupported claims; wrongly rejected 3 of 11 supported claims; none undecided | Per distinct labelled claim (25); temperature-0 repeats not counted as extra cases |
| Tabular Q&A | 6 of 9 functional cases correct (failures were extra filters added by the model); no security violations observed in checks of file state and blocked execution | Per case (9), identical across 3 runs |
| LLM judge (local `gemma3:12b`) | Matched 20 of 22 hand-written calibration answers (two known-incorrect answers passed); agreed with the keyword proxy on 26 of 30 subset judgments | Per calibration answer; per judgment |

The judge is supporting evidence only, not ground truth.

## Limitations

- The keyword proxy can miss correct paraphrases and can be satisfied by
  answers that contain the right terms in a wrong statement; it is not
  semantic correctness. The LLM judge has passed wrong answers and failed
  correct ones.
- Citation validity shows that cited IDs refer to retrieved passages, not
  that the passages support the claims; the claim verifier is a small model
  with the error rates above.
- One small model (3B), one machine, 42 synthetic questions, small categories.
- Latency was measured locally and depends on hardware and answer length.
- The original system differs in more than its answering strategy (see the
  protocol), so the comparison with it is between whole systems.

## Reproducing

These commands were checked against the implemented command-line parser.
Clone with full history: the Original baseline is extracted from commit
`909df31`, which a shallow clone does not contain.

```bash
git clone https://github.com/Namantyagi2727/ragbase.git
cd ragbase
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt -c constraints.txt
```

Install Ollama from https://ollama.com, make sure it is running (the desktop
app, or `ollama serve` in another terminal), and pull the generator model:

```bash
ollama pull llama3.2
```

Run the benchmark — all 42 questions, all three systems, three repeats, base
seed 1000 (so the repeats use seeds 1001–1003), scored with labels 1.3 and
scorer 1.2:

```bash
python -m evals run --out evals/results/run-a --modes original,standard,agentic --repeats 3 --seed 1000 --labels 1.3 --scoring 1.2
```

This writes the records, `summary.csv` and `report.md` to
`evals/results/run-a` (ignored by git). The original run took about ten
minutes on the Apple Silicon Mac described above.

To compare two runs — for example a second run to see run-to-run variation —
score both with the same label and scorer versions (original-mode results
are taken from the first run):

```bash
python -m evals run --out evals/results/run-b --modes standard,agentic --repeats 3 --seed 1000 --labels 1.3 --scoring 1.2
python -m evals compare --run first=evals/results/run-a --run second=evals/results/run-b --labels 1.3 --scoring 1.2 --out evals/results/compare-a-b
```

Two caveats:

- The raw outputs of the runs reported on this page (the initial, post-fix
  and final runs) are kept locally and are not in the public repository, so
  the exact historical comparison cannot be regenerated without those saved
  artifacts. A fresh run is an independent measurement of the same
  benchmark.
- Outputs depend on hardware, the Ollama version and the model build. With
  fixed seeds, Standard RAG's answers were byte-identical across separate
  runs on the machine used here; identical outputs on other machines or
  runtime versions are not guaranteed, and the Original baseline is
  unseeded.
