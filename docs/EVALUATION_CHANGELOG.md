# Evaluation change log

Changes to the benchmark, kept in three separate categories so that a change
in scores can be traced to its cause. Any before/after comparison must score
both result sets with the same label and scoring version
(`python -m evals compare`).

## Generated outputs

| Result set | What it is | Status |
|---|---|---|
| `official-2026-10-15` | Initial official run (commit `a83197f`): Original, Standard, Agentic × 42 cases × 3 repeats; verifier challenge set; tabular suite; judge subset (rubric 1) | **Pre-fix baseline.** Kept unchanged; file hashes are checked before and after every rescoring |
| `official-2026-10-15-rescored-labels-1.1[-proposed]` | The same saved answers, rescored with labels 1.1 (and with the proposed revisions) | Answers unchanged; only the scoring differs |
| `postfix-2026-10-15` | Standard and Agentic rerun after the comparison-mode fix, with the same corpus 1.0.0, model digest, retrieval configuration, generation settings, seeds and scoring implementation | Post-source-fix result, reviewed and accepted; kept unchanged |
| `judge-v2-2026-10-15` | Judge rubric 2 calibration and subset judgments | Kept unchanged |
| `final-2026-10-15` | Standard and Agentic rerun after the abstention-classification fix, same approved configuration and seeds | Final controlled run |
| `judge-v3-2026-10-15` | Judge rubric 3 calibration and subset judgments | Previous judgments kept alongside |

Original RAGBase is not rerun: its code snapshot (`909df31`), corpus, model and
settings are unchanged, so its baseline measurements are reused.

Result files stay local (`evals/results/` is gitignored) until reviewed.

## Application changes that affect outputs

| Version | Change | Reason |
|---|---|---|
| final | A full abstention requires that no clause gives a substantive answer; sentences mixing a fact with an absence statement go to claim verification | Benchmark case M03 stated both facts plus "the evidence does not specify ...", was classified as a full abstention, and its claims skipped verification |
| post-fix | Comparison mode selects documents named in the question or by the planner, otherwise by one bounded relevance discovery search, instead of the first five documents alphabetically; passages are interleaved by source before the evidence cap; limitations are recorded | The initial benchmark showed the travel policies were never searched in comparison mode (8 documents, alphabetical cap of 5); all 18 comparison-mode answers failed |

## Dataset labels

`evals/cases.jsonl` stays at version 1.0 exactly as approved. Later versions
are applied from `evals/label_revisions.jsonl`, each entry with a rationale and
its own review status. Proposed revisions apply only when explicitly requested
(`--include-proposed`) and are labelled as such in reports.

| Version | Case | Change | Status |
|---|---|---|---|
| 1.1 | FP01 | Forbidden pattern now also catches active statements ("the policy reduced the allowance from $45 to $60"); the 1.0 pattern only caught the passive voice | Approved |
| 1.2 | T01 | Naming the 2024 policy and its effective date optional when the answer states a 230-dollar night needs no separate approval under the 250-dollar threshold (which exists only in the 2024 policy); f2 also accepts "does not require approval" | Approved |
| 1.2 | V05 | Naming the edition optional when the answer gives the current 67-cent rate (2024-only); the superseded 58-cent rate still fails | Approved |
| 1.2 | C01 | The superseded Handbook limit optional when the answer states three days after probation; the addendum states it amends the Handbook | Approved |
| 1.3 | V01–V04, V06, FP01 | Explicit old → new change and direction recorded for the comparison-consistency check | Approved |

Optional facts keep their evidence: retrieval and citation metrics still
require the same source passages.

## Scoring implementation

| Version | Change | Effect on earlier results |
|---|---|---|
| 1.0 | Initial official scoring | — |
| 1.1 | Optional expected facts: reported in `facts_covered`, not needed for `keyword_correct` | None under labels 1.0: rescoring the baseline with scoring 1.1 and labels 1.0 reproduces all 378 scored rows exactly |
| 1.1 | Verifier challenge results counted per distinct claim; repeats at temperature 0 are not counted as separate cases, and a claim whose repeats disagree is reported as inconsistent | Same verdicts, different denominators |
| 1.2 | Abstention re-derived from the answer text with the clause-level classifier, for every mode | Reclassifies answers that state facts plus a gap (e.g. M03, Original P02) as answers |
| 1.2 | Comparison-consistency check where labels (1.3) state an old → new change: wrong pairings ("from 75 to 60" when the change is 75 → 90), reversed figures and the opposite direction fail. Narrow and deterministic; it does not establish semantic correctness | None for cases without a comparison label |

## LLM judge

| Rubric | Change |
|---|---|
| 1 | Initial: the model decided `passed` from correctness and faithfulness only |
| 2 | Correctness, faithfulness, completeness and abstention judged separately; the pass rule (all ≥ 4, abstention not inappropriate) is applied in code; abstention is judged against the expected answerability and reference facts, not the retrieved passages. Checked first on 13 calibration answers (`evals/judge_calibration.jsonl`) whose outcome follows from the approved labels |
| 3 | As rubric 2, except for partially answerable questions: stating the available facts and saying that genuinely unavailable information is not in the documents is complete; inventing it is a correctness error; declining the whole question is inappropriate. 22 calibration answers |

The judge is supporting evidence, not ground truth.

## Known limitations (recorded at the end of Phase 3)

- Standard RAG still outperforms Agentic RAG overall on the reviewed benchmark.
- Multi-document synthesis and two-step tasks remain weak in both modes.
- Agentic mode uses more model calls and has higher latency than Standard RAG.
- The verifier rejects some supported claims and accepts some incorrect ones
  (verifier challenge set: false positives and false negatives per distinct claim).
- The tabular suite passed 6 of 9 functional cases with zero observed security
  violations; the failures are spurious filters added by the small model.
- The keyword proxy and the LLM judge are both fallible: the proxy misses
  paraphrases and can be satisfied by lucky matches; the judge has passed a
  wrong answer and failed correct ones. Neither is ground truth.
- Small benchmark (42 cases), one 3B generator model, one machine.
