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
| `postfix-2026-10-15` | Standard and Agentic rerun after the comparison-mode fix, with the same corpus 1.0.0, model digest, retrieval configuration, generation settings, seeds and scoring implementation | Post-fix result, reported next to the baseline, not instead of it |

Original RAGBase is not rerun: its code snapshot (`909df31`), corpus, model and
settings are unchanged, so its baseline measurements are reused.

Result files stay local (`evals/results/` is gitignored) until reviewed.

## Application changes that affect outputs

| Version | Change | Reason |
|---|---|---|
| post-fix | Comparison mode selects documents named in the question or by the planner, otherwise by one bounded relevance discovery search, instead of the first five documents alphabetically; passages are interleaved by source before the evidence cap; limitations are recorded | The initial benchmark showed the travel policies were never searched in comparison mode (8 documents, alphabetical cap of 5); all 18 comparison-mode answers failed |

## Dataset labels

`evals/cases.jsonl` stays at version 1.0 exactly as approved. Later versions
are applied from `evals/label_revisions.jsonl`, each entry with a rationale and
its own review status. Proposed revisions apply only when explicitly requested
(`--include-proposed`) and are labelled as such in reports.

| Version | Case | Change | Status |
|---|---|---|---|
| 1.1 | FP01 | Forbidden pattern now also catches active statements ("the policy reduced the allowance from $45 to $60"); the 1.0 pattern only caught the passive voice | Approved |
| 1.1 | T01 | f1 (naming the 2024 edition) optional; f2 also accepts "does not require approval" for "approved automatically" | Proposed |
| 1.1 | V05 | f2 (the 2024 edition replaced 2023) optional | Proposed |
| 1.1 | C01 | f2 (the Handbook's earlier two-day limit) optional | Proposed |

## Scoring implementation

| Version | Change | Effect on earlier results |
|---|---|---|
| 1.0 | Initial official scoring | — |
| 1.1 | Optional expected facts: reported in `facts_covered`, not needed for `keyword_correct` | None under labels 1.0: rescoring the baseline with scoring 1.1 and labels 1.0 reproduces all 378 scored rows exactly |
| 1.1 | Verifier challenge results counted per distinct claim; repeats at temperature 0 are not counted as separate cases, and a claim whose repeats disagree is reported as inconsistent | Same verdicts, different denominators |

## LLM judge

| Rubric | Change |
|---|---|
| 1 | Initial: the model decided `passed` from correctness and faithfulness only |
| 2 | Correctness, faithfulness, completeness and abstention judged separately; the pass rule (all ≥ 4, abstention not inappropriate) is applied in code; abstention is judged against the expected answerability and reference facts, not the retrieved passages. Checked first on 13 calibration answers (`evals/judge_calibration.jsonl`) whose outcome follows from the approved labels |

The judge is supporting evidence, not ground truth.
