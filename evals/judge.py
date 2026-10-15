"""
evals/judge.py — Optional LLM-as-a-judge over run records (local gemma3:12b by default).

The judge is a supplementary signal, not ground truth. It is:
  - blind to the answer mode (items are shuffled and identified by opaque IDs;
    the mapping is stored separately);
  - given the question, the reviewed reference facts and answer, the evidence
    the system retrieved, and the answer — with a fixed rubric;
  - bounded: the number of items and repeated judgments are configurable, and
    --estimate times a few items and projects the full workload first.
Judge outcomes are compared with the keyword proxy; disagreements are listed
for human review.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from evals.dataset import load_cases
from evals.report import load_rows
from llm import ChatModel
from llm.structured import complete_structured


class JudgeVerdict(BaseModel):
    model_config = ConfigDict(extra="ignore")
    correctness: int = Field(ge=1, le=5)
    faithfulness: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    abstention: Literal["appropriate", "inappropriate", "not_applicable"]
    passed: bool
    issues: list[str] = Field(default_factory=list, max_length=5)
    rationale: str = Field(default="", max_length=400)


RUBRIC = """You grade an answer produced by a document question-answering system.

Use the reference facts as the definition of a correct answer. Use the retrieved evidence to judge faithfulness.
- correctness (1-5): 5 = states all reference facts correctly; 3 = partly correct or missing a key fact; 1 = wrong.
- faithfulness (1-5): 5 = every statement is supported by the retrieved evidence; 1 = mostly unsupported or invented.
- completeness (1-5): 5 = answers every part of the question, including saying what the documents do not cover.
- abstention: "appropriate" if the system correctly said the documents do not contain the answer;
  "inappropriate" if it refused although the answer was available, or answered although it was not;
  "not_applicable" otherwise.
- passed: true only if correctness >= 4 and faithfulness >= 4 and abstention is not "inappropriate".
List concrete issues (at most 5). Keep the rationale to two sentences. Grade only what is written."""


def judge_messages(case, answer: str, evidence_texts: list[str]) -> list:
    facts = "\n".join(f"- {f.statement}" for f in case.expected_facts) or "- (none: the documents do not answer this)"
    missing = "; ".join(case.missing_information) or "none"
    ev = "\n\n".join(f"[{i}] {t[:600]}" for i, t in enumerate(evidence_texts[:8], start=1)) or "(no evidence retrieved)"
    return [
        {"role": "system", "content": RUBRIC},
        {"role": "user", "content": (
            f"Question: {case.question}\n\nReference facts:\n{facts}\nInformation the documents lack: {missing}\n"
            f"Reference answer: {case.reference_answer}\n\nRetrieved evidence:\n{ev}\n\nAnswer to grade:\n{answer}")},
    ]


def select_items(rows: list[dict], max_items: int, modes: list[str], case_ids: list[str], seed: int) -> list[dict]:
    pool = [r for r in rows if r["mode"] in modes and r["repeat"] == 1 and (not case_ids or r["case_id"] in case_ids)]
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:max_items] if max_items else pool


def run(run_dir: Path, model: ChatModel, max_items: int = 10, judgments: int = 1,
        modes: tuple[str, ...] = ("standard", "agentic"), case_ids: tuple[str, ...] = (),
        seed: int = 7, estimate_only: bool = False, progress=print) -> dict:
    _, rows = load_rows(run_dir)
    cases = {c.id: c for c in load_cases()}
    items = select_items(rows, max_items, list(modes), list(case_ids), seed)
    if estimate_only:
        items = items[:3]
    mapping, verdicts, timings = {}, [], []
    for n, row in enumerate(items, start=1):
        item_id = f"J{n:03d}"
        mapping[item_id] = {"mode": row["mode"], "case_id": row["case_id"], "repeat": row["repeat"]}
        for j in range(judgments):
            t = time.perf_counter()
            res = complete_structured(model, judge_messages(cases[row["case_id"]], row.get("answer") or "",
                                                            [e["text"] for e in row.get("evidence") or []]),
                                      JudgeVerdict, temperature=0.0)
            timings.append(time.perf_counter() - t)
            verdicts.append({"item": item_id, "judgment": j + 1,
                             "verdict": res.value.model_dump() if res.value else None, "error": res.error,
                             "keyword_correct": row.get("keyword_correct")})
            progress(f"[judge] {item_id} #{j + 1} {timings[-1]:.1f}s")
    per_call = sum(timings) / len(timings) if timings else 0.0
    full = len(select_items(rows, max_items, list(modes), list(case_ids), seed)) * judgments
    estimate = {"seconds_per_judgment": round(per_call, 2), "items_selected": full,
                "projected_minutes": round(per_call * full / 60, 1)}
    if estimate_only:
        return {"estimate": estimate}
    judged = [v for v in verdicts if v["verdict"]]
    disagreements = [{"item": v["item"], **mapping[v["item"]], "judge_passed": v["verdict"]["passed"],
                      "keyword_correct": v["keyword_correct"], "issues": v["verdict"]["issues"]}
                     for v in judged if v["verdict"]["passed"] != bool(v["keyword_correct"])]
    consistency = None
    if judgments > 1:
        by_item: dict = {}
        for v in judged:
            by_item.setdefault(v["item"], []).append(v["verdict"]["passed"])
        consistency = round(sum(len(set(x)) == 1 for x in by_item.values()) / len(by_item), 4) if by_item else None
    out = {"judge_model": getattr(model, "model", None), "items": len(items), "judgments_per_item": judgments,
           "failed_judgments": sum(1 for v in verdicts if not v["verdict"]), "estimate": estimate,
           "agreement_with_keyword_proxy": round(1 - len(disagreements) / len(judged), 4) if judged else None,
           "self_consistency": consistency, "disagreements": disagreements,
           "note": "The judge is not ground truth; disagreements need human review."}
    (run_dir / "judge_items.json").write_text(json.dumps(mapping, indent=2))
    with open(run_dir / "judge.jsonl", "w") as f:
        for v in verdicts:
            f.write(json.dumps(v) + "\n")
    (run_dir / "judge_summary.json").write_text(json.dumps(out, indent=2))
    return out
