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

Rubric v2 keeps correctness, faithfulness, completeness and abstention as
separate decisions and applies the pass rule in code: an incomplete answer
cannot pass because its claims are true, and abstention is judged against
the expected answerability rather than the passages the system retrieved.
Rubric v3 clarifies partially answerable questions: stating the available
facts and saying the rest is not in the documents is complete.
calibrate() checks the judge itself on answers whose outcome is known.
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
    """Rubric v1 (the initial benchmark): the model also decided `passed`."""

    model_config = ConfigDict(extra="ignore")
    correctness: int = Field(ge=1, le=5)
    faithfulness: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    abstention: Literal["appropriate", "inappropriate", "not_applicable"]
    passed: bool
    issues: list[str] = Field(default_factory=list, max_length=5)
    rationale: str = Field(default="", max_length=400)


class JudgeVerdictV2(BaseModel):
    """Rubric v2: four separate decisions; the pass rule is applied in code, not by the model."""

    model_config = ConfigDict(extra="ignore")
    correctness: int = Field(ge=1, le=5)
    faithfulness: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    abstention: Literal["appropriate", "inappropriate", "not_applicable"]
    missing_parts: list[str] = Field(default_factory=list, max_length=5)
    issues: list[str] = Field(default_factory=list, max_length=5)
    rationale: str = Field(default="", max_length=400)


RUBRIC_V1 = """You grade an answer produced by a document question-answering system.

Use the reference facts as the definition of a correct answer. Use the retrieved evidence to judge faithfulness.
- correctness (1-5): 5 = states all reference facts correctly; 3 = partly correct or missing a key fact; 1 = wrong.
- faithfulness (1-5): 5 = every statement is supported by the retrieved evidence; 1 = mostly unsupported or invented.
- completeness (1-5): 5 = answers every part of the question, including saying what the documents do not cover.
- abstention: "appropriate" if the system correctly said the documents do not contain the answer;
  "inappropriate" if it refused although the answer was available, or answered although it was not;
  "not_applicable" otherwise.
- passed: true only if correctness >= 4 and faithfulness >= 4 and abstention is not "inappropriate".
List concrete issues (at most 5). Keep the rationale to two sentences. Grade only what is written."""

RUBRIC_V2 = """You grade an answer produced by a document question-answering system. Make four separate decisions.

The reviewed reference facts and the expected answerability define what a correct, complete answer contains.
The retrieved evidence is only what the system happened to find: it may be incomplete or wrong.

- correctness (1-5): are the statements in the answer true according to the reference facts?
  5 = everything stated is true; 3 = a mix of true and false; 1 = mainly false. Accepting a false premise in the
  question is a correctness error. Do not lower correctness for missing facts; that is completeness.
- faithfulness (1-5): is every statement supported by the retrieved evidence? 5 = all supported; 1 = mostly
  unsupported or invented. For an answer that only declines to answer, give 5.
- completeness (1-5): does the answer cover every reference fact and every part of the question?
  5 = all of them; 3 = one required part missing; 1 = most missing. List each missing reference fact in
  missing_parts. An answer that declines to answer an answerable question has completeness 1.
- abstention: compare with the expected answerability, not with the retrieved evidence.
  "appropriate": the expected answerability is "no" (or "partial") and the answer says the documents do not
  contain the answer (or the missing part).
  "inappropriate": the expected answerability is "yes" but the answer declines, even if the retrieved evidence
  lacked the answer; or the expected answerability is "no" but the answer gives one anyway.
  "not_applicable": the question is answerable and the answer attempts it.
List concrete issues (at most 5). Keep the rationale to two sentences. Grade only what is written."""

# Rubric v3 changes only how completeness and abstention treat partially answerable
# questions: stating that genuinely unavailable information is missing is complete.
RUBRIC_V3 = RUBRIC_V2.replace(
    """  5 = all of them; 3 = one required part missing; 1 = most missing. List each missing reference fact in
  missing_parts. An answer that declines to answer an answerable question has completeness 1.""",
    """  5 = all of them; 3 = one required part missing; 1 = most missing. List each missing reference fact in
  missing_parts. An answer that declines to answer an answerable question has completeness 1.
  When the expected answerability is "partial", a complete answer states the available reference facts and
  says that the information listed under "Information the documents lack" is not in the documents. Saying so
  counts as complete; never expect that information to be supplied, and do not list it in missing_parts.
  An answer that states the available facts but does not say the rest is unavailable has completeness 3.
  An answer that invents the unavailable information is a correctness error.""",
).replace(
    """  "appropriate": the expected answerability is "no" (or "partial") and the answer says the documents do not
  contain the answer (or the missing part).""",
    """  "appropriate": the expected answerability is "no" and the answer says the documents do not contain the
  answer; or it is "partial" and the answer gives the available part and says the rest is missing.
  For "partial", declining the whole question although part of it is answerable is "inappropriate".""",
)
assert RUBRIC_V3 != RUBRIC_V2

RUBRICS = {"1": (RUBRIC_V1, JudgeVerdict), "2": (RUBRIC_V2, JudgeVerdictV2), "3": (RUBRIC_V3, JudgeVerdictV2)}


def passes(verdict: dict, rubric: str) -> bool:
    """Rubric v2's pass rule, applied in code so it cannot drift: every dimension must hold."""
    if rubric == "1":
        return bool(verdict["passed"])
    return (verdict["correctness"] >= 4 and verdict["faithfulness"] >= 4 and verdict["completeness"] >= 4
            and verdict["abstention"] != "inappropriate")


def judge_messages(case, answer: str, evidence_texts: list[str], rubric: str = "1") -> list:
    facts = "\n".join(f"- {f.statement}" for f in case.expected_facts) or "- (none: the documents do not answer this)"
    missing = "; ".join(case.missing_information) or "none"
    ev = "\n\n".join(f"[{i}] {t[:600]}" for i, t in enumerate(evidence_texts[:8], start=1)) or "(no evidence retrieved)"
    answerability = f"Expected answerability: {case.answerable}\n" if rubric != "1" else ""
    return [
        {"role": "system", "content": RUBRICS[rubric][0]},
        {"role": "user", "content": (
            f"Question: {case.question}\n\n{answerability}Reference facts:\n{facts}\nInformation the documents lack: {missing}\n"
            f"Reference answer: {case.reference_answer}\n\nRetrieved evidence:\n{ev}\n\nAnswer to grade:\n{answer}")},
    ]


def select_items(rows: list[dict], max_items: int, modes: list[str], case_ids: list[str], seed: int) -> list[dict]:
    pool = [r for r in rows if r["mode"] in modes and r["repeat"] == 1 and (not case_ids or r["case_id"] in case_ids)]
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:max_items] if max_items else pool


def run(run_dir: Path, model: ChatModel, max_items: int = 10, judgments: int = 1,
        modes: tuple[str, ...] = ("standard", "agentic"), case_ids: tuple[str, ...] = (),
        seed: int = 7, estimate_only: bool = False, progress=print, rubric: str = "3",
        labels: str = "1.0", out_dir: Path | None = None) -> dict:
    """Judge a run's answers. Outputs go to out_dir (default: run_dir); run_dir is only read."""
    _, rows = load_rows(run_dir, labels=labels)
    cases = {c.id: c for c in load_cases(labels=labels)}
    schema = RUBRICS[rubric][1]
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
                                                            [e["text"] for e in row.get("evidence") or []], rubric),
                                      schema, temperature=0.0)
            timings.append(time.perf_counter() - t)
            verdict = res.value.model_dump() if res.value else None
            if verdict is not None and rubric != "1":
                verdict["passed"] = passes(verdict, rubric)
            verdicts.append({"item": item_id, "judgment": j + 1, "verdict": verdict, "error": res.error,
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
    digest = None
    try:
        import httpx

        import config

        tags = httpx.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=3).json().get("models", [])
        digest = next((t.get("digest", "")[:12] for t in tags if t["name"] == getattr(model, "model", None)), None)
    except Exception:
        pass
    out = {"judge_model": getattr(model, "model", None), "judge_model_digest": digest, "rubric": rubric,
           "labels": labels,
           "items": len(items), "judgments_per_item": judgments,
           "failed_judgments": sum(1 for v in verdicts if not v["verdict"]), "estimate": estimate,
           "agreement_with_keyword_proxy": round(1 - len(disagreements) / len(judged), 4) if judged else None,
           "self_consistency": consistency, "disagreements": disagreements,
           "note": "The judge is not ground truth; disagreements need human review."}
    out_dir = Path(out_dir) if out_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "judge_items.json").write_text(json.dumps(mapping, indent=2))
    with open(out_dir / "judge.jsonl", "w") as f:
        for v in verdicts:
            f.write(json.dumps(v) + "\n")
    (out_dir / "judge_summary.json").write_text(json.dumps(out, indent=2))
    return out


CALIBRATION_FILE = Path(__file__).resolve().parent / "judge_calibration.jsonl"


def calibrate(model: ChatModel, rubric: str = "3", labels: str = "1.0", path: Path = CALIBRATION_FILE,
              progress=print) -> dict:
    """Judge hand-written answers whose pass/fail follows from the reviewed labels.

    Checks the judge, not the system: a judge that passes a known-incorrect
    answer here (or fails a known-correct one) is not reliable on that kind of
    case. "reference" evidence means the expected evidence quotes of the case.
    """
    cases = {c.id: c for c in load_cases(labels=labels)}
    schema = RUBRICS[rubric][1]
    results = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        case = cases[item["case_id"]]
        evidence = ([e.quote for f in case.expected_facts for e in f.evidence] if item["evidence"] == "reference"
                    else item["evidence"])
        res = complete_structured(model, judge_messages(case, item["answer"], evidence, rubric), schema,
                                  temperature=0.0)
        verdict = res.value.model_dump() if res.value else None
        passed = passes(verdict, rubric) if verdict else None
        results.append({"id": item["id"], "case_id": item["case_id"], "expected_pass": item["expected_pass"],
                        "judge_passed": passed, "agrees": passed == item["expected_pass"] if verdict else None,
                        "reason": item["reason"], "verdict": verdict, "error": res.error})
        progress(f"[calibrate] {item['id']} expected {'pass' if item['expected_pass'] else 'fail'} -> "
                 f"{'error' if verdict is None else 'pass' if passed else 'fail'}")
    judged = [r for r in results if r["agrees"] is not None]
    return {"judge_model": getattr(model, "model", None), "rubric": rubric, "items": len(results),
            "agreement": f"{sum(r['agrees'] for r in judged)}/{len(judged)}",
            "false_passes": [r["id"] for r in judged if r["judge_passed"] and not r["expected_pass"]],
            "false_fails": [r["id"] for r in judged if not r["judge_passed"] and r["expected_pass"]],
            "failed_judgments": len(results) - len(judged), "results": results}
