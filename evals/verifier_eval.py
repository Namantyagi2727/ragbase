"""
evals/verifier_eval.py — Measure the verifier alone on fixed answer/evidence pairs.

For each challenge pair the real verification path runs (structured verifier
call + deterministic checks via agent.verifier.assess). Each claim's outcome is
accepted / rejected / undecided, compared with its label for support by the
*cited* evidence. Answer-level outcomes are reported per answer label, so a
factually true claim with a wrong citation is counted separately from a false one.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from agent.prompts import verifier_messages
from agent.schemas import VerifierOutput
from agent.verifier import assess
from citations import Evidence, validate_citations
from evals.dataset import VerifierCase
from llm import ChatModel
from llm.structured import complete_structured


def evaluate_case(case: VerifierCase, model: ChatModel) -> dict:
    evidence = [Evidence(e.id, f"{case.id}-{e.id}", e.source, "", e.text) for e in case.evidence]
    claims = [c.text for c in case.claims]
    report = validate_citations(case.answer, evidence)
    res = complete_structured(model, verifier_messages(case.question, evidence, claims), VerifierOutput, temperature=0.0)
    v = assess(case.answer, evidence, report, res.value, claims=claims, llm_error=res.error)
    numeric = " ".join(v.numeric_mismatches)
    outcomes = []
    for c in case.claims:
        if res.value is None:
            pred = "undecided"
        elif c.text in v.unsupported_claims or c.text in numeric:
            pred = "rejected"
        elif c.text in v.unverified_claims:
            pred = "undecided"
        else:
            pred = "accepted"
        outcomes.append({"claim": c.text, "expected": "accept" if c.label == "supported" else "reject",
                         "factually_true": c.factually_true, "predicted": pred})
    return {"id": case.id, "kind": case.kind, "answer_label": case.answer_label, "required": case.required,
            "grounding": v.grounding, "llm_status": v.llm_status, "claims": outcomes,
            "llm_error": res.error}


def summarize(results: list[dict]) -> dict:
    cells = Counter()
    for r in results:
        for c in r["claims"]:
            cells[(c["predicted"], c["expected"])] += 1
    tp, fp = cells[("accepted", "accept")], cells[("accepted", "reject")]
    tn, fn = cells[("rejected", "reject")], cells[("rejected", "accept")]
    by_label = defaultdict(Counter)
    for r in results:
        by_label[r["answer_label"]]["verified" if r["grounding"] == "verified" else "not_verified"] += 1
    return {
        "claims": {
            "accepted_supported": tp, "accepted_unsupported (false positive)": fp,
            "rejected_unsupported": tn, "rejected_supported (false negative)": fn,
            "undecided_supported": cells[("undecided", "accept")], "undecided_unsupported": cells[("undecided", "reject")],
            "false_positive_rate": round(fp / (fp + tn), 4) if fp + tn else None,
            "false_negative_rate": round(fn / (fn + tp), 4) if fn + tp else None,
        },
        "answers_by_label": {k: dict(v) for k, v in by_label.items()},
        "required": {r["id"]: r["grounding"] for r in results if r["required"]},
        "n_runs": len(results),
    }


def run(cases: list[VerifierCase], model: ChatModel, repeats: int = 1) -> dict:
    results = [dict(evaluate_case(c, model), repeat=i + 1) for i in range(repeats) for c in cases]
    return {"model": getattr(model, "model", None), "repeats": repeats, "summary": summarize(results), "results": results}
