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


def summarize_unique(results: list[dict]) -> dict:
    """Counts per distinct challenge claim, not per repeated verdict.

    Repeats at temperature 0 are not independent cases, so each claim counts
    once: its outcome when every repeat agrees, otherwise "inconsistent".
    """
    outcomes: dict[tuple, dict] = {}
    for r in results:
        for i, c in enumerate(r["claims"]):
            entry = outcomes.setdefault((r["id"], i), {"case": r["id"], "claim": c["claim"],
                                                        "expected": c["expected"], "predicted": []})
            entry["predicted"].append(c["predicted"])
    cells = Counter()
    claims = []
    for e in outcomes.values():
        pred = e["predicted"][0] if len(set(e["predicted"])) == 1 else "inconsistent"
        cells[(pred, e["expected"])] += 1
        claims.append({**e, "outcome": pred})
    n_accept = sum(1 for e in outcomes.values() if e["expected"] == "accept")
    n_reject = len(outcomes) - n_accept
    fp, fn = cells[("accepted", "reject")], cells[("rejected", "accept")]
    return {
        "unique_claims": len(outcomes), "should_accept": n_accept, "should_reject": n_reject,
        "accepted_supported": cells[("accepted", "accept")], "accepted_unsupported (false positive)": fp,
        "rejected_unsupported": cells[("rejected", "reject")], "rejected_supported (false negative)": fn,
        "undecided_supported": cells[("undecided", "accept")], "undecided_unsupported": cells[("undecided", "reject")],
        "inconsistent_supported": cells[("inconsistent", "accept")],
        "inconsistent_unsupported": cells[("inconsistent", "reject")],
        "false_positive_rate": f"{fp}/{n_reject}", "false_negative_rate": f"{fn}/{n_accept}",
        "errors": sorted({(c["case"], c["outcome"]) for c in claims
                          if (c["outcome"] == "accepted") != (c["expected"] == "accept")}),
    }


def run(cases: list[VerifierCase], model: ChatModel, repeats: int = 1) -> dict:
    results = [dict(evaluate_case(c, model), repeat=i + 1) for i in range(repeats) for c in cases]
    return {"model": getattr(model, "model", None), "repeats": repeats, "summary": summarize(results),
            "unique": summarize_unique(results), "results": results}
