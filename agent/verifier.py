"""
agent/verifier.py — Combine deterministic checks with the model's claim verdicts.

Three independent signals, never collapsed into one:

  1. Citation validity (deterministic, citations.py): every cited ID exists in
     the evidence actually provided, and no free-form references appear.
  2. Numeric grounding (deterministic): every number in a cited claim appears
     in the evidence that claim cites.
  3. Claim support (model): per-claim verdicts from the verifier model,
     whose cited evidence IDs are themselves checked.

A valid citation report is not proof of grounding, and a verifier call that
fails or returns no verdicts yields `not_verified` — never `verified`.
"""

from __future__ import annotations

import re
from typing import Optional

from agent.schemas import VerifierOutput
from agent.state import Verification
from citations import CITE_RE, CitationReport, Evidence, extract_citations, split_claims

_NUMBER_RE = re.compile(r"(?<![\w.])(\d+(?:[.,]\d+)*|\.\d+)(?:\s?%)?")


def _normalize_number(raw: str) -> str:
    whole, _, frac = raw.replace(",", "").partition(".")
    whole = whole.lstrip("0") or "0"
    frac = frac.rstrip("0")
    return f"{whole}.{frac}" if frac else whole


def numbers_in(text: str) -> set[str]:
    return {_normalize_number(m.group(1)) for m in _NUMBER_RE.finditer(CITE_RE.sub(" ", text))}


def numeric_mismatches(answer: str, evidence: list[Evidence]) -> list[str]:
    """Cited claims containing numbers that none of their cited evidence contains."""
    by_id = {e.eid: e for e in evidence}
    problems = []
    for claim in split_claims(answer):
        cited = [by_id[c] for c in extract_citations(claim) if c in by_id]
        if not cited:
            continue
        available = set().union(*(numbers_in(e.text) for e in cited))
        missing = sorted(numbers_in(claim) - available)
        if missing:
            problems.append(f"{claim} (not in cited evidence: {', '.join(missing)})")
    return problems


def assess(
    answer: str,
    evidence: list[Evidence],
    report: CitationReport,
    llm_output: Optional[VerifierOutput],
    llm_error: Optional[str] = None,
    llm_skipped: bool = False,
) -> Verification:
    known = {e.eid for e in evidence}
    issues: list[str] = []
    if report.invalid_ids:
        issues.append(f"cites evidence that was not provided: {', '.join(report.invalid_ids)}")
    if report.non_evidence_citations:
        issues.append("contains references that are not evidence IDs")
    if not report.abstained and not report.valid_ids:
        issues.append("answer has no evidence citations")
    if report.uncited_claims:
        issues.append(f"{len(report.uncited_claims)} claim(s) without citations")

    numeric = [] if report.abstained else numeric_mismatches(answer, evidence)

    unsupported: list[str] = []
    partial: list[str] = []
    accepted_verdicts = 0
    followup = None
    if llm_output is not None:
        for v in llm_output.verdicts:
            bad_ids = [i for i in v.evidence_ids if i not in known]
            if bad_ids:
                issues.append(f"verifier referenced unknown evidence {', '.join(bad_ids)}; verdict ignored")
                continue
            accepted_verdicts += 1
            if v.status == "unsupported":
                unsupported.append(v.claim)
            elif v.status == "partially_supported":
                partial.append(v.claim)
        followup = (llm_output.followup_query or "").strip() or None
        if llm_output.missing_information:
            issues.append(f"missing information: {llm_output.missing_information}")

    llm_status = "skipped" if llm_skipped else ("completed" if llm_output is not None else "failed")

    if report.abstained:
        grounding = "skipped"
    elif llm_status != "completed":
        grounding = "unsupported" if numeric else "not_verified"
    elif unsupported or numeric:
        grounding = "unsupported"
    elif partial:
        grounding = "partially_verified"
    elif accepted_verdicts == 0:
        grounding = "not_verified"
        issues.append("verifier returned no usable claim verdicts")
    else:
        grounding = "verified"

    citation_ok = report.ok
    if not citation_ok or grounding in ("unsupported", "partially_verified"):
        decision = "revise"
    elif llm_output is not None and llm_output.recommendation != "accept" and followup:
        decision = "revise"   # e.g. an abstention the verifier thinks more retrieval could fix
    elif grounding == "not_verified":
        decision = "finalize"
    else:
        decision = "accept"

    return Verification(
        citation_ok=citation_ok,
        grounding=grounding,
        llm_status=llm_status,
        decision=decision,
        unsupported_claims=unsupported + partial,
        numeric_mismatches=numeric,
        issues=issues,
        followup_query=followup,
        llm_error=llm_error,
    )


def revision_feedback(v: Verification) -> str:
    """Instructions for the next synthesis round, built from verification findings."""
    lines = ["Your previous answer had these problems. Fix them using only the evidence:"]
    lines += [f"- {i}" for i in v.issues]
    lines += [f"- not supported by the evidence: {c}" for c in v.unsupported_claims[:6]]
    lines += [f"- number not found in cited evidence: {m}" for m in v.numeric_mismatches[:6]]
    lines.append("Remove or correct unsupported statements. If the evidence is insufficient, say so.")
    return "\n".join(lines)
