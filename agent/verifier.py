"""
agent/verifier.py — Combine deterministic checks with the model's claim verdicts.

Independent signals, never collapsed into one:

  1. Citation validity (deterministic, citations.py): every cited ID exists in
     the evidence actually provided, and no free-form references appear.
  2. Numeric grounding (deterministic): every number in a cited claim appears
     in the evidence that claim cites.
  3. Claim support (model): per-claim verdicts from the verifier model.
  4. Coverage (deterministic): `verified` requires every checkable claim to
     have exactly one valid verdict, and every "supported" verdict to rely on
     evidence the claim itself cites. Missing, duplicate, invalid or
     unattributable verdicts (including an extra invalid verdict alongside
     otherwise complete coverage), uncited claims, claims over the limit and
     unconfirmed absence statements all make the result `incomplete`.

A valid citation report is not proof of grounding, and a verifier call that
fails yields `not_verified` — never `verified`.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Optional

from agent.schemas import AbstentionCheck, VerifierOutput
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
        # The model sees each passage's source name (e.g. "policy_2023.txt"), so
        # numbers in it are legitimately available to the answer.
        available = set().union(*(
            numbers_in(f"{e.text} {re.sub(r'[^0-9A-Za-z]+', ' ', e.source)}") for e in cited
        ))
        missing = sorted(numbers_in(claim) - available)
        if missing:
            problems.append(f"{claim} (not in cited evidence: {', '.join(missing)})")
    return problems


def _citation_issues(report: CitationReport) -> list[str]:
    issues = []
    if report.invalid_ids:
        issues.append(f"cites evidence that was not provided: {', '.join(report.invalid_ids)}")
    if report.non_evidence_citations:
        issues.append("contains references that are not evidence IDs")
    if not report.abstained and not report.valid_ids:
        issues.append("answer has no evidence citations")
    return issues


def assess(
    answer: str,
    evidence: list[Evidence],
    report: CitationReport,
    llm_output: Optional[VerifierOutput],
    *,
    claims: list[str],
    total_checkable: Optional[int] = None,
    absence_claims: Optional[list[str]] = None,
    absence_searched: bool = False,
    llm_error: Optional[str] = None,
    llm_skipped: bool = False,
) -> Verification:
    """Assess a non-abstaining answer.

    claims: the numbered, checkable claims shown to the verifier (absence
        statements excluded, at most the configured limit).
    total_checkable: number of checkable claims before the limit was applied.
    absence_claims: statements that information is missing; they cannot be
        grounded by citation and keep the result `incomplete` while present.
    """
    known = {e.eid for e in evidence}
    absence_claims = list(absence_claims or [])
    total_checkable = len(claims) if total_checkable is None else total_checkable
    issues = _citation_issues(report)
    numeric = numeric_mismatches(answer, evidence)

    unsupported: list[str] = []
    partial: list[str] = []
    unverified: list[str] = []
    invalid_verdicts = 0
    verified = 0
    followup = None
    llm_status = "skipped" if llm_skipped else ("completed" if llm_output is not None else "failed")

    if llm_output is not None:
        by_claim: dict[int, list] = defaultdict(list)
        for v in llm_output.verdicts:
            bad_ids = [i for i in v.evidence_ids if i not in known]
            if bad_ids:
                invalid_verdicts += 1
                issues.append(f"verifier referenced unknown evidence {', '.join(bad_ids)}; verdict ignored")
            elif v.claim_number > len(claims):
                invalid_verdicts += 1
                issues.append(f"verifier referenced claim {v.claim_number}, which does not exist; verdict ignored")
            else:
                by_claim[v.claim_number].append(v)

        for number, claim in enumerate(claims, start=1):
            verdicts = by_claim.get(number, [])
            cited = set(extract_citations(claim))
            if not verdicts:
                unverified.append(claim)
                issues.append(f"no verdict for claim {number}")
                continue
            if len(verdicts) > 1:
                unverified.append(claim)
                issues.append(f"{len(verdicts)} verdicts for claim {number}; treated as unresolved")
                continue
            v = verdicts[0]
            if v.status == "unsupported":
                unsupported.append(claim)
            elif v.status == "partially_supported":
                partial.append(claim)
            elif not cited:
                unverified.append(claim)
                issues.append(f"claim {number} has no citation, so its support cannot be attributed")
            elif not cited & set(v.evidence_ids):
                unverified.append(claim)
                issues.append(f"claim {number} judged supported by evidence it does not cite")
            else:
                verified += 1

        followup = (llm_output.followup_query or "").strip() or None
        if llm_output.missing_information:
            issues.append(f"missing information: {llm_output.missing_information}")

    over_limit = max(0, total_checkable - len(claims))
    if over_limit:
        issues.append(f"{over_limit} claim(s) beyond the verification limit were not checked")
    if absence_claims:
        where = "retrieved passages after a targeted search" if absence_searched else "retrieved passages"
        issues.append(
            f"{len(absence_claims)} statement(s) that information is missing: not found in the {where}, "
            "which does not establish that the documents lack it"
        )

    if llm_status != "completed":
        grounding = "unsupported" if numeric else "not_verified"
    elif unsupported or numeric:
        grounding = "unsupported"
    elif partial:
        grounding = "partially_verified"
    elif unverified or over_limit or absence_claims or not claims or invalid_verdicts:
        # A malformed verdict means the verifier's output cannot be fully trusted,
        # even if the remaining verdicts cover every claim.
        grounding = "incomplete"
    else:
        grounding = "verified"

    uncited_substantive = [c for c in report.uncited_claims if c not in absence_claims]
    if not followup and absence_claims and not absence_searched:
        followup = absence_claims[0][:300]   # bounded: at most one targeted search round

    if not report.ok or grounding in ("unsupported", "partially_verified") or uncited_substantive:
        decision = "revise"
    elif absence_claims and not absence_searched:
        decision = "revise"
    elif llm_output is not None and llm_output.recommendation != "accept" and followup:
        decision = "revise"
    elif grounding in ("not_verified", "incomplete"):
        decision = "finalize"
    else:
        decision = "accept"

    return Verification(
        citation_ok=report.ok,
        grounding=grounding,
        llm_status=llm_status,
        decision=decision,
        unsupported_claims=unsupported + partial,
        numeric_mismatches=numeric,
        unverified_claims=unverified,
        absence_claims=absence_claims,
        absence_searched=absence_searched,
        claims_checked=total_checkable,
        claims_verified=verified,
        issues=issues,
        followup_query=followup if decision == "revise" else None,
        llm_error=llm_error,
    )


def assess_abstention(
    evidence: list[Evidence],
    report: CitationReport,
    check: Optional[AbstentionCheck],
    *,
    attempted: bool,
    llm_error: Optional[str] = None,
) -> Verification:
    """Assess a full abstention. Producing the abstention phrase is not proof it was right."""
    issues = _citation_issues(report)
    known = {e.eid for e in evidence}
    decision = "accept"
    llm_status = "skipped"

    if not evidence:
        outcome = "no_evidence"
    elif not attempted:
        outcome, decision = "not_checked", "finalize"
        issues.append("evidence was retrieved but the abstention was not checked against it")
    elif check is None:
        outcome, llm_status, decision = "failed", "failed", "finalize"
        issues.append("abstention check failed")
    else:
        llm_status = "completed"
        cited = [i for i in check.evidence_ids if i in known]
        if check.answerable in ("yes", "partially") and cited:
            outcome, decision = "evidence_may_answer", "revise"
            issues.append(f"verifier found evidence that may answer the question: {', '.join(cited)}")
        elif check.answerable in ("yes", "partially"):
            outcome, decision = "failed", "finalize"
            issues.append("verifier said the question is answerable but cited no valid evidence")
        else:
            outcome = "confirmed"

    return Verification(
        citation_ok=report.ok,
        grounding="skipped",
        llm_status=llm_status,
        decision=decision,
        abstention_check=outcome,
        issues=issues,
        llm_error=llm_error,
    )


def revision_feedback(v: Verification) -> str:
    """Instructions for the next synthesis round, built from verification findings."""
    lines = ["Your previous answer had these problems. Fix them using only the evidence:"]
    lines += [f"- {i}" for i in v.issues]
    lines += [f"- not supported by the evidence: {c}" for c in v.unsupported_claims[:6]]
    lines += [f"- number not found in cited evidence: {m}" for m in v.numeric_mismatches[:6]]
    if v.absence_claims:
        lines.append("- check whether the evidence (including any new passages) contains the information "
                     "you said was missing: " + "; ".join(v.absence_claims[:3]))
    if v.abstention_check == "evidence_may_answer":
        lines.append("- the evidence listed above may answer the question; answer from it with citations, "
                     "or explain precisely what is missing")
    lines.append("Every factual statement needs an evidence citation. Remove or correct unsupported "
                 "statements. If the evidence is insufficient, say so.")
    return "\n".join(lines)
