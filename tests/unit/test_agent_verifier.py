"""Deterministic parts of verification."""

import pytest

from agent.schemas import ClaimVerdict, VerifierOutput
from agent.verifier import assess, numbers_in, numeric_mismatches, revision_feedback
from citations import Evidence, validate_citations

EV = [
    Evidence("E1", "a", "p.txt", "txt", "The limit is .08 percent for adults and 1,200 dollars fines."),
    Evidence("E2", "b", "q.txt", "txt", "Leave accrues at 1.5 days per month, 18 days a year."),
]


@pytest.mark.parametrize("text,expected", [
    (".08 percent", {"0.08"}),
    ("0.08 percent", {"0.08"}),
    ("1,200 dollars", {"1200"}),
    ("45% of claims", {"45"}),
    ("1.50 days", {"1.5"}),
    ("cited [E12] only", set()),
])
def test_number_normalization(text, expected):
    assert numbers_in(text) == expected


def test_numbers_present_in_cited_evidence_pass():
    assert numeric_mismatches("The adult limit is 0.08 percent [E1].", EV) == []


def test_number_from_uncited_evidence_is_a_mismatch():
    problems = numeric_mismatches("Leave accrues at 1.5 days per month [E1].", EV)
    assert problems and "1.5" in problems[0]


def test_uncited_claims_are_not_numerically_checked():
    assert numeric_mismatches("The fine is 9,999 dollars for everyone here.", EV) == []


def _assess(answer, verdicts=None, recommendation="accept", **kw):
    out = None if verdicts is None else VerifierOutput(
        verdicts=[ClaimVerdict(**v) for v in verdicts], recommendation=recommendation, **kw)
    return assess(answer, EV, validate_citations(answer, EV), out)


def test_verdicts_citing_unknown_evidence_are_ignored_and_reported():
    v = _assess("The adult limit is 0.08 percent [E1].",
                [{"claim": "x", "status": "supported", "evidence_ids": ["E7"]}])
    assert v.grounding == "not_verified"
    assert any("unknown evidence E7" in i for i in v.issues)


def test_partial_support_requests_revision():
    v = _assess("The adult limit is 0.08 percent [E1].",
                [{"claim": "x", "status": "partially_supported", "evidence_ids": ["E1"]}])
    assert v.grounding == "partially_verified" and v.decision == "revise"


def test_abstention_with_followup_query_requests_revision():
    v = _assess("I don't have enough information in the provided documents to answer that.",
                [], recommendation="insufficient_evidence", followup_query="parking policy")
    assert v.decision == "revise" and v.followup_query == "parking policy"


def test_feedback_lists_concrete_problems():
    v = _assess("Leave accrues at 1.5 days per month [E1].",
                [{"claim": "Leave accrues", "status": "unsupported", "evidence_ids": ["E1"]}])
    fb = revision_feedback(v)
    assert "not supported by the evidence: Leave accrues" in fb and "1.5" in fb
