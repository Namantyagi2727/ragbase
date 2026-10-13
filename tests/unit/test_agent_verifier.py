"""Deterministic parts of verification."""

import pytest

from agent.schemas import ClaimVerdict, VerifierOutput
from agent.verifier import assess, numbers_in, numeric_mismatches, revision_feedback
from citations import Evidence, split_claims, validate_citations

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
    return assess(answer, EV, validate_citations(answer, EV), out, claims=split_claims(answer))


def test_verdicts_citing_unknown_evidence_are_ignored_and_reported():
    v = _assess("The adult limit is 0.08 percent [E1].",
                [{"claim_number": 1, "status": "supported", "evidence_ids": ["E7"]}])
    assert v.grounding == "incomplete" and v.claims_verified == 0
    assert any("unknown evidence E7" in i for i in v.issues)


def test_partial_support_requests_revision():
    v = _assess("The adult limit is 0.08 percent [E1].",
                [{"claim_number": 1, "status": "partially_supported", "evidence_ids": ["E1"]}])
    assert v.grounding == "partially_verified" and v.decision == "revise"


def test_supported_but_incomplete_answer_with_followup_requests_revision():
    v = _assess("The adult limit is 0.08 percent [E1].",
                [{"claim_number": 1, "status": "supported", "evidence_ids": ["E1"]}],
                recommendation="insufficient_evidence", followup_query="limit for drivers under 21")
    assert v.grounding == "verified" and v.decision == "revise"
    assert v.followup_query == "limit for drivers under 21"


def test_verifier_schema_requires_claim_verdicts():
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        VerifierOutput(verdicts=[], recommendation="accept")


def test_feedback_lists_concrete_problems():
    v = _assess("Leave accrues at 1.5 days per month [E1].",
                [{"claim_number": 1, "status": "unsupported", "evidence_ids": ["E1"]}])
    fb = revision_feedback(v)
    assert "not supported by the evidence: Leave accrues" in fb and "1.5" in fb


def test_numbers_in_the_cited_source_name_count_as_available():
    ev = [Evidence("E1", "a", "policy_2023.txt", "txt", "Reports are due within 30 days.")]
    assert numeric_mismatches("Under the 2023 policy, reports are due within 30 days [E1].", ev) == []
