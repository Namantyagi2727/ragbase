"""Verification must be complete before an answer is called verified.

Scripted-model regression tests for claim coverage, absence statements and
abstention handling, including the driver's-manual failure seen live.
"""

import pytest

from agent import AgentSettings, run_agent
from citations import ABSTAIN_TEXT
from tests.unit.agent_helpers import MemoryRetriever, RoleModel

COMPLEX = {"route": "complex", "subqueries": [{"query": "2023 meal allowance"}, {"query": "2024 meal allowance"}]}
SIMPLE = {"route": "simple"}
TWO_FACTS = "In 2023 the meal allowance was 45 dollars [E1]. In 2024 it rose to 60 dollars [E2]."


def verdict(n, status="supported", ids=("E1",)):
    return {"claim_number": n, "status": status, "evidence_ids": list(ids)}


def _run(model, question="Compare the meal allowance in 2023 and 2024.", retriever=None, **kw):
    retriever = retriever or MemoryRetriever()
    settings = AgentSettings(**{"k_per_query": 1, **kw})
    return run_agent(question, retriever, model, settings=settings), retriever


# --- 1. complete claim coverage --------------------------------------------------

def test_one_verdict_for_two_claims_is_not_verification():
    model = RoleModel(planner=[COMPLEX], synth=[TWO_FACTS],
                      verifier=[{"verdicts": [verdict(1)], "recommendation": "accept"}])
    result, _ = _run(model)
    v = result.verification
    assert v.grounding == "incomplete" and (v.claims_verified, v.claims_checked) == (1, 2)
    assert v.unverified_claims == ["In 2024 it rose to 60 dollars [E2]."]
    assert result.status == "completed_unverified"


def test_duplicate_verdicts_leave_the_claim_unresolved():
    dup = {"verdicts": [verdict(1), verdict(1, "unsupported"), verdict(2, ids=["E2"])], "recommendation": "accept"}
    result, _ = _run(RoleModel(planner=[COMPLEX], synth=[TWO_FACTS], verifier=[dup]))
    assert result.verification.grounding == "incomplete"
    assert any("2 verdicts for claim 1" in i for i in result.verification.issues)


def test_verdict_with_unknown_evidence_does_not_count():
    bad = {"verdicts": [verdict(1), verdict(2, ids=["E9"])], "recommendation": "accept"}
    result, _ = _run(RoleModel(planner=[COMPLEX], synth=[TWO_FACTS], verifier=[bad]))
    assert result.verification.grounding == "incomplete" and result.verification.claims_verified == 1


def test_supported_verdict_must_rely_on_the_claims_own_citation():
    # Claim 2 cites E2, but the verifier says E1 supports it.
    crossed = {"verdicts": [verdict(1), verdict(2, ids=["E1"])], "recommendation": "accept"}
    result, _ = _run(RoleModel(planner=[COMPLEX], synth=[TWO_FACTS], verifier=[crossed]))
    v = result.verification
    assert v.grounding == "incomplete"
    assert any("evidence it does not cite" in i for i in v.issues)


def test_full_attributable_coverage_is_verified():
    ok = {"verdicts": [verdict(1), verdict(2, ids=["E2"])], "recommendation": "accept"}
    result, _ = _run(RoleModel(planner=[COMPLEX], synth=[TWO_FACTS], verifier=[ok]))
    assert result.verification.grounding == "verified" and result.verification.coverage == 1.0
    assert result.status == "completed"


def test_uncited_substantive_claim_gets_one_revision_then_stays_incomplete():
    uncited = "In 2023 the meal allowance was 45 dollars [E1]. In 2024 the allowance rose to sixty dollars."
    model = RoleModel(planner=[COMPLEX], synth=[uncited],
                      verifier=[{"verdicts": [verdict(1), verdict(2, ids=["E2"])], "recommendation": "accept"}])
    result, _ = _run(model, max_revisions=1)
    assert model.calls.count("synth") == 2           # exactly one bounded revision
    assert result.verification.grounding == "incomplete"
    assert result.status == "completed_unverified"


def test_claims_beyond_the_verification_limit_are_reported():
    ok = {"verdicts": [verdict(1)], "recommendation": "accept"}
    result, _ = _run(RoleModel(planner=[COMPLEX], synth=[TWO_FACTS], verifier=[ok]), max_verified_claims=1)
    v = result.verification
    assert v.grounding == "incomplete" and v.claims_checked == 2
    assert any("beyond the verification limit" in i for i in v.issues)


# --- 2. absence statements ---------------------------------------------------------

DMV = {
    "dmv_manual.pdf": [
        "Penalties: a driver under 21 convicted of an alcohol related violation faces a minimum one year revocation.",
        "Zero Tolerance Law: a driver under age 21 will have their license suspended for six months "
        "if found to have a blood alcohol content from .02 to .07.",
        "Parallel parking requires signalling and checking mirrors.",
    ]
}
DMV_Q = "What is the blood alcohol limit for drivers under 21, and what are the penalties?"
DMV_PLAN = {"route": "complex", "subqueries": [{"query": "penalties driver under 21 alcohol violation"}]}
FIRST = ("Drivers under 21 convicted of an alcohol related violation face a minimum one year revocation [E1]. "
         "The evidence does not specify a blood alcohol limit for drivers under 21.")
FIXED = ("Drivers under 21 convicted of an alcohol related violation face a minimum one year revocation [E1]. "
         "Under the Zero Tolerance Law, a driver under 21 with a blood alcohol content from .02 to .07 "
         "has their license suspended for six months [E2].")


def test_driver_manual_regression_absence_claim_triggers_targeted_search():
    """Live failure: "the evidence does not specify a limit" slipped through while page 21 had it."""
    model = RoleModel(
        planner=[DMV_PLAN], synth=[FIRST, FIXED],
        verifier=[{"verdicts": [verdict(1)], "recommendation": "accept"},
                  {"verdicts": [verdict(1), verdict(2, ids=["E2"])], "recommendation": "accept"}],
    )
    result, retriever = _run(model, DMV_Q, retriever=MemoryRetriever(DMV))
    assert retriever.queries[-1][0].startswith("The evidence does not specify a blood alcohol limit")
    assert any("Zero Tolerance" in e.text for e in result.evidence)
    assert result.verification.grounding == "verified" and result.status == "completed"
    assert result.revision_count == 1


def test_absence_claim_that_survives_the_search_is_never_verified():
    model = RoleModel(planner=[DMV_PLAN], synth=[FIRST],
                      verifier=[{"verdicts": [verdict(1)], "recommendation": "accept"}])
    result, _ = _run(model, DMV_Q, retriever=MemoryRetriever(DMV), max_revisions=3)
    v = result.verification
    assert model.calls.count("synth") == 2           # one targeted round, no further loop
    assert v.absence_searched and v.grounding == "incomplete"
    assert result.status == "completed_unverified"
    assert "one targeted search" in result.answer and "do not establish" in result.answer


def test_absence_claim_without_revision_budget_is_reported_as_unsearched():
    model = RoleModel(planner=[DMV_PLAN], synth=[FIRST],
                      verifier=[{"verdicts": [verdict(1)], "recommendation": "accept"}])
    result, _ = _run(model, DMV_Q, retriever=MemoryRetriever(DMV), max_revisions=0)
    v = result.verification
    assert not v.absence_searched and v.grounding == "incomplete"
    assert any("not found in the retrieved passages," in i for i in v.issues)


# --- 3. abstentions -------------------------------------------------------------------

def test_abstention_with_answering_evidence_is_challenged_and_revised():
    model = RoleModel(
        planner=[COMPLEX], synth=[ABSTAIN_TEXT, TWO_FACTS],
        abscheck=[{"answerable": "yes", "evidence_ids": ["E1", "E2"]}],
        verifier=[{"verdicts": [verdict(1), verdict(2, ids=["E2"])], "recommendation": "accept"}],
    )
    result, _ = _run(model)
    assert model.calls == ["planner", "synth", "abscheck", "synth", "verifier"]
    assert result.status == "completed" and result.revision_count == 1


def test_challenged_abstention_without_revision_budget_is_qualified():
    model = RoleModel(planner=[COMPLEX], synth=[ABSTAIN_TEXT],
                      abscheck=[{"answerable": "partially", "evidence_ids": ["E2"]}])
    result, _ = _run(model, max_revisions=0)
    assert result.verification.abstention_check == "evidence_may_answer"
    assert result.status == "qualified" and "may contain an answer" in result.answer


def test_abstention_confirmed_by_check():
    model = RoleModel(planner=[COMPLEX], synth=[ABSTAIN_TEXT], abscheck=[{"answerable": "no"}])
    result, _ = _run(model)
    assert result.verification.abstention_check == "confirmed" and result.status == "completed"


@pytest.mark.parametrize("reply", ["not json", {"answerable": "yes", "evidence_ids": ["E42"]}])
def test_failed_or_unsupported_abstention_check_is_not_a_confirmation(reply):
    model = RoleModel(planner=[COMPLEX], synth=[ABSTAIN_TEXT], abscheck=[reply])
    result, _ = _run(model)
    assert result.verification.abstention_check == "failed"
    assert result.status == "completed_unverified"


def test_simple_route_abstention_with_evidence_is_not_checked():
    model = RoleModel(planner=[SIMPLE], standard=[ABSTAIN_TEXT])
    result, _ = _run(model, "What is the parking policy?")
    assert result.verification.abstention_check == "not_checked"
    assert result.status == "completed_unverified"


def test_abstention_with_no_evidence_is_labelled_no_evidence():
    model = RoleModel(planner=[SIMPLE], standard=[ABSTAIN_TEXT])
    result, _ = _run(model, "What is the parking policy?", retriever=MemoryRetriever(corpus={}))
    assert result.verification.abstention_check == "no_evidence"
    assert result.status == "completed"


@pytest.mark.parametrize("extra", [
    verdict(1, ids=["E9"]),   # unknown evidence ID
    verdict(7),               # nonexistent claim number
])
def test_extra_invalid_verdict_prevents_full_verification(extra):
    """Every claim is validly covered, but one additional verdict is malformed."""
    output = {"verdicts": [verdict(1), verdict(2, ids=["E2"]), extra], "recommendation": "accept"}
    result, _ = _run(RoleModel(planner=[COMPLEX], synth=[TWO_FACTS], verifier=[output]))
    v = result.verification
    assert (v.claims_verified, v.claims_checked) == (2, 2)
    assert v.grounding == "incomplete"
    assert result.status == "completed_unverified"
    assert any("verdict ignored" in i for i in v.issues)
