"""Full abstention vs. substantive or partial answers (regression for benchmark case M03).

M03's answer stated both requested facts but contained "the evidence does not
specify ...", and was classified as a full abstention.
"""

import json

from agent import AgentSettings, run_agent
from citations import ABSTAIN_TEXT, Evidence, is_full_abstention, validate_citations
from tests.unit.agent_helpers import MemoryRetriever, RoleModel

M03_ANSWER = (
    "According to the employee handbook, the primary caregiver receives 16 weeks of fully paid parental leave, "
    "but the evidence does not specify the total amount of paid leave for dependants.\n\n"
    "The benefits FAQ states that the company pays 50% of the health insurance premium for dependants, "
    "but it does not specify the total amount of paid leave for primary caregivers.\n\n"
    "> ⚠ Verification: the retrieved passages may contain an answer to this question."
)
EVIDENCE = [Evidence("1", "c1", "handbook.docx", "p.1", "Primary caregivers receive 16 weeks of fully paid leave."),
            Evidence("2", "c2", "benefits.txt", "", "The company pays 50 percent of the premium for dependants.")]


def test_m03_answer_with_facts_and_a_gap_is_not_a_full_abstention():
    assert not is_full_abstention(M03_ANSWER)
    report = validate_citations(M03_ANSWER, EVIDENCE)
    assert not report.abstained


def test_ordinary_full_abstentions_are_still_abstentions():
    assert is_full_abstention(ABSTAIN_TEXT)
    assert is_full_abstention("The provided documents do not contain information about pets.")
    v05_style = (f"{ABSTAIN_TEXT} The evidence only mentions the reimbursement rate for personal car use, "
                 "but it does not specify the mileage reimbursement rate.")
    assert is_full_abstention(v05_style)
    assert validate_citations(ABSTAIN_TEXT, EVIDENCE).abstained


def test_partial_answer_with_citation_and_stated_gap_is_substantive():
    answer = ("The company matches contributions up to 4 percent of base salary [1]. "
              "The documents do not describe a vesting schedule.")
    report = validate_citations(answer, EVIDENCE)
    assert not report.abstained and report.valid_ids == ["1"]


def test_facts_without_citations_are_reported_and_never_treated_as_valid():
    synthesized = M03_ANSWER.split("\n\n> ")[0]     # the verification note is appended after validation
    report = validate_citations(synthesized, EVIDENCE)
    assert not report.abstained
    assert not report.ok                         # neither cited nor an abstention
    assert len(report.uncited_claims) == 2       # the two factual sentences are reported as uncited
    assert report.citation_coverage == 0.0


def test_mixed_fact_and_absence_sentence_is_sent_for_verification():
    seen = []

    def verifier(messages):
        seen.append(messages[-1]["content"])
        return json.dumps({"verdicts": [{"claim_number": 1, "status": "unsupported", "evidence_ids": ["1"]}],
                           "recommendation": "accept"})

    answer = ("In 2023 the meal allowance was 99 dollars [1], but the evidence does not specify the allowance "
              "for interns.")
    plan = {"route": "complex", "subqueries": [{"query": "2023 meal allowance"}]}
    model = RoleModel(planner=[plan], synth=[answer], verifier=[verifier])
    result = run_agent("What was the 2023 meal allowance?", MemoryRetriever(), model,
                       settings=AgentSettings(k_per_query=1, max_revisions=0))
    assert seen and "99 dollars" in seen[0]          # the factual part was checked, not skipped
    assert result.verification.grounding != "verified"
    assert result.status in ("qualified", "completed_unverified")


def test_unsupported_pure_absence_claim_stays_unverified():
    answer = "In 2023 the meal allowance was 45 dollars [1]. The evidence does not provide the allowance for interns."
    accept = {"verdicts": [{"claim_number": 1, "status": "supported", "evidence_ids": ["1"]}],
              "recommendation": "accept"}
    plan = {"route": "complex", "subqueries": [{"query": "2023 meal allowance"}]}
    model = RoleModel(planner=[plan], synth=[answer], verifier=[accept])
    result = run_agent("What was the 2023 meal allowance?", MemoryRetriever(), model,
                       settings=AgentSettings(k_per_query=1, max_revisions=0))
    assert result.verification.grounding == "incomplete"
    assert result.status == "completed_unverified"


def test_source_lists_after_an_abstention_are_not_substantive_content():
    answer = ("I don't have enough information to answer that.\n\nSources:\n"
              "- Fleet Driver Safety Handbook [filename @ location]\n- Travel Policy (2024 edition) [filename @ location]")
    assert is_full_abstention(answer)
    partial = ("The international daily meal allowance under the 2024 policy is $90.\n\n"
               "I don't have enough information to answer the question about visa costs.\n\nSources:\n- Travel Policy")
    assert not is_full_abstention(partial)
