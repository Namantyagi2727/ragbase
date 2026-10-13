"""Agent graph: routing, transitions, tool use, bounded refinement, failures, traces."""

import json

import pytest

from agent import AgentSettings, run_agent
from agent.graph import recursion_limit
from agent.tools import RetrievalTools, ToolError
from llm import LLMUnavailableError
from tests.unit.agent_helpers import MemoryRetriever, RoleModel

SIMPLE = {"route": "simple", "rationale": "single lookup"}
COMPLEX = {
    "route": "complex",
    "rationale": "two policies",
    "subqueries": [{"query": "2023 meal allowance"}, {"query": "2024 meal allowance"}],
}
TWO_FACTS = "In 2023 the meal allowance was 45 dollars [E1]. In 2024 it rose to 60 dollars [E2]."
ACCEPT = {
    "verdicts": [
        {"claim_number": 1, "status": "supported", "evidence_ids": ["E1"]},
        {"claim_number": 2, "status": "supported", "evidence_ids": ["E2"]},
    ],
    "recommendation": "accept",
}


def _settings(**kw):
    return AgentSettings(**{"k_per_query": 1, **kw})


def _run(model, question="Compare the meal allowance in 2023 and 2024.", retriever=None, **settings):
    retriever = retriever or MemoryRetriever()
    return run_agent(question, retriever, model, settings=_settings(**settings)), retriever


def _nodes(result):
    return [e["node"] for e in result.trace]


# --- routing ------------------------------------------------------------------

def test_simple_route_uses_standard_rag_path_and_skips_claim_verification():
    model = RoleModel(planner=[SIMPLE], standard=["The 2024 meal allowance is 60 dollars [E1]."])
    result, _ = _run(model, "What is the 2024 meal allowance?")
    assert result.route == "simple"
    assert model.calls == ["planner", "standard"]
    assert result.verification.llm_status == "skipped"
    assert result.status == "completed_unverified"  # never reported as verified
    assert _nodes(result) == ["plan", "retrieve", "answer", "finalize"]
    assert result.trace[0]["data"]["route"] == "simple"


def test_invalid_planner_output_falls_back_to_standard_and_is_recorded():
    model = RoleModel(planner=["not json", "still not json"], standard=["Sixty dollars per day [E1]."])
    result, _ = _run(model, "What is the 2024 meal allowance?")
    assert result.route == "simple"
    assert result.planner_fallback and "invalid structured output" in result.planner_fallback
    assert result.trace[0]["data"]["fallback"] == result.planner_fallback
    assert model.calls == ["planner", "planner", "standard"]


def test_planner_backend_failure_stops_the_run():
    model = RoleModel(planner=[LLMUnavailableError("ollama down")])
    result, _ = _run(model)
    assert result.status == "failed"
    assert "ollama down" in result.error and "could not complete" in result.answer
    assert model.calls == ["planner"]
    assert _nodes(result) == ["plan", "finalize"]


# --- complex path -----------------------------------------------------------

def test_complex_route_runs_planned_searches_synthesizes_and_verifies():
    model = RoleModel(planner=[COMPLEX], synth=[TWO_FACTS], verifier=[ACCEPT])
    result, retriever = _run(model)
    assert result.route == "complex" and result.status == "completed"
    assert model.calls == ["planner", "synth", "verifier"]
    assert [q for q, _ in retriever.queries] == ["2023 meal allowance", "2024 meal allowance"]
    assert [e.chunk_id for e in result.evidence] == ["policy_2023.txt#0", "policy_2024.txt#0"]
    assert result.verification.grounding == "verified" and result.verification.citation_ok
    assert (result.llm_calls, result.retrieval_calls) == (3, 2)
    assert _nodes(result) == ["plan", "retrieve", "synthesize", "verify", "finalize"]


def test_compare_documents_searches_each_document_separately():
    plan = {"route": "complex", "compare_documents": True, "subqueries": [{"query": "meal allowance"}]}
    model = RoleModel(planner=[plan], synth=[TWO_FACTS], verifier=[ACCEPT])
    result, retriever = _run(model, k_per_query=4)
    searched_sources = [s for _, s in retriever.queries]
    assert searched_sources == [("handbook.txt",), ("policy_2023.txt",), ("policy_2024.txt",)]
    assert result.retrieval_calls == 3


def test_planner_cannot_widen_search_with_unknown_sources():
    plan = {"route": "complex", "subqueries": [{"query": "meal allowance", "sources": ["../../etc/passwd"]}]}
    model = RoleModel(planner=[plan], synth=[TWO_FACTS], verifier=[ACCEPT])
    result, retriever = _run(model)
    search = result.trace[1]["data"]["searches"][0]
    assert search["dropped_sources"] == ["../../etc/passwd"] and search["sources"] == []
    assert retriever.queries[0][1] == ()


def test_complex_route_with_no_evidence_abstains_without_calling_the_synthesizer():
    model = RoleModel(planner=[COMPLEX])
    result, _ = _run(model, retriever=MemoryRetriever(corpus={}))
    assert result.citations.abstained and result.status == "completed"
    assert model.calls == ["planner"]
    assert result.verification.grounding == "skipped"


# --- verification is separate from citation validity -----------------------

def test_valid_citations_are_not_enough_when_verifier_finds_unsupported_claims():
    unsupported = {
        "verdicts": [{"claim_number": 2, "status": "unsupported", "evidence_ids": ["E2"]}],
        "recommendation": "revise",
    }
    model = RoleModel(planner=[COMPLEX], synth=[TWO_FACTS], verifier=[unsupported])
    result, _ = _run(model, max_revisions=0)
    assert result.citations.ok                       # citations are valid...
    assert result.verification.grounding == "unsupported"
    assert result.status == "qualified"              # ...but the answer is not verified
    assert "⚠ Verification" in result.answer


def test_failed_verifier_output_is_never_treated_as_verified():
    model = RoleModel(planner=[COMPLEX], synth=[TWO_FACTS], verifier=["garbage", "more garbage"])
    result, _ = _run(model)
    v = result.verification
    assert v.llm_status == "failed" and v.grounding == "not_verified" and v.llm_error
    assert result.status == "completed_unverified"
    assert "refine" not in _nodes(result)


def test_verifier_backend_error_degrades_to_unverified_not_failed():
    model = RoleModel(planner=[COMPLEX], synth=[TWO_FACTS], verifier=[LLMUnavailableError("down")])
    result, _ = _run(model)
    assert result.status == "completed_unverified"
    assert result.verification.llm_status == "failed"


def test_empty_verdict_list_is_not_verification():
    model = RoleModel(planner=[COMPLEX], synth=[TWO_FACTS],
                      verifier=[{"verdicts": [], "recommendation": "accept"}])
    result, _ = _run(model)
    assert result.verification.grounding == "not_verified"
    assert result.status == "completed_unverified"


def test_number_absent_from_cited_evidence_is_caught_deterministically():
    wrong = "In 2023 the meal allowance was 75 dollars [E1]. In 2024 it rose to 60 dollars [E2]."
    model = RoleModel(planner=[COMPLEX], synth=[wrong], verifier=[ACCEPT])  # verifier wrongly accepts
    result, _ = _run(model, max_revisions=0)
    assert result.verification.numeric_mismatches and "75" in result.verification.numeric_mismatches[0]
    assert result.status == "qualified"


# --- bounded refinement ---------------------------------------------------------

REVISE = {
    "verdicts": [{"claim_number": 2, "status": "unsupported", "evidence_ids": ["E2"]}],
    "followup_query": "2024 hotel approval",
    "recommendation": "revise",
}


def test_refinement_runs_a_targeted_search_and_reverifies():
    model = RoleModel(planner=[COMPLEX], synth=[TWO_FACTS, TWO_FACTS], verifier=[REVISE, ACCEPT])
    result, retriever = _run(model)
    assert result.revision_count == 1 and result.status == "completed"
    assert retriever.queries[-1][0] == "2024 hotel approval"
    assert _nodes(result) == [
        "plan", "retrieve", "synthesize", "verify", "refine", "retrieve", "synthesize", "verify", "finalize",
    ]
    assert len(result.evidence) == 3  # follow-up evidence appended with continued numbering
    assert result.evidence[-1].eid == "E3"


def test_revision_prompt_carries_verifier_feedback():
    captured = []

    def synth(messages):
        captured.append(messages[-1]["content"])
        return TWO_FACTS

    model = RoleModel(planner=[COMPLEX], synth=[synth], verifier=[REVISE, ACCEPT])
    _run(model)
    assert "problems" not in captured[0]
    assert "not supported by the evidence: In 2024 it rose to 60 dollars [E2]." in captured[1]


@pytest.mark.parametrize("max_revisions", [0, 1, 2])
def test_revision_rounds_never_exceed_the_limit(max_revisions):
    model = RoleModel(planner=[COMPLEX], synth=[TWO_FACTS], verifier=[REVISE])
    result, _ = _run(model, max_revisions=max_revisions)
    assert result.revision_count == max_revisions
    assert model.calls.count("synth") == max_revisions + 1
    assert result.status == "qualified"


def test_invalid_citation_triggers_resynthesis_without_new_search():
    bad = "In 2023 the meal allowance was 45 dollars [E9]."
    model = RoleModel(planner=[COMPLEX], synth=[bad, TWO_FACTS], verifier=[ACCEPT])
    result, retriever = _run(model)
    assert result.revision_count == 1 and result.status == "completed"
    assert len(retriever.queries) == 2  # only the two planned searches


def test_no_revision_when_time_budget_is_nearly_spent():
    model = RoleModel(planner=[COMPLEX], synth=[TWO_FACTS], verifier=[REVISE])
    result, _ = _run(model, min_time_for_revision_s=10_000)
    assert result.revision_count == 0


# --- failures and termination -------------------------------------------------

def test_synthesis_backend_failure_stops_with_failed_status():
    model = RoleModel(planner=[COMPLEX], synth=[LLMUnavailableError("down")])
    result, _ = _run(model)
    assert result.status == "failed" and "verify" not in _nodes(result)


def test_exhausted_time_budget_stops_before_any_model_call():
    model = RoleModel(planner=[COMPLEX])
    result, _ = _run(model, time_budget_s=0)
    assert result.status == "timeout" and model.calls == []


def test_recursion_limit_covers_the_longest_bounded_path():
    model = RoleModel(planner=[COMPLEX], synth=[TWO_FACTS], verifier=[REVISE])
    result, _ = _run(model, max_revisions=3)
    assert result.status == "qualified" and result.revision_count == 3
    assert recursion_limit(_settings(max_revisions=3)) >= len(result.trace)


# --- trace and result shape ---------------------------------------------------------

def test_trace_is_ordered_and_serializable():
    model = RoleModel(planner=[COMPLEX], synth=[TWO_FACTS], verifier=[ACCEPT])
    result, _ = _run(model)
    assert [e["step"] for e in result.trace] == list(range(1, len(result.trace) + 1))
    assert all(e["summary"] for e in result.trace)
    data = json.loads(json.dumps(result.to_dict()))
    assert data["status"] == "completed" and data["verification"]["grounding"] == "verified"
    assert data["prompt_tokens"] == 30 and data["completion_tokens"] == 15


# --- tools ------------------------------------------------------------------------

def test_agent_has_only_read_only_retrieval_tools():
    assert RetrievalTools.NAMES == ("hybrid_search", "list_documents")
    public = {n for n in dir(RetrievalTools) if not n.startswith("_") and n != "NAMES"}
    assert public == {"hybrid_search", "list_documents"}


@pytest.mark.parametrize("kwargs", [{"query": "   "}, {"query": "x" * 301}, {"query": "ok", "k": 0}, {"query": "ok", "k": 99}])
def test_tool_rejects_invalid_arguments(kwargs):
    with pytest.raises(ToolError):
        RetrievalTools(MemoryRetriever()).hybrid_search(**kwargs)


def test_trace_steps_are_unique_on_the_simple_route():
    model = RoleModel(planner=[SIMPLE], standard=["Sixty dollars per day [E1]."])
    result, _ = _run(model, "What is the 2024 meal allowance?")
    assert [e["step"] for e in result.trace] == [1, 2, 3, 4]


def test_file_name_prefixes_are_stripped_from_planned_queries():
    plan = {"route": "complex", "subqueries": [{"query": "policy_2024.txt: meal allowance"},
                                                {"query": "handbook annual leave"}]}
    model = RoleModel(planner=[plan], synth=[TWO_FACTS], verifier=[ACCEPT])
    _, retriever = _run(model)
    assert [q for q, _ in retriever.queries] == ["meal allowance", "annual leave"]


def test_verdict_for_nonexistent_claim_number_is_ignored():
    bogus = {"verdicts": [{"claim_number": 9, "status": "supported", "evidence_ids": ["E1"]}],
             "recommendation": "accept"}
    model = RoleModel(planner=[COMPLEX], synth=[TWO_FACTS], verifier=[bogus])
    result, _ = _run(model)
    assert result.verification.grounding == "not_verified"
    assert any("claim 9" in i for i in result.verification.issues)


def test_absence_statements_are_not_sent_for_claim_verification():
    seen = []

    def verifier(messages):
        seen.append(messages[-1]["content"])
        return json.dumps({"verdicts": [{"claim_number": 1, "status": "supported", "evidence_ids": ["E1"]}],
                           "recommendation": "accept"})

    answer = ("In 2023 the meal allowance was 45 dollars [E1]. "
              "The evidence does not provide the allowance for interns.")
    model = RoleModel(planner=[COMPLEX], synth=[answer], verifier=[verifier])
    result, _ = _run(model)
    assert "interns" not in seen[0]
    assert result.verification.grounding == "verified"
    assert any("missing information were not claim-verified" in i for i in result.verification.issues)
