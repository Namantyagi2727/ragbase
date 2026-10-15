"""Comparison-mode document selection (regression for the official Phase 3 benchmark).

The agent used to search the first max_compare_docs documents in alphabetical
order, so with more than five documents the relevant policies were never
searched. These tests use a larger corpus whose irrelevant documents sort
before the relevant ones, plus similarly named policy versions.
"""

import pytest

from agent import AgentSettings, run_agent
from agent.sources import named_sources, select_compare_sources
from agent.tools import RetrievalTools
from tests.unit.agent_helpers import MemoryRetriever, RoleModel

CORPUS = {
    # irrelevant documents that sort alphabetically before the travel policies
    "a_benefits_faq.txt": ["The wellness stipend is 50 dollars per month."],
    "b_driver_handbook.txt": ["Collisions must be reported to the fleet coordinator."],
    "c_employee_handbook.txt": ["Employees accrue annual leave every month."],
    "d_it_security_policy.txt": ["Lost laptops must be reported within 24 hours."],
    "e_remote_work_addendum_2024.txt": ["Remote work is allowed three days per week from 2024."],
    "f_vendor_contract.txt": ["The vendor contract renews every year."],
    # similarly named versions
    "travel_policy_2022.txt": ["The 2022 travel policy expense report deadline was 45 days."],
    "travel_policy_2023.txt": [
        "The 2023 travel policy expense report deadline is 30 days after the trip.",
        "The 2023 travel policy meal allowance is 45 dollars per day.",
    ],
    "travel_policy_2024.txt": [
        "The 2024 travel policy expense report deadline is 15 days after the trip.",
        "The 2024 travel policy meal allowance is 60 dollars per day.",
    ],
}
DOCS = sorted(CORPUS)
ANSWER = "The deadline was 30 days in 2023 [1] and 15 days in 2024 [2]."
ACCEPT = {"verdicts": [{"claim_number": 1, "status": "supported", "evidence_ids": ["1", "2"]}],
          "recommendation": "accept"}


def _compare_plan(query="expense report deadline", sources=()):
    return {"route": "complex", "compare_documents": True,
            "subqueries": [{"query": query, "sources": list(sources)}]}


def _run(question, plan, corpus=CORPUS, **settings):
    retriever = MemoryRetriever(corpus=corpus)
    model = RoleModel(planner=[plan], synth=[ANSWER], verifier=[ACCEPT])
    result = run_agent(question, retriever, model,
                       settings=AgentSettings(**{"k_per_query": 4, **settings}))
    return result, retriever


def _filtered_searches(retriever):
    return [sources for _, sources in retriever.queries if sources]


# --- naming -------------------------------------------------------------------

@pytest.mark.parametrize("question,expected", [
    ("How did the expense deadline change between the 2023 and 2024 travel policies?",
     ["travel_policy_2023.txt", "travel_policy_2024.txt"]),
    ("Compare the 2024 and 2023 travel policy meal allowances",
     ["travel_policy_2024.txt", "travel_policy_2023.txt"]),        # order of mention
    ("What changed between the travel policies?", []),              # no version given: not explicit
    ("How did the deadline change between 2023 and 2024?", []),     # versions but no document words
])
def test_named_sources_match_versions_and_names_exactly(question, expected):
    assert named_sources(question, DOCS) == expected


def test_similarly_named_versions_are_not_confused():
    named = named_sources("Compare the 2023 and 2024 travel policies", DOCS)
    assert "travel_policy_2022.txt" not in named
    assert "e_remote_work_addendum_2024.txt" not in named   # shares "2024" but not the name


# --- agent behaviour ------------------------------------------------------------

def test_named_versions_are_searched_and_alphabetically_earlier_documents_are_not():
    result, retriever = _run("How did the expense report deadline change between the 2023 and 2024 travel policies?",
                             _compare_plan())
    assert _filtered_searches(retriever) == [("travel_policy_2023.txt",), ("travel_policy_2024.txt",)]
    assert len(retriever.queries) == 2                         # no discovery search was needed
    sources = {e.source for e in result.evidence}
    assert sources == {"travel_policy_2023.txt", "travel_policy_2024.txt"}
    sel = result.source_selections[0]
    assert sel["method"] == "named" and sel["limitation"] is None
    assert result.retrieval_calls == 2


def test_unnamed_comparison_uses_relevance_not_alphabetical_order():
    result, retriever = _run("How did the expense report deadline change between the two travel policies?",
                             _compare_plan())
    discovery, *filtered = retriever.queries
    assert discovery[1] == ()                                  # one unfiltered discovery search
    searched = [s[0] for _, s in filtered]
    assert {"travel_policy_2023.txt", "travel_policy_2024.txt"} <= set(searched)
    assert searched.index("travel_policy_2023.txt") < 3 and searched.index("travel_policy_2024.txt") < 3
    assert "a_benefits_faq.txt" not in searched
    sel = result.source_selections[0]
    assert sel["method"] == "relevance" and sel["candidates"][:3] == searched[:3]
    assert result.retrieval_calls == 1 + len(filtered)


def test_both_versions_reach_the_evidence_when_the_cap_is_tight():
    # One document has many matching passages; the evidence cap must not fill with it alone.
    corpus = dict(CORPUS)
    corpus["travel_policy_2023.txt"] = [f"The 2023 travel policy expense report deadline rule {i}." for i in range(6)]
    result, _ = _run("Compare the expense report deadline in the 2023 and 2024 travel policies",
                     _compare_plan(), corpus=corpus, max_evidence=3)
    sources = [e.source for e in result.evidence]
    assert len(sources) == 3 and set(sources) == {"travel_policy_2023.txt", "travel_policy_2024.txt"}


def test_planner_named_sources_are_respected():
    plan = _compare_plan(sources=["travel_policy_2022.txt", "travel_policy_2024.txt"])
    result, retriever = _run("Compare the expense deadline across the oldest and newest versions", plan)
    assert _filtered_searches(retriever)[:2] == [("travel_policy_2022.txt",), ("travel_policy_2024.txt",)]
    assert result.source_selections[0]["named"][:2] == ["travel_policy_2022.txt", "travel_policy_2024.txt"]


def test_unknown_planner_sources_cannot_widen_the_selection():
    plan = _compare_plan(sources=["../../secrets.txt"])
    result, retriever = _run("Compare the 2023 and 2024 travel policies on expense deadlines", plan)
    assert all(s[0] in CORPUS for s in _filtered_searches(retriever))
    assert "../../secrets.txt" not in result.source_selections[0]["selected"]


def test_search_count_stays_bounded():
    plan = {"route": "complex", "compare_documents": True,
            "subqueries": [{"query": "expense report deadline"}, {"query": "meal allowance"},
                           {"query": "travel policy"}]}
    result, retriever = _run("What differs between the travel policies?", plan, max_compare_docs=3)
    assert len(result.source_selections[0]["selected"]) <= 3
    assert result.retrieval_calls == len(retriever.queries) <= 1 + 3 * 3


def test_limitation_recorded_when_fewer_than_two_relevant_documents_found():
    corpus = {"z_only_policy.txt": ["The expense report deadline is 15 days."],
              **{f"{c}_filler.txt": ["Unrelated."] for c in "abcdef"}}
    tools = RetrievalTools(_SourceLimitedRetriever(corpus, hits=["z_only_policy.txt"]))
    sel = select_compare_sources("How did the expense report deadline change?", tools, max_docs=5)
    assert sel.selected == ["z_only_policy.txt"]
    assert sel.limitation and "not exhaustive" in sel.limitation


def test_limitation_recorded_when_relevant_candidates_exceed_the_bound():
    tools = RetrievalTools(MemoryRetriever(corpus=CORPUS))
    sel = select_compare_sources("travel policy expense report deadline meal allowance", tools, max_docs=2)
    assert len(sel.selected) == 2 and sel.limitation and "not searched" in sel.limitation


def test_refinement_reuses_the_selection_without_another_discovery_search():
    revise = {"verdicts": [{"claim_number": 1, "status": "unsupported", "evidence_ids": ["1"]}],
              "followup_query": "expense report deadline 2024", "recommendation": "revise"}
    retriever = MemoryRetriever(corpus=CORPUS)
    model = RoleModel(planner=[_compare_plan()], synth=[ANSWER, ANSWER], verifier=[revise, ACCEPT])
    result = run_agent("How did the expense report deadline change between the two travel policies?",
                       retriever, model, settings=AgentSettings(k_per_query=4, max_revisions=1))
    unfiltered = [q for q, s in retriever.queries if not s]
    assert len(unfiltered) == 1                                 # discovery ran once
    assert [s["method"] for s in result.source_selections] == ["relevance", "reused"]


class _SourceLimitedRetriever(MemoryRetriever):
    """Unfiltered searches only return documents from `hits` (simulates low relevance elsewhere)."""

    def __init__(self, corpus, hits):
        super().__init__(corpus=corpus)
        self.hits = set(hits)

    def search(self, query, k=None, sources=None):
        docs = super().search(query, k=k, sources=sources)
        return docs if sources else [d for d in docs if d.metadata["source"] in self.hits]
