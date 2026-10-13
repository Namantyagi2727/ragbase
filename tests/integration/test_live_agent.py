"""Agentic RAG end to end with real embeddings, reranker and a local Ollama model.

Marked `live`: skipped unless Ollama is reachable. Asserts structure and
safety properties, not exact wording.
"""

from pathlib import Path

import pytest

import ingest
import retriever as r
from agent import run_agent
from llm import get_chat_model
from tests.integration.test_live_ollama import _ollama_up

pytestmark = [pytest.mark.live, pytest.mark.integration]
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "agent_docs"


@pytest.fixture(scope="module")
def retriever(tmp_path_factory):
    if not _ollama_up():
        pytest.skip("Ollama is not running")
    idx = str(tmp_path_factory.mktemp("agent_idx"))
    ingest.ingest_documents(ingest_dir=str(FIXTURES), index_dir=idx)
    return r.get_retriever(index_dir=idx)


def test_comparison_uses_both_policies_and_terminates(retriever):
    result = run_agent(
        "Compare the meal allowance in the 2023 and 2024 travel policies.", retriever, get_chat_model()
    )
    assert result.route == "complex"
    assert result.status in ("completed", "completed_unverified", "qualified")
    assert {"travel_policy_2023.txt", "travel_policy_2024.txt"} <= {e.source for e in result.evidence}
    assert not result.citations.invalid_ids and not result.citations.non_evidence_citations
    assert "45" in result.answer and "60" in result.answer
    assert result.revision_count <= 1


def test_simple_question_takes_the_single_search_path(retriever):
    result = run_agent("What is the daily meal allowance in the 2024 travel policy?", retriever, get_chat_model())
    assert result.route == "simple" and result.retrieval_calls == 1
    assert result.verification.llm_status == "skipped"
    assert "60" in result.answer


def test_unanswerable_question_abstains(retriever):
    result = run_agent("What is the company's policy on bringing pets to the office?", retriever, get_chat_model())
    assert result.citations.abstained
