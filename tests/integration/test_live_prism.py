"""Real Prism gateway integration (optional).

Skipped unless PRISM_API_KEY is set and Prism answers /healthz at PRISM_BASE_URL.
A skipped run is NOT a successful integration. See docs/PRISM.md for setup.

These tests record Prism's current real behaviour, including the known
guardrail incompatibility with RAGBase's evidence IDs (documented in
docs/PRISM.md). If Prism's configuration changes, they are expected to change.
"""

import os
import uuid
from pathlib import Path

import httpx
import pytest

import config
from agent.prompts import planner_messages
from agent.schemas import QueryPlan
from citations import build_evidence
from llm import LLMGuardrailError, ProviderRejectedError
from llm.openai_compat import PrismChatModel
from llm.structured import complete_structured
from retriever import build_standard_messages

pytestmark = [pytest.mark.prism, pytest.mark.live, pytest.mark.integration]
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "guardrail_docs"


def _prism_ready() -> bool:
    if not os.environ.get("PRISM_API_KEY"):
        return False
    try:
        return httpx.get(f"{config.PRISM_BASE_URL}/healthz", timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


@pytest.fixture(scope="module")
def prism():
    if not _prism_ready():
        pytest.skip("Prism is not reachable or PRISM_API_KEY is not set")
    return PrismChatModel(api_key=os.environ["PRISM_API_KEY"])


def _unique_question() -> str:
    # A fresh question avoids Prism's exact-match cache, whose hits are rejected by design.
    return f"What is the refund window for order {uuid.uuid4().hex[:8]}?"


def test_genuine_ollama_response_is_accepted_and_attributed(prism):
    res = complete_structured(prism, planner_messages(_unique_question(), ["policy.txt"]), QueryPlan,
                              temperature=0.0)
    assert res.ok, res.error
    meta = res.llm_results[-1].metadata
    assert meta["provider_used"] == "ollama" and meta["inference_mode"] == "prism_local"
    assert res.llm_results[-1].is_mock is False


def test_repeated_identical_request_is_rejected_as_unattributed_cache_hit(prism):
    messages = [{"role": "user", "content": _unique_question()}]
    prism.complete(messages, temperature=0.0)
    with pytest.raises(ProviderRejectedError, match="cached response"):
        prism.complete(messages, temperature=0.0)


def test_evidence_prompt_is_blocked_by_current_guardrails(prism):
    """Known incompatibility: evidence IDs like 'E1' match Presidio's US_DRIVER_LICENSE pattern."""
    text = (FIXTURES / "plain_policy.txt").read_text()
    from langchain_core.documents import Document

    evidence = build_evidence([Document(page_content=text, metadata={
        "chunk_id": "p", "source": "plain_policy.txt", "loc": "txt"})])
    with pytest.raises(LLMGuardrailError) as exc:
        prism.complete(build_standard_messages(_unique_question(), evidence), temperature=0.0)
    assert "US_DRIVER_LICENSE" in exc.value.reason
