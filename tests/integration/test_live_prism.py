"""Real Prism gateway integration (optional): RAGBase → Prism → Ollama over HTTP.

Skipped unless PRISM_API_KEY is set and Prism answers /healthz at PRISM_BASE_URL.
A skipped run is NOT a verified integration. Setup: docs/PRISM.md.

Prism caches full responses for an hour and does not record which provider
produced a cached response, so RAGBase rejects cache hits. Tests that send
fixed prompts therefore skip — rather than pass — if Prism serves them from
cache on a re-run within that window.
"""

import os
import random
from pathlib import Path

import httpx
import pytest
from langchain_core.documents import Document

import config
import ingest
import retriever as r
from agent import run_agent
from agent.prompts import planner_messages
from agent.schemas import QueryPlan
from citations import build_evidence
from llm import LLMGuardrailError, ProviderRejectedError
from llm.openai_compat import PrismChatModel
from llm.structured import complete_structured
from retriever import build_standard_messages

pytestmark = [pytest.mark.prism, pytest.mark.live, pytest.mark.integration]
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


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


@pytest.fixture(scope="module")
def travel_retriever(tmp_path_factory, prism):
    idx = str(tmp_path_factory.mktemp("prism_idx"))
    ingest.ingest_documents(ingest_dir=str(FIXTURES / "agent_docs"), index_dir=idx)
    return r.get_retriever(index_dir=idx)


def _skip_if_cached(error: str | None):
    if error and "cached response" in error:
        pytest.skip("Prism served a cached response without provenance (expected on re-runs within the cache TTL)")


_WORDS = ["table", "river", "window", "paper", "stone", "cloud", "garden", "glass",
          "bridge", "lantern", "harbour", "meadow", "pencil", "ladder", "orchard", "candle"]


def _unique(prefix: str) -> str:
    """Make a first-time request (needed for attribution and cache tests).

    Plain words only: random letter-digit tokens (e.g. hex) can match
    Presidio's US_DRIVER_LICENSE pattern and get the request blocked.
    """
    rng = random.SystemRandom()
    return f"{prefix} (reference {' '.join(rng.choice(_WORDS) for _ in range(5))})"


def test_genuine_ollama_response_is_accepted_and_attributed(prism):
    res = complete_structured(
        prism, planner_messages(_unique("What is the refund window?"), ["policy.txt"]), QueryPlan, temperature=0.0
    )
    assert res.ok, res.error
    llm = res.llm_results[-1]
    assert llm.backend == "prism" and llm.is_mock is False
    assert llm.metadata["provider_used"] == "ollama" and llm.metadata["inference_mode"] == "prism_local"
    assert llm.metadata["upstream_model"] == "llama3.2:3b"
    assert llm.metadata["cost_usd"] == 0.0 and llm.metadata["fallback_triggered"] is False
    assert llm.prompt_tokens and llm.completion_tokens and "gateway_latency_ms" in llm.metadata


def test_standard_rag_answers_with_citations_through_prism(prism, travel_retriever):
    chain = r.build_rag_chain(travel_retriever, chat_model=prism)
    try:
        out = chain.invoke("What is the daily meal allowance in the 2024 travel policy?")
    except ProviderRejectedError as e:
        _skip_if_cached(str(e))
        raise
    assert out["citations"].ok and out["citations"].valid_ids, out["answer"]
    assert "60" in out["answer"]
    assert out["llm"].metadata["provider_used"] == "ollama"


def test_agentic_multi_document_question_through_prism(prism, travel_retriever):
    result = run_agent("Compare the meal allowance in the 2023 and 2024 travel policies.", travel_retriever, prism)
    _skip_if_cached(result.error)
    assert result.status != "failed", result.error
    assert result.route == "complex"
    assert {"travel_policy_2023.txt", "travel_policy_2024.txt"} <= {e.source for e in result.evidence}
    assert result.citations.ok and "45" in result.answer and "60" in result.answer
    summary = result.inference_summary()
    assert summary["backends"] == ["prism"] and summary["providers"] == ["ollama"]
    assert summary["inference_modes"] == ["prism_local"]


def test_cached_response_without_provenance_is_rejected(prism):
    messages = [{"role": "user", "content": _unique("Name one benefit of keeping receipts.")}]
    first = prism.complete(messages, temperature=0.0)
    assert first.metadata["provider_used"] == "ollama"
    with pytest.raises(ProviderRejectedError, match="cached response"):
        prism.complete(messages, temperature=0.0)


def test_mock_provider_response_is_rejected(prism):
    # Prism's seeded 'mock-fast' route is served by its mock OpenAI provider.
    mock_route = PrismChatModel(api_key=os.environ["PRISM_API_KEY"], model="mock-fast")
    with pytest.raises(ProviderRejectedError) as exc:
        mock_route.complete([{"role": "user", "content": _unique("Hello")}], temperature=0.0)
    assert "not in PRISM_ALLOWED_PROVIDERS" in str(exc.value) or "cached response" in str(exc.value)


def test_guardrail_blocks_synthetic_pii(prism):
    text = (FIXTURES / "guardrail_docs" / "named_approvers.txt").read_text()
    evidence = build_evidence([Document(page_content=text, metadata={
        "chunk_id": "n", "source": "named_approvers.txt", "loc": "txt"})])
    with pytest.raises(LLMGuardrailError) as exc:
        prism.complete(build_standard_messages("Who approves large purchases?", evidence), temperature=0.0)
    assert "PERSON" in exc.value.reason


def test_evidence_prompt_without_pii_passes_guardrails(prism):
    text = (FIXTURES / "guardrail_docs" / "plain_policy.txt").read_text()
    evidence = build_evidence([Document(page_content=text, metadata={
        "chunk_id": "p", "source": "plain_policy.txt", "loc": "txt"})])
    res = prism.complete(build_standard_messages(_unique("What is the daily meal allowance?"), evidence),
                         temperature=0.0)
    assert res.metadata["provider_used"] == "ollama"
