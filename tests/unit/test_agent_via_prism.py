"""Agentic and Standard RAG end to end through a mocked Prism gateway (httpx.MockTransport)."""

import json

import httpx
import pytest
from langchain_core.retrievers import BaseRetriever

import retriever as r
from agent import AgentSettings, run_agent
from llm import ProviderRejectedError
from llm.openai_compat import PrismChatModel
from tests.unit.agent_helpers import ROLE_MARKERS, MemoryRetriever
from tests.unit.test_prism_adapter import KEY, prism_body

PLAN = {"route": "complex", "subqueries": [{"query": "2023 meal allowance"}, {"query": "2024 meal allowance"}]}
ANSWER = "In 2023 the meal allowance was 45 dollars [1]. In 2024 it rose to 60 dollars [2]."
VERDICTS = {"verdicts": [{"claim_number": 1, "status": "supported", "evidence_ids": ["1"]},
                         {"claim_number": 2, "status": "supported", "evidence_ids": ["2"]}],
            "recommendation": "accept"}


class FakePrism:
    """Mocked Prism endpoint: replies per agent role; records every request body."""

    def __init__(self, **overrides):
        self.requests = []
        self.replies = {
            "planner": httpx.Response(200, json=prism_body(json.dumps(PLAN))),
            "synth": httpx.Response(200, json=prism_body(ANSWER)),
            "verifier": httpx.Response(200, json=prism_body(json.dumps(VERDICTS))),
            "standard": httpx.Response(200, json=prism_body("The 2024 allowance is 60 dollars [1].")),
        }
        self.replies.update(overrides)

    def __call__(self, request):
        body = json.loads(request.content)
        self.requests.append(body)
        system = body["messages"][0]["content"]
        role = next(role for role, marker in ROLE_MARKERS.items() if marker in system)
        return self.replies[role]

    def model(self):
        return PrismChatModel(api_key=KEY, base_url="http://prism.test", transport=httpx.MockTransport(self))


def _run(prism, question="Compare the meal allowance in 2023 and 2024."):
    return run_agent(question, MemoryRetriever(), prism.model(), settings=AgentSettings(k_per_query=1))


def test_agent_completes_through_prism_and_reports_gateway_metadata():
    prism = FakePrism()
    result = _run(prism)
    assert result.status == "completed" and result.verification.grounding == "verified"
    summary = result.inference_summary()
    assert summary == {"backends": ["prism"], "inference_modes": ["prism_local"],
                       "providers": ["ollama"], "cost_usd": 0.0, "fallbacks": 0}
    assert len(prism.requests) == 3 == result.llm_calls
    assert all("temperature" in b for b in prism.requests)
    assert [b["temperature"] for b in prism.requests] == [0.0, 0.1, 0.0]  # planner, synth, verifier
    assert "Reply with a single json object" in prism.requests[0]["messages"][0]["content"]


def _mock(content="[mock openai response]", provider="openai"):
    body = prism_body(content, provider=provider)
    body["id"] = "chatcmpl-mock"
    return httpx.Response(200, json=body)


def test_mock_verifier_response_never_counts_as_verification():
    result = _run(FakePrism(verifier=httpx.Response(200, json=prism_body(json.dumps(VERDICTS), provider="openai"))))
    v = result.verification
    assert v.llm_status == "failed" and v.grounding == "not_verified"
    assert "ProviderRejectedError" in v.llm_error
    assert result.status == "completed_unverified"
    assert len(result.llm_meta) == 2  # planner + synthesizer only


def test_mock_synthesis_response_fails_the_run_without_leaking_text():
    result = _run(FakePrism(synth=_mock()))
    assert result.status == "failed"
    assert "[mock" not in result.answer and "ProviderRejectedError" in result.error


def test_cached_planner_response_fails_closed():
    cached = prism_body(json.dumps(PLAN), provider="cache", cache_hit=True)
    result = _run(FakePrism(planner=httpx.Response(200, json=cached)))
    assert result.status == "failed" and "cached response" in result.error


def test_guardrail_block_surfaces_reason():
    blocked = httpx.Response(400, json={"detail": "Request blocked by guardrail: PII detected: PERSON"})
    result = _run(FakePrism(synth=blocked))
    assert result.status == "failed" and "PII detected: PERSON" in result.error


def test_gateway_outage_does_not_fall_back_to_direct_ollama():
    prism = FakePrism(planner=httpx.Response(503, json={"detail": "All providers failed"}))
    result = _run(prism)
    assert result.status == "failed" and len(prism.requests) == 1
    assert result.llm_meta == []


class Fixed(BaseRetriever):
    docs: list

    def _get_relevant_documents(self, query, *, run_manager=None):
        return self.docs


def test_standard_rag_through_prism():
    docs = MemoryRetriever().search("2024 meal allowance", k=1)
    chain = r.build_rag_chain(Fixed(docs=docs), chat_model=FakePrism().model())
    out = chain.invoke("What is the 2024 meal allowance?")
    assert out["citations"].ok and out["llm"].metadata["provider_used"] == "ollama"


def test_standard_rag_rejects_mock_provider():
    docs = MemoryRetriever().search("2024 meal allowance", k=1)
    chain = r.build_rag_chain(Fixed(docs=docs), chat_model=FakePrism(standard=_mock()).model())
    with pytest.raises(ProviderRejectedError):
        chain.invoke("What is the 2024 meal allowance?")
