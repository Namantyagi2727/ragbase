"""Prism backend: request shape, provider verification, error mapping, secret redaction.

All tests use mocked HTTP (httpx.MockTransport); no Prism server is involved.
"""

import json
import logging

import httpx
import pytest
from pydantic import BaseModel

from llm import (
    LLMAuthError,
    LLMGuardrailError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
    LLMUnavailableError,
    ProviderRejectedError,
    get_chat_model,
)
from llm.openai_compat import PrismChatModel, with_json_instruction
from llm.structured import complete_structured

KEY = "pk-test-SECRET-123456"
MSGS = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]


def prism_body(content="hello", provider="ollama", **meta):
    return {
        "id": "chatcmpl-42",
        "object": "chat.completion",
        "model": "llama3.2:3b",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
        "prism_metadata": {"provider_used": provider, "cache_hit": False, "fallback_triggered": False,
                           "cost_usd": 0.0, "latency_ms": 40, **meta},
    }


def model_with(handler, **kw):
    return PrismChatModel(api_key=KEY, base_url="http://prism.test", transport=httpx.MockTransport(handler), **kw)


def reply(body, status=200):
    return lambda request: httpx.Response(status, json=body)


# --- successful requests ---------------------------------------------------------

def test_request_shape_auth_and_explicit_temperature():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=prism_body())

    model_with(handler).complete(MSGS)
    assert seen["url"] == "http://prism.test/v1/chat/completions"
    assert seen["auth"] == f"Bearer {KEY}"
    assert seen["body"] == {"model": "fast", "messages": MSGS, "temperature": 0.1, "stream": False}


def test_temperature_override_is_sent():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=prism_body())

    model_with(handler).complete(MSGS, temperature=0.0)
    assert seen["temperature"] == 0.0


def test_result_carries_only_gateway_supplied_metadata():
    res = model_with(reply(prism_body())).complete(MSGS)
    assert (res.text, res.backend, res.is_mock) == ("hello", "prism", False)
    assert (res.prompt_tokens, res.completion_tokens) == (12, 3)
    assert res.metadata["provider_used"] == "ollama"
    assert res.metadata["inference_mode"] == "prism_local"
    assert res.metadata["cost_usd"] == 0.0 and res.metadata["gateway_latency_ms"] == 40
    assert res.metadata["fallback_triggered"] is False


def test_missing_optional_fields_are_not_fabricated():
    body = prism_body()
    body.pop("usage")
    body["prism_metadata"] = {"provider_used": "ollama"}
    res = model_with(reply(body)).complete(MSGS)
    assert res.prompt_tokens is None and res.completion_tokens is None
    assert "cost_usd" not in res.metadata and "fallback_triggered" not in res.metadata


def test_fallback_to_an_allowed_provider_is_accepted_and_recorded():
    res = model_with(reply(prism_body(fallback_triggered=True))).complete(MSGS)
    assert res.metadata["fallback_triggered"] is True


def test_non_local_allowed_provider_is_labelled_remote():
    res = model_with(reply(prism_body(provider="vllm-remote")),
                     allowed_providers=frozenset({"vllm-remote"})).complete(MSGS)
    assert res.metadata["inference_mode"] == "prism_remote"


# --- structured output -----------------------------------------------------------

class Plan(BaseModel):
    route: str


def test_json_schema_becomes_explicit_prompt_instruction():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=prism_body('{"route": "simple"}'))

    res = model_with(handler).complete(MSGS, json_schema=Plan.model_json_schema())
    system = seen["messages"][0]["content"]
    assert system.startswith("sys\n\nRespond with ONLY a JSON object")
    assert '"route"' in system and "format" not in seen and "response_format" not in seen
    assert res.metadata["structured_output"] == "prompted"


def test_instruction_added_when_no_system_message():
    out = with_json_instruction([{"role": "user", "content": "q"}], {"type": "object"})
    assert out[0]["role"] == "system" and out[1] == {"role": "user", "content": "q"}


def test_structured_parsing_and_bounded_repair_through_prism():
    replies = iter([prism_body("not json"), prism_body('```json\n{"route": "complex"}\n```')])
    res = complete_structured(model_with(lambda r: httpx.Response(200, json=next(replies))), MSGS, Plan)
    assert res.ok and res.value.route == "complex" and res.attempts == 2


# --- provider verification (fail closed) ------------------------------------------

@pytest.mark.parametrize("provider", ["openai", "anthropic", "blocked", "something-new"])
def test_provider_outside_allowlist_is_rejected(provider):
    with pytest.raises(ProviderRejectedError, match="not in PRISM_ALLOWED_PROVIDERS"):
        model_with(reply(prism_body("plausible answer", provider=provider))).complete(MSGS)


def test_cache_hits_are_rejected_because_origin_is_unknown():
    body = prism_body("cached text", provider="cache", cache_hit=True)
    with pytest.raises(ProviderRejectedError, match="cached response"):
        model_with(reply(body)).complete(MSGS)


def test_cached_mock_response_is_rejected():
    body = prism_body("[mock openai response]", provider="cache", cache_hit=True)
    body["id"] = "chatcmpl-mock"
    with pytest.raises(ProviderRejectedError):
        model_with(reply(body)).complete(MSGS)


@pytest.mark.parametrize("meta", [None, {}, {"provider_used": ""}, {"cache_hit": False}])
def test_missing_provider_metadata_fails_closed(meta):
    body = prism_body()
    if meta is None:
        body.pop("prism_metadata")
    else:
        body["prism_metadata"] = meta
    with pytest.raises(ProviderRejectedError, match="no provider metadata"):
        model_with(reply(body)).complete(MSGS)


@pytest.mark.parametrize("content,rid", [("[mock openai response]", "chatcmpl-42"), ("ok", "chatcmpl-mock")])
def test_mock_payload_rejected_even_from_allowed_provider(content, rid):
    body = prism_body(content)
    body["id"] = rid
    with pytest.raises(ProviderRejectedError, match="mock"):
        model_with(reply(body)).complete(MSGS)


def test_rejected_text_never_reaches_structured_parsing():
    body = prism_body('{"route": "simple"}', provider="openai")
    res = complete_structured(model_with(reply(body)), MSGS, Plan)
    assert not res.ok and res.backend_error and res.value is None and res.raw_outputs == []


# --- error mapping -------------------------------------------------------------------

def test_guardrail_block_is_reported_with_reason():
    body = {"detail": "Request blocked by guardrail: PII detected: EMAIL_ADDRESS, PERSON"}
    with pytest.raises(LLMGuardrailError) as exc:
        model_with(reply(body, 400)).complete(MSGS)
    assert exc.value.reason == "PII detected: EMAIL_ADDRESS, PERSON"


@pytest.mark.parametrize("status,error", [
    (401, LLMAuthError), (403, LLMAuthError), (429, LLMRateLimitError),
    (404, LLMResponseError), (503, LLMUnavailableError), (500, LLMUnavailableError),
    (422, LLMResponseError),
])
def test_http_errors_map_to_typed_errors(status, error):
    with pytest.raises(error):
        model_with(reply({"detail": "x"}, status)).complete(MSGS)


def test_timeout_and_connection_errors():
    def slow(request):
        raise httpx.ReadTimeout("slow")

    def down(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(LLMTimeoutError):
        model_with(slow).complete(MSGS)
    with pytest.raises(LLMUnavailableError, match="Cannot reach Prism"):
        model_with(down).complete(MSGS)


def test_malformed_payload():
    with pytest.raises(LLMResponseError, match="Malformed"):
        model_with(reply({"choices": []})).complete(MSGS)


# --- secrets ---------------------------------------------------------------------------

def test_api_key_never_appears_in_errors_repr_or_logs(caplog):
    echo = {"detail": f"upstream said Authorization: Bearer {KEY} and key={KEY}"}
    caplog.set_level(logging.DEBUG)
    errors = []
    for status in (400, 422, 503):
        with pytest.raises(Exception) as exc:
            model_with(reply(echo, status)).complete(MSGS)
        errors.append(str(exc.value))
    model = model_with(reply(prism_body()))
    text = " ".join(errors) + repr(model) + caplog.text
    assert KEY not in text


def test_missing_key_refuses_to_build(monkeypatch):
    monkeypatch.delenv("PRISM_API_KEY", raising=False)
    with pytest.raises(LLMAuthError, match="PRISM_API_KEY is not set"):
        get_chat_model(backend="prism")


def test_factory_builds_prism_from_environment_without_ollama_fallback(monkeypatch):
    monkeypatch.setenv("PRISM_API_KEY", KEY)
    m = get_chat_model(backend="prism", model="llama3.2")
    assert m.backend == "prism" and m.model == "fast"  # Ollama name is not sent to Prism
