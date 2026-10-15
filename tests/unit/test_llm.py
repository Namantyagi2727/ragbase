"""LLM adapter: Ollama request/response handling (mocked HTTP) and structured output."""

import json

import httpx
import pytest
from pydantic import BaseModel

from llm import LLMResponseError, LLMTimeoutError, LLMUnavailableError, get_chat_model
from llm.fake import FakeChatModel
from llm.ollama import OllamaChatModel
from llm.structured import complete_structured, extract_json

MSGS = [{"role": "user", "content": "hi"}]


def _ollama(handler):
    return OllamaChatModel(model="m", base_url="http://ollama.test", transport=httpx.MockTransport(handler))


def test_ollama_parses_text_and_token_counts():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={
            "message": {"role": "assistant", "content": "hello"},
            "prompt_eval_count": 12, "eval_count": 3,
        })

    res = _ollama(handler).complete(MSGS, temperature=0.0)
    assert (res.text, res.prompt_tokens, res.completion_tokens) == ("hello", 12, 3)
    assert res.backend == "ollama" and not res.is_mock
    assert seen["stream"] is False and seen["options"]["temperature"] == 0.0
    assert "format" not in seen


def test_ollama_passes_json_schema_as_format():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "{}"}})

    _ollama(handler).complete(MSGS, json_schema={"type": "object"})
    assert seen["format"] == {"type": "object"}


def test_ollama_connection_error_is_unavailable():
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(LLMUnavailableError):
        _ollama(handler).complete(MSGS)


def test_ollama_timeout_is_surfaced():
    def handler(request):
        raise httpx.ReadTimeout("slow")

    with pytest.raises(LLMTimeoutError):
        _ollama(handler).complete(MSGS)


def test_ollama_missing_model_gives_actionable_error():
    with pytest.raises(LLMResponseError, match="ollama pull m"):
        _ollama(lambda r: httpx.Response(404, json={"error": "not found"})).complete(MSGS)


def test_ollama_malformed_payload():
    with pytest.raises(LLMResponseError):
        _ollama(lambda r: httpx.Response(200, json={"unexpected": True})).complete(MSGS)


def test_factory_rejects_unknown_backend():
    with pytest.raises(ValueError):
        get_chat_model(backend="nope")


def test_factory_builds_ollama():
    assert get_chat_model(backend="ollama", model="llama3.2").model == "llama3.2"


# --- structured output -------------------------------------------------------

class Plan(BaseModel):
    route: str
    n: int


def test_extract_json_tolerates_fences_and_prose():
    assert json.loads(extract_json('Sure!\n```json\n{"a": {"b": "}"}}\n```')) == {"a": {"b": "}"}}


def test_structured_success_first_try():
    res = complete_structured(FakeChatModel(['{"route": "simple", "n": 1}']), MSGS, Plan)
    assert res.ok and res.value.route == "simple" and res.attempts == 1


def test_structured_repairs_once():
    model = FakeChatModel(["not json", '{"route": "complex", "n": 2}'])
    res = complete_structured(model, MSGS, Plan)
    assert res.ok and res.attempts == 2
    assert "not valid JSON" in model.calls[1]["messages"][-1]["content"]


def test_structured_failure_returns_none_never_a_default():
    res = complete_structured(FakeChatModel(["nope", '{"route": 1}']), MSGS, Plan)
    assert not res.ok and res.value is None and res.error
    assert res.attempts == 2


def test_structured_backend_error_is_not_retried():
    model = FakeChatModel([LLMUnavailableError("down"), '{"route": "x", "n": 1}'])
    res = complete_structured(model, MSGS, Plan)
    assert not res.ok and "LLMUnavailableError" in res.error and res.attempts == 1


def test_ollama_seed_is_sent_only_when_set():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content)["options"])
        return httpx.Response(200, json={"message": {"content": "ok"}})

    _ollama(handler).complete(MSGS)
    OllamaChatModel(model="m", base_url="http://ollama.test", seed=7, transport=httpx.MockTransport(handler)).complete(MSGS)
    assert "seed" not in seen[0] and seen[1]["seed"] == 7


def test_caller_timeout_can_shorten_but_not_extend_the_configured_timeout():
    m = OllamaChatModel(model="m", base_url="http://ollama.test", timeout_s=30)
    assert m._effective_timeout(None) == 30
    assert m._effective_timeout(5) == 5
    assert m._effective_timeout(170) == 30     # the agent's remaining budget must not extend it
