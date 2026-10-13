"""
llm/openai_compat.py — Prism gateway backend (OpenAI-compatible chat completions).

Prism accepts only `model`, `messages`, `temperature` and `stream`; other
fields are dropped. So this backend:

  - always sends `temperature` (Prism defaults to 0.7 otherwise);
  - requests structured output through explicit JSON instructions in the
    prompt (no native schema mode); callers still validate and repair;
  - verifies every response's provider from `prism_metadata` against an
    allowlist and fails closed when it cannot be established — including
    cache hits, whose metadata does not say which provider produced them.

Mock-content and mock-ID detection run as an extra check, not as the
primary provider test. A rejected response raises ProviderRejectedError and
its text is never returned. There is no fallback to direct Ollama.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Optional

import httpx

import config
from llm.base import (
    LLMAuthError,
    LLMGuardrailError,
    LLMRateLimitError,
    LLMResponseError,
    LLMResult,
    LLMTimeoutError,
    LLMUnavailableError,
    Message,
    ProviderRejectedError,
)

_MOCK_CONTENT_RE = re.compile(r"^\s*\[mock [\w\- ]*response\]\s*$", re.I)
_BEARER_RE = re.compile(r"(?i)bearer\s+\S+")
_MAX_DETAIL = 200


def json_instruction(schema: dict[str, Any]) -> str:
    return (
        "Respond with ONLY a JSON object — no prose and no code fences — that conforms to "
        f"this JSON schema:\n{json.dumps(schema, separators=(',', ':'))}"
    )


def with_json_instruction(messages: list[Message], schema: dict[str, Any]) -> list[Message]:
    """Append the schema instruction to the system message (or add one)."""
    out = [dict(m) for m in messages]
    instruction = json_instruction(schema)
    for m in out:
        if m["role"] == "system":
            m["content"] = f"{m['content']}\n\n{instruction}"
            return out
    return [{"role": "system", "content": instruction}, *out]


class PrismChatModel:
    backend = "prism"

    def __init__(
        self,
        api_key: str,
        model: str = config.PRISM_MODEL,
        base_url: str = config.PRISM_BASE_URL,
        temperature: float = config.LLM_TEMPERATURE,
        timeout_s: float = config.PRISM_TIMEOUT_S,
        allowed_providers: frozenset = config.PRISM_ALLOWED_PROVIDERS,
        local_providers: frozenset = config.LOCAL_PROVIDERS,
        transport: Optional[httpx.BaseTransport] = None,
    ):
        if not api_key:
            raise LLMAuthError("PRISM_API_KEY is not set; the Prism backend needs an API key")
        if not allowed_providers:
            raise ValueError("PRISM_ALLOWED_PROVIDERS is empty; no response could be accepted")
        self._api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.allowed_providers = frozenset(allowed_providers)
        self.local_providers = frozenset(local_providers)
        self._transport = transport

    def __repr__(self) -> str:  # never include the key
        return f"PrismChatModel(model={self.model!r}, base_url={self.base_url!r})"

    # -- secrets -----------------------------------------------------------------

    def _redact(self, text: str) -> str:
        text = text.replace(self._api_key, "[redacted]") if self._api_key else text
        return _BEARER_RE.sub("Bearer [redacted]", text)[:_MAX_DETAIL]

    def _detail(self, resp: httpx.Response) -> str:
        try:
            detail = resp.json().get("detail", "")
        except ValueError:
            detail = ""
        return self._redact(detail if isinstance(detail, str) else json.dumps(detail))

    # -- request -----------------------------------------------------------------

    def complete(
        self,
        messages: list[Message],
        *,
        json_schema: dict[str, Any] | None = None,
        temperature: float | None = None,
        timeout_s: float | None = None,
    ) -> LLMResult:
        sent = with_json_instruction(messages, json_schema) if json_schema else list(messages)
        payload = {
            "model": self.model,
            "messages": sent,
            "temperature": self.temperature if temperature is None else temperature,
            "stream": False,
        }
        start = time.perf_counter()
        try:
            with httpx.Client(timeout=timeout_s or self.timeout_s, transport=self._transport) as client:
                resp = client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                )
        except httpx.TimeoutException as e:
            raise LLMTimeoutError(f"Prism did not respond within {timeout_s or self.timeout_s:.0f}s") from e
        except httpx.TransportError as e:
            raise LLMUnavailableError(f"Cannot reach Prism at {self.base_url}") from e
        latency = time.perf_counter() - start

        self._raise_for_status(resp)
        try:
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as e:
            raise LLMResponseError("Malformed response from Prism") from e

        meta = self._verify_provider(data, text)
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        return LLMResult(
            text=text,
            model=self.model,
            backend=self.backend,
            latency_s=latency,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            is_mock=False,
            metadata={
                **meta,
                "structured_output": "prompted" if json_schema else None,
                "upstream_model": data.get("model"),
            },
        )

    def _raise_for_status(self, resp: httpx.Response) -> None:
        code = resp.status_code
        if code < 400:
            return
        detail = self._detail(resp)
        if code == 400 and "guardrail" in detail.lower():
            raise LLMGuardrailError(detail.split(":", 1)[-1].strip() or "unspecified")
        if code in (401, 403):
            raise LLMAuthError(f"Prism rejected the API key (HTTP {code})")
        if code == 429:
            raise LLMRateLimitError("Prism rate limit exceeded for this API key")
        if code == 404:
            raise LLMResponseError(f"Prism has no route configured for model '{self.model}'")
        if code == 503:
            raise LLMUnavailableError("All Prism providers failed for this request")
        if code >= 500:
            raise LLMUnavailableError(f"Prism returned HTTP {code}")
        raise LLMResponseError(f"Prism rejected the request (HTTP {code}){': ' + detail if detail else ''}")

    def _verify_provider(self, data: dict, text: str) -> dict[str, Any]:
        meta = data.get("prism_metadata")
        if not isinstance(meta, dict) or not meta.get("provider_used"):
            raise ProviderRejectedError(
                "Prism response has no provider metadata; the provider cannot be established"
            )
        provider = str(meta["provider_used"])
        if provider == "cache" or meta.get("cache_hit"):
            raise ProviderRejectedError(
                "Prism served a cached response whose original provider is not recorded; "
                "rejected because the provider cannot be established"
            )
        if provider not in self.allowed_providers:
            raise ProviderRejectedError(
                f"Prism response came from provider '{provider}', which is not in "
                f"PRISM_ALLOWED_PROVIDERS ({', '.join(sorted(self.allowed_providers))})"
            )
        # Defence in depth: recognisable mock payloads are rejected even from an allowed provider.
        if _MOCK_CONTENT_RE.match(text or "") or str(data.get("id", "")).startswith("chatcmpl-mock"):
            raise ProviderRejectedError("Prism response looks like mock-provider output")

        out: dict[str, Any] = {
            "gateway": "prism",
            "provider_used": provider,
            "cache_hit": False,
            "inference_mode": "prism_local" if provider in self.local_providers else "prism_remote",
        }
        # Only fields the gateway actually supplied.
        for key in ("fallback_triggered", "cost_usd", "latency_ms"):
            if key in meta:
                out["gateway_" + key if key == "latency_ms" else key] = meta[key]
        return out
