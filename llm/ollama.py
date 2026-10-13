"""llm/ollama.py — Direct local Ollama backend (fully offline)."""

from __future__ import annotations

import time
from typing import Any

import httpx

import config
from llm.base import (
    LLMResponseError,
    LLMResult,
    LLMTimeoutError,
    LLMUnavailableError,
    Message,
)


class OllamaChatModel:
    backend = "ollama"

    def __init__(
        self,
        model: str = config.OLLAMA_MODEL,
        base_url: str = config.OLLAMA_BASE_URL,
        temperature: float = config.LLM_TEMPERATURE,
        num_ctx: int = config.LLM_NUM_CTX,
        timeout_s: float = config.LLM_TIMEOUT_S,
        transport: httpx.BaseTransport | None = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.timeout_s = timeout_s
        self._transport = transport

    def complete(
        self,
        messages: list[Message],
        *,
        json_schema: dict[str, Any] | None = None,
        temperature: float | None = None,
        timeout_s: float | None = None,
    ) -> LLMResult:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "stream": False,
            "options": {
                "temperature": self.temperature if temperature is None else temperature,
                "num_ctx": self.num_ctx,
            },
        }
        if json_schema is not None:
            # Ollama constrains decoding to the schema (structured outputs).
            payload["format"] = json_schema

        start = time.perf_counter()
        try:
            with httpx.Client(
                timeout=timeout_s or self.timeout_s, transport=self._transport
            ) as client:
                resp = client.post(f"{self.base_url}/api/chat", json=payload)
        except httpx.TimeoutException as e:
            raise LLMTimeoutError(
                f"Ollama did not respond within {timeout_s or self.timeout_s:.0f}s"
            ) from e
        except httpx.TransportError as e:
            raise LLMUnavailableError(
                f"Cannot reach Ollama at {self.base_url} — is `ollama serve` running?"
            ) from e
        latency = time.perf_counter() - start

        if resp.status_code == 404:
            raise LLMResponseError(
                f"Ollama model '{self.model}' not found — run `ollama pull {self.model}`"
            )
        if resp.status_code >= 400:
            raise LLMResponseError(f"Ollama returned HTTP {resp.status_code}")

        try:
            data = resp.json()
            text = data["message"]["content"]
        except (ValueError, KeyError, TypeError) as e:
            raise LLMResponseError("Malformed response from Ollama") from e

        return LLMResult(
            text=text,
            model=self.model,
            backend=self.backend,
            latency_s=latency,
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
            metadata={
                "inference_mode": "local_direct",
                "structured_output": "native" if json_schema is not None else None,
            },
        )
