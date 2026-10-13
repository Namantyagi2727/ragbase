"""
llm/base.py — Provider-neutral chat-model interface.

Every LLM call in RAGBase goes through `ChatModel.complete()`, so the
standard chain, the agent graph, tabular Q&A and tests can swap backends
(local Ollama, a scripted fake) without touching call sites.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, TypedDict, runtime_checkable


class Message(TypedDict):
    role: str      # "system" | "user" | "assistant"
    content: str


@dataclass
class LLMResult:
    text: str
    model: str
    backend: str                      # "ollama", "fake", ...
    latency_s: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    is_mock: bool = False             # True for scripted/mock output, never real reasoning
    metadata: dict[str, Any] = field(default_factory=dict)


class LLMError(RuntimeError):
    """Base class for LLM backend failures surfaced to callers."""


class LLMUnavailableError(LLMError):
    """The backend could not be reached (server down, connection refused)."""


class LLMTimeoutError(LLMError):
    """The backend did not respond within the configured timeout."""


class LLMResponseError(LLMError):
    """The backend responded with an error or a malformed payload."""


class LLMAuthError(LLMError):
    """The backend rejected the credentials."""


class LLMRateLimitError(LLMError):
    """The backend refused the request because a rate limit was exceeded."""


class LLMGuardrailError(LLMResponseError):
    """A gateway guardrail blocked the request; nothing was generated."""

    def __init__(self, reason: str):
        super().__init__(f"request blocked by gateway guardrail: {reason}")
        self.reason = reason


class ProviderRejectedError(LLMResponseError):
    """The response cannot be attributed to an allowed, genuine provider.

    Raised for mock providers, providers outside the allowlist, and responses
    whose metadata does not establish the provider (fail closed). The rejected
    text is discarded and never returned to callers.
    """


@runtime_checkable
class ChatModel(Protocol):
    backend: str
    model: str

    def complete(
        self,
        messages: list[Message],
        *,
        json_schema: dict[str, Any] | None = None,
        temperature: float | None = None,
        timeout_s: float | None = None,
    ) -> LLMResult:
        """Return one completion.

        json_schema: when given, the backend should constrain output to JSON
        matching the schema if it supports that natively; callers still
        validate the result (see llm/structured.py).
        """
        ...
