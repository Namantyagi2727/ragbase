"""llm/factory.py — Build the configured chat model."""

from __future__ import annotations

import os

import config
from llm.base import ChatModel


def get_chat_model(
    backend: str | None = None,
    model: str | None = None,
    prism_model: str | None = None,
) -> ChatModel:
    """Build the configured backend.

    model: Ollama model name, used by the direct backend.
    prism_model: Prism virtual model name, used by the Prism backend. Kept
        separate so callers passing an Ollama name never send it to Prism.
    """
    backend = (backend or config.LLM_BACKEND).lower()
    if backend == "ollama":
        from llm.ollama import OllamaChatModel

        return OllamaChatModel(model=model or config.OLLAMA_MODEL)
    if backend == "prism":
        # No fallback: if Prism is misconfigured or unavailable, calls fail visibly.
        from llm.openai_compat import PrismChatModel

        return PrismChatModel(
            api_key=os.environ.get("PRISM_API_KEY", ""), model=prism_model or config.PRISM_MODEL
        )
    raise ValueError(
        f"Unknown LLM backend '{backend}'. Supported: 'ollama', 'prism'. "
        "(The 'fake' backend is constructed directly in tests.)"
    )
