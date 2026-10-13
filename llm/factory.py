"""llm/factory.py — Build the configured chat model."""

from __future__ import annotations

import config
from llm.base import ChatModel


def get_chat_model(
    backend: str | None = None,
    model: str | None = None,
) -> ChatModel:
    backend = (backend or config.LLM_BACKEND).lower()
    if backend == "ollama":
        from llm.ollama import OllamaChatModel

        return OllamaChatModel(model=model or config.OLLAMA_MODEL)
    raise ValueError(
        f"Unknown LLM backend '{backend}'. Supported: 'ollama'. "
        "(The 'fake' backend is constructed directly in tests.)"
    )
