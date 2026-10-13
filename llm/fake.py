"""
llm/fake.py — Scripted chat model for deterministic tests.

Output is marked `is_mock=True`: it exercises control flow, never stands in
for real model reasoning in evaluation results.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from llm.base import LLMResult, Message

Reply = str | Exception | Callable[[list[Message]], str]


class FakeChatModel:
    backend = "fake"

    def __init__(self, replies: list[Reply] | Reply, model: str = "fake-model"):
        self.model = model
        self._replies = list(replies) if isinstance(replies, list) else [replies]
        self._repeat_last = not isinstance(replies, list)
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        messages: list[Message],
        *,
        json_schema: dict[str, Any] | None = None,
        temperature: float | None = None,
        timeout_s: float | None = None,
    ) -> LLMResult:
        self.calls.append({"messages": list(messages), "json_schema": json_schema})
        if not self._replies:
            raise AssertionError("FakeChatModel ran out of scripted replies")
        reply = self._replies[0] if self._repeat_last else self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        text = reply(messages) if callable(reply) else reply
        return LLMResult(text=text, model=self.model, backend=self.backend, is_mock=True)
