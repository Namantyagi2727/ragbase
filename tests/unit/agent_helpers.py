"""Test doubles for the agent: a role-scripted model and an in-memory retriever."""

from __future__ import annotations

import json

from langchain_core.documents import Document

from llm.base import LLMResult

ROLE_MARKERS = {
    "planner": "You plan how to answer",
    "synth": "You answer questions using ONLY",
    "verifier": "You check whether an answer",
    "abscheck": "An assistant said it could not answer",
    "standard": "You are a document Q&A assistant",
}


class RoleModel:
    """Replies from a per-role script; records every call. Output is marked as mock."""

    backend = "fake"
    model = "role-fake"

    def __init__(self, **scripts):
        self.scripts = {role: list(replies) for role, replies in scripts.items()}
        self.calls: list[str] = []

    def _role(self, messages) -> str:
        system = messages[0]["content"]
        for role, marker in ROLE_MARKERS.items():
            if marker in system:
                return role
        raise AssertionError(f"unrecognised prompt: {system[:60]}")

    def complete(self, messages, *, json_schema=None, temperature=None, timeout_s=None):
        role = self._role(messages)
        self.calls.append(role)
        queue = self.scripts.get(role)
        if not queue:
            raise AssertionError(f"no scripted reply left for role {role!r}")
        reply = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, dict):
            reply = json.dumps(reply)
        text = reply(messages) if callable(reply) else reply
        return LLMResult(text=text, model=self.model, backend=self.backend, is_mock=True,
                         prompt_tokens=10, completion_tokens=5)


CORPUS = {
    "policy_2023.txt": [
        "The 2023 travel policy sets the daily meal allowance at 45 dollars.",
        "Under the 2023 policy, hotel stays above 200 dollars need manager approval.",
    ],
    "policy_2024.txt": [
        "The 2024 travel policy raises the daily meal allowance to 60 dollars.",
        "Under the 2024 policy, hotel stays above 250 dollars need director approval.",
    ],
    "handbook.txt": ["Employees accrue 1.5 days of annual leave per month."],
}


class MemoryRetriever:
    """Keyword-overlap retriever over CORPUS implementing search() and sources()."""

    def __init__(self, corpus=CORPUS):
        self.docs = []
        for source, texts in corpus.items():
            for i, text in enumerate(texts):
                self.docs.append(Document(page_content=text, metadata={
                    "chunk_id": f"{source}#{i}", "source": source, "loc": "txt"}))
        self.queries: list[tuple[str, tuple]] = []

    def sources(self):
        return sorted({d.metadata["source"] for d in self.docs})

    def search(self, query, k=None, sources=None):
        self.queries.append((query, tuple(sources or ())))
        words = set(query.lower().split())
        pool = [d for d in self.docs if not sources or d.metadata["source"] in sources]
        ranked = sorted(pool, key=lambda d: -len(words & set(d.page_content.lower().split())))
        return ranked[: k or 5]
