"""
agent/tools.py — The only capabilities the agent has: read-only retrieval.

Arguments are validated before use. Unknown source names are dropped (and
reported) rather than passed through, so model output — including anything
a document might have induced it to write — cannot widen what is searched.
There is deliberately no filesystem, shell or network tool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from langchain_core.documents import Document
from pydantic import BaseModel, Field, ValidationError


class SearchableRetriever(Protocol):
    def search(self, query: str, k: int | None = None, sources=None) -> list[Document]: ...
    def sources(self) -> list[str]: ...


class ToolError(ValueError):
    """Tool arguments were invalid; nothing was executed."""


class HybridSearchArgs(BaseModel):
    query: str = Field(min_length=1, max_length=300)
    k: int = Field(default=4, ge=1, le=10)
    sources: list[str] = Field(default_factory=list, max_length=10)


@dataclass
class SearchOutcome:
    documents: list[Document]
    query: str
    sources: list[str]
    dropped_sources: list[str] = field(default_factory=list)


class RetrievalTools:
    """Retrieval tools exposed to the agent, backed by the existing hybrid retriever."""

    NAMES = ("hybrid_search", "list_documents")

    def __init__(self, retriever: SearchableRetriever):
        self._retriever = retriever
        self._known = set(retriever.sources())

    def list_documents(self) -> list[str]:
        return sorted(self._known)

    def hybrid_search(self, query: str, k: int = 4, sources: list[str] | None = None) -> SearchOutcome:
        try:
            args = HybridSearchArgs(query=query.strip(), k=k, sources=list(sources or []))
        except ValidationError as e:
            raise ToolError(f"invalid hybrid_search arguments: {e.errors()[0]['msg']}") from e
        valid = [s for s in args.sources if s in self._known]
        dropped = [s for s in args.sources if s not in self._known]
        # If every requested source was unknown, search all documents rather than none.
        docs = self._retriever.search(args.query, k=args.k, sources=valid or None)
        return SearchOutcome(documents=docs, query=args.query, sources=valid, dropped_sources=dropped)
