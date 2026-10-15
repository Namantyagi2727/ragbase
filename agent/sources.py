"""
agent/sources.py — Choose which documents a comparison searches.

A comparison searches each selected document separately, so the choice of
documents decides what evidence the answer can use. Selection, in order:

  1. Documents the question (or the planner, with validated names) identifies
     explicitly, e.g. "the 2023 and 2024 travel policies". A document counts as
     named when every version number in its file name appears in the question
     together with at least one of its distinguishing words, or, for names
     without a version number, when all of its distinguishing words appear.
  2. Otherwise, or when fewer than two documents are named, one bounded
     discovery search over all documents ranks sources by relevance; sources
     in the top of that ranking are selected.

The number of documents searched stays bounded (max_docs). When fewer than two
relevant documents are identified, or relevant candidates exceed the bound,
the limitation is recorded instead of treating the search as exhaustive.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from agent.tools import RetrievalTools, ToolError

_GENERIC = {
    "the", "and", "for", "of", "policy", "policies", "document", "doc", "docs", "file", "final",
    "draft", "copy", "edition", "version", "rev", "pdf", "docx", "txt", "md", "csv",
}
_TOKEN_RE = re.compile(r"[a-z]+|\d+")


def _name_tokens(source: str) -> tuple[set[str], set[str]]:
    """(version numbers, distinguishing words) of a file name."""
    stem = source.rsplit(".", 1)[0].lower()
    tokens = _TOKEN_RE.findall(stem)
    numbers = {t for t in tokens if t.isdigit()}
    words = {t for t in tokens if not t.isdigit() and len(t) >= 3 and t not in _GENERIC}
    return numbers, words


def named_sources(question: str, documents: list[str]) -> list[str]:
    """Documents the question identifies explicitly, in the order they are mentioned."""
    q_tokens = _TOKEN_RE.findall(question.lower())
    q_numbers = {t for t in q_tokens if t.isdigit()}
    q_words = [t for t in q_tokens if not t.isdigit()]

    def word_in_question(w: str) -> bool:
        # prefix match either way covers plurals and inflections ("policy"/"policies", "travel"/"travelling")
        return any(qw.startswith(w) or (len(qw) >= 4 and w.startswith(qw)) for qw in q_words)

    named = []
    for doc in documents:
        numbers, words = _name_tokens(doc)
        matched = {w for w in words if word_in_question(w)}
        if numbers:
            ok = numbers <= q_numbers and (matched or not words)
        else:
            ok = bool(words) and matched == words
        if ok:
            named.append(doc)

    def first_mention(doc: str) -> int:
        numbers, words = _name_tokens(doc)
        positions = [i for i, t in enumerate(q_tokens) if t in numbers or any(t.startswith(w) for w in words)]
        return min(positions, default=len(q_tokens))

    return sorted(named, key=first_mention)


@dataclass
class SourceSelection:
    method: str                                   # "named" | "named+relevance" | "relevance"
    selected: list[str]
    named: list[str] = field(default_factory=list)
    candidates: list[str] = field(default_factory=list)   # discovery ranking, most relevant first
    discovery_query: str | None = None
    retrieval_calls: int = 0
    limitation: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def select_compare_sources(
    question: str, tools: RetrievalTools, *, planner_sources: list[str] = (), max_docs: int = 5,
    discovery_k: int = 10, primary_window: int = 5,
) -> SourceSelection:
    """Documents to search separately for a comparison. Uses at most one discovery search."""
    documents = tools.list_documents()
    known = set(documents)
    named = [s for s in planner_sources if s in known]
    named += [s for s in named_sources(question, documents) if s not in named]

    if len(named) >= 2:
        limitation = None
        if len(named) > max_docs:
            limitation = (f"The question names {len(named)} documents; only the first {max_docs} were searched: "
                          f"{', '.join(named[max_docs:])} not searched.")
        return SourceSelection(method="named", selected=named[:max_docs], named=named, limitation=limitation)

    query = " ".join(question.split())[:300]
    try:
        outcome = tools.hybrid_search(query, k=discovery_k)
    except ToolError as e:
        return SourceSelection(method="named" if named else "relevance", selected=named[:max_docs], named=named,
                               limitation=f"Relevance discovery failed ({e}); only named documents were searched.")
    candidates: list[str] = []
    for d in outcome.documents:
        src = d.metadata.get("source")
        if src in known and src not in candidates:
            candidates.append(src)
    # The retriever always returns k results, so the tail of the ranking can hold
    # documents with little relevance. Prefer sources in the top of the ranking and
    # extend into the tail only when that yields fewer than two documents.
    top = []
    for d in outcome.documents[:primary_window]:
        src = d.metadata.get("source")
        if src in known and src not in top:
            top.append(src)

    selected = list(named)
    selected += [s for s in top if s not in selected]
    if len(selected) < 2:
        selected += [s for s in candidates if s not in selected][: 2 - len(selected)]
    method = "named+relevance" if named else "relevance"
    limitation = None
    if len(selected) < 2:
        limitation = ("Fewer than two relevant documents were identified, so the comparison may be incomplete; "
                      "the search is not exhaustive.")
    elif len(selected) > max_docs:
        limitation = (f"{len(selected)} documents appeared in the relevance discovery; only the top {max_docs} "
                      f"were searched: {', '.join(selected[max_docs:])} not searched.")
    return SourceSelection(method=method, selected=selected[:max_docs], named=named, candidates=candidates,
                           discovery_query=query, retrieval_calls=1, limitation=limitation)
