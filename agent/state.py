"""
agent/state.py — Typed graph state and the result returned to callers.

The state is plain data so nodes stay testable without LangGraph, and the
agent stays independent of the UI.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, Optional, TypedDict

from citations import CitationReport, Evidence

Route = Literal["simple", "complex"]
Status = Literal[
    "running",
    "completed",            # answer produced and accepted by verification
    "completed_unverified", # answer produced; verification incomplete or not run
    "qualified",            # answer produced with unresolved verification problems
    "failed",               # backend error; no reliable answer
    "timeout",              # time budget exhausted
]
Grounding = Literal["verified", "partially_verified", "unsupported", "not_verified", "skipped"]


class TraceEvent(TypedDict):
    step: int
    node: str
    summary: str            # user-facing description of what happened
    data: dict[str, Any]    # small structured details (decisions, counts)
    elapsed_s: float


@dataclass
class Verification:
    """Outcome of the verification step, kept separate from citation validity."""

    citation_ok: bool                       # deterministic: IDs exist, no invented references
    grounding: Grounding                    # factual support of claims by cited evidence
    llm_status: Literal["completed", "failed", "skipped"]
    decision: Literal["accept", "revise", "finalize"]
    unsupported_claims: list[str] = field(default_factory=list)
    numeric_mismatches: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    followup_query: Optional[str] = None
    llm_error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "citation_ok": self.citation_ok,
            "grounding": self.grounding,
            "llm_status": self.llm_status,
            "decision": self.decision,
            "unsupported_claims": self.unsupported_claims,
            "numeric_mismatches": self.numeric_mismatches,
            "issues": self.issues,
            "followup_query": self.followup_query,
            "llm_error": self.llm_error,
        }


class AgentState(TypedDict, total=False):
    question: str
    started_at: float
    deadline: float

    # planning
    route: Route
    plan: dict[str, Any]
    planner_fallback: Optional[str]         # why the planner's output was not used

    # retrieval
    evidence: list[Evidence]                # numbered E1..En; numbering is stable across rounds
    pending_queries: list[dict[str, Any]]   # [{query, sources}] for the next retrieval round

    # synthesis / verification
    answer: str
    citations: CitationReport
    verification: Verification
    revision_count: int
    revision_feedback: Optional[str]

    # termination
    status: Status
    error: Optional[str]

    # accounting (reducers: values from each node are added)
    trace: Annotated[list[TraceEvent], operator.add]
    llm_calls: Annotated[int, operator.add]
    retrieval_calls: Annotated[int, operator.add]
    prompt_tokens: Annotated[int, operator.add]
    completion_tokens: Annotated[int, operator.add]


@dataclass
class AgentResult:
    question: str
    answer: str
    route: Route
    status: Status
    evidence: list[Evidence]
    citations: Optional[CitationReport]
    verification: Optional[Verification]
    plan: dict[str, Any]
    planner_fallback: Optional[str]
    revision_count: int
    trace: list[TraceEvent]
    llm_calls: int
    retrieval_calls: int
    prompt_tokens: int
    completion_tokens: int
    elapsed_s: float
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "route": self.route,
            "status": self.status,
            "plan": self.plan,
            "planner_fallback": self.planner_fallback,
            "revision_count": self.revision_count,
            "evidence": [
                {"eid": e.eid, "chunk_id": e.chunk_id, "source": e.source, "loc": e.loc}
                for e in self.evidence
            ],
            "citations": self.citations.to_dict() if self.citations else None,
            "verification": self.verification.to_dict() if self.verification else None,
            "trace": self.trace,
            "llm_calls": self.llm_calls,
            "retrieval_calls": self.retrieval_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "elapsed_s": round(self.elapsed_s, 3),
            "error": self.error,
        }
