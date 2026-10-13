"""
agent/schemas.py — Structured outputs the agent's model calls must satisfy.

Every model decision in the graph is parsed into one of these pydantic
models; anything that fails validation takes an explicit fallback path.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class SubQuery(BaseModel):
    model_config = ConfigDict(extra="ignore")
    query: str = Field(min_length=1, max_length=300)
    sources: list[str] = Field(default_factory=list, max_length=5)


class QueryPlan(BaseModel):
    """Planner decision: how to answer the question."""

    model_config = ConfigDict(extra="ignore")
    route: Literal["simple", "complex"]
    rationale: str = Field(default="", max_length=300)
    subqueries: list[SubQuery] = Field(default_factory=list, max_length=3)
    compare_documents: bool = False


class ClaimVerdict(BaseModel):
    model_config = ConfigDict(extra="ignore")
    claim: str = Field(max_length=500)
    status: Literal["supported", "partially_supported", "unsupported"]
    evidence_ids: list[str] = Field(default_factory=list, max_length=10)
    note: str = Field(default="", max_length=300)


class VerifierOutput(BaseModel):
    """Verifier decision: which claims the evidence supports, and whether to retry."""

    model_config = ConfigDict(extra="ignore")
    verdicts: list[ClaimVerdict] = Field(default_factory=list, max_length=12)
    missing_information: Optional[str] = Field(default=None, max_length=300)
    followup_query: Optional[str] = Field(default=None, max_length=300)
    recommendation: Literal["accept", "revise", "insufficient_evidence"]
