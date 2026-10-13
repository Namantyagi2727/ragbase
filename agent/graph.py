"""
agent/graph.py — LangGraph wiring for Agentic RAG.

    plan ──simple──▶ standard ─────────────────────────────▶ finalize
      └──complex──▶ retrieve ▶ synthesize ▶ verify ──accept──▶ finalize
                       ▲            ▲          │
                       │            │       revise (≤ max_revisions, time left)
                       └── refine ──┘◀─────────┘

Any node that sets status "failed" or "timeout" routes straight to finalize.
The loop is bounded three ways: max_revisions, the time budget, and
LangGraph's recursion limit.
"""

from __future__ import annotations

import time
from typing import Optional

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

from agent.nodes import AgentDeps, AgentNodes, AgentSettings
from agent.state import AgentResult, AgentState
from agent.tools import RetrievalTools, SearchableRetriever
from llm import ChatModel

_STOPPED = ("failed", "timeout")


def _stopped(state: AgentState) -> bool:
    return state.get("status") in _STOPPED


def build_graph(deps: AgentDeps):
    nodes = AgentNodes(deps)
    s = deps.settings

    def after_plan(state: AgentState) -> str:
        if _stopped(state):
            return "finalize"
        return "standard" if state.get("route") == "simple" else "retrieve"

    def after_step(next_node: str):
        def route(state: AgentState) -> str:
            return "finalize" if _stopped(state) else next_node
        return route

    def after_verify(state: AgentState) -> str:
        v = state.get("verification")
        if (
            not _stopped(state)
            and v is not None
            and v.decision == "revise"
            and state.get("revision_count", 0) < s.max_revisions
            and state["deadline"] - time.monotonic() > s.min_time_for_revision_s
        ):
            return "refine"
        return "finalize"

    def after_refine(state: AgentState) -> str:
        return "retrieve" if state.get("pending_queries") else "synthesize"

    g = StateGraph(AgentState)
    g.add_node("plan", nodes.plan)
    g.add_node("standard", nodes.standard)
    g.add_node("retrieve", nodes.retrieve)
    g.add_node("synthesize", nodes.synthesize)
    g.add_node("verify", nodes.verify)
    g.add_node("refine", nodes.refine)
    g.add_node("finalize", nodes.finalize)

    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", after_plan, ["standard", "retrieve", "finalize"])
    g.add_edge("standard", "finalize")
    g.add_conditional_edges("retrieve", after_step("synthesize"), ["synthesize", "finalize"])
    g.add_conditional_edges("synthesize", after_step("verify"), ["verify", "finalize"])
    g.add_conditional_edges("verify", after_verify, ["refine", "finalize"])
    g.add_conditional_edges("refine", after_refine, ["retrieve", "synthesize"])
    g.add_edge("finalize", END)
    return g.compile()


def recursion_limit(settings: AgentSettings) -> int:
    # plan, retrieve, synthesize, verify, finalize + 4 nodes per revision, with slack.
    return 6 + 4 * settings.max_revisions + 2


def run_agent(
    question: str,
    retriever: SearchableRetriever,
    chat_model: ChatModel,
    *,
    verifier_model: Optional[ChatModel] = None,
    settings: Optional[AgentSettings] = None,
) -> AgentResult:
    """Answer `question` with the agentic workflow. Never raises for model/backend failures."""
    settings = settings or AgentSettings()
    deps = AgentDeps(
        tools=RetrievalTools(retriever),
        chat_model=chat_model,
        verifier_model=verifier_model,
        settings=settings,
    )
    start = time.monotonic()
    initial: AgentState = {
        "question": question,
        "started_at": start,
        "deadline": start + settings.time_budget_s,
        "status": "running",
        "revision_count": 0,
        "evidence": [],
        "trace": [],
        "llm_calls": 0,
        "retrieval_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }
    try:
        final = build_graph(deps).invoke(initial, config={"recursion_limit": recursion_limit(settings)})
    except GraphRecursionError:
        final = {**initial, "status": "failed", "error": "agent exceeded its step limit",
                 "answer": "The agent could not complete this question: step limit exceeded."}

    return AgentResult(
        question=question,
        answer=final.get("answer", ""),
        route=final.get("route", "simple"),
        status=final.get("status", "failed"),
        evidence=final.get("evidence", []),
        citations=final.get("citations"),
        verification=final.get("verification"),
        plan=final.get("plan", {}),
        planner_fallback=final.get("planner_fallback"),
        revision_count=final.get("revision_count", 0),
        trace=final.get("trace", []),
        llm_calls=final.get("llm_calls", 0),
        retrieval_calls=final.get("retrieval_calls", 0),
        prompt_tokens=final.get("prompt_tokens", 0),
        completion_tokens=final.get("completion_tokens", 0),
        elapsed_s=time.monotonic() - start,
        error=final.get("error"),
    )
