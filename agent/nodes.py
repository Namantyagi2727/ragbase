"""
agent/nodes.py — Graph node implementations.

Each node takes the current state and returns a partial update. Nodes never
raise for expected failures: backend errors set status="failed", an
exhausted time budget sets status="timeout", and the graph routes both to
`finalize`. Model decisions that fail validation take recorded fallbacks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import config
from agent.prompts import (
    abstention_check_messages,
    planner_messages,
    synthesis_messages,
    verifier_messages,
)
from agent.schemas import AbstentionCheck, QueryPlan, VerifierOutput
from agent.state import AgentState, TraceEvent
from agent.tools import RetrievalTools, ToolError
from agent.verifier import assess, assess_abstention, revision_feedback
from citations import (
    ABSTAIN_TEXT,
    build_evidence,
    is_absence_statement,
    normalize_citations,
    split_claims,
    validate_citations,
)
from llm import ChatModel, LLMError, LLMResult
from llm.structured import complete_structured
from retriever import build_standard_messages


@dataclass
class AgentSettings:
    max_revisions: int = config.AGENT_MAX_REVISIONS
    time_budget_s: float = config.AGENT_TIME_BUDGET_S
    max_subqueries: int = config.AGENT_MAX_SUBQUERIES
    k_per_query: int = config.AGENT_K_PER_QUERY
    simple_k: int = config.TOP_K
    max_evidence: int = config.AGENT_MAX_EVIDENCE
    max_compare_docs: int = config.AGENT_MAX_COMPARE_DOCS
    verify_with_llm: bool = True
    max_verified_claims: int = 12
    min_time_for_revision_s: float = 15.0


@dataclass
class AgentDeps:
    tools: RetrievalTools
    chat_model: ChatModel
    verifier_model: Optional[ChatModel] = None
    settings: AgentSettings = field(default_factory=AgentSettings)

    @property
    def verifier(self) -> ChatModel:
        return self.verifier_model or self.chat_model


_META_KEYS = ("inference_mode", "provider_used", "fallback_triggered", "cost_usd", "gateway_latency_ms")


def _accounting(results: list[LLMResult]) -> dict:
    return {
        "llm_calls": len(results),
        "prompt_tokens": sum(r.prompt_tokens or 0 for r in results),
        "completion_tokens": sum(r.completion_tokens or 0 for r in results),
        "llm_meta": [
            {
                "backend": r.backend,
                "model": r.model,
                "latency_s": round(r.latency_s, 3),
                **{k: r.metadata[k] for k in _META_KEYS if r.metadata.get(k) is not None},
                **({"prompt_tokens": r.prompt_tokens} if r.prompt_tokens is not None else {}),
                **({"completion_tokens": r.completion_tokens} if r.completion_tokens is not None else {}),
            }
            for r in results
        ],
    }


def _clean_query(query: str, documents: list[str]) -> str:
    """Strip file-name prefixes small models put inside query text ("handbook.txt: leave")."""
    q = query.strip()
    for doc in documents:
        for prefix in (doc, doc.rsplit(".", 1)[0]):
            if q.lower().startswith(prefix.lower()) and q[len(prefix):len(prefix) + 1] in (":", " "):
                q = q[len(prefix) + 1:].strip()
    return " ".join(q.split())


def _partition_claims(answer: str) -> tuple[list[str], list[str]]:
    """(checkable claims, absence statements) of an answer."""
    claims = split_claims(answer)
    absence = [c for c in claims if is_absence_statement(c)]
    return [c for c in claims if c not in absence], absence


class AgentNodes:
    def __init__(self, deps: AgentDeps):
        self.deps = deps
        self.s = deps.settings

    # -- helpers ---------------------------------------------------------------

    def _event(
        self, state: AgentState, node: str, summary: str, *, offset: int = 0, **data: Any
    ) -> TraceEvent:
        """A trace event; `offset` numbers additional events emitted by the same node."""
        return {
            "step": len(state.get("trace", [])) + 1 + offset,
            "node": node,
            "summary": summary,
            "data": data,
            "elapsed_s": round(time.monotonic() - state["started_at"], 3),
        }

    def _remaining(self, state: AgentState) -> float:
        return state["deadline"] - time.monotonic()

    def _timed_out(self, state: AgentState, node: str) -> Optional[dict]:
        if self._remaining(state) > 0:
            return None
        return {
            "status": "timeout",
            "error": f"time budget of {self.s.time_budget_s:.0f}s exhausted before {node}",
            "trace": [self._event(state, node, "Stopped: time budget exhausted")],
        }

    def _failed(self, state: AgentState, node: str, err: Exception | str) -> dict:
        msg = f"{type(err).__name__}: {err}" if isinstance(err, Exception) else err
        return {
            "status": "failed",
            "error": msg,
            "trace": [self._event(state, node, "Stopped: language model request failed", error=msg)],
        }

    # -- nodes -----------------------------------------------------------------

    def plan(self, state: AgentState) -> dict:
        if (stop := self._timed_out(state, "plan")):
            return stop
        question = state["question"]
        docs = self.deps.tools.list_documents()
        res = complete_structured(
            self.deps.chat_model, planner_messages(question, docs), QueryPlan,
            timeout_s=self._remaining(state), temperature=0.0,
        )
        update: dict = _accounting(res.llm_results)
        if res.backend_error:
            return {**update, **self._failed(state, "plan", res.error)}

        if not res.ok:
            update.update(route="simple", planner_fallback=res.error, plan={})
            update["trace"] = [self._event(
                state, "plan",
                "Planner output was invalid; falling back to standard retrieval",
                route="simple", fallback=res.error, attempts=res.attempts,
            )]
            return update

        plan: QueryPlan = res.value
        subqueries = [
            {"query": _clean_query(sq.query, docs), "sources": sq.sources}
            for sq in plan.subqueries[: self.s.max_subqueries]
        ]
        subqueries = [sq for sq in subqueries if sq["query"]]
        if plan.route == "complex" and not subqueries:
            subqueries = [{"query": question[:300], "sources": []}]
        update.update(
            route=plan.route,
            plan=plan.model_dump(),
            planner_fallback=None,
            pending_queries=subqueries if plan.route == "complex" else [],
        )
        if plan.route == "simple":
            summary = "Planned a single search (simple question)"
        else:
            summary = f"Planned {len(subqueries)} search(es)" + (
                " across documents for comparison" if plan.compare_documents else ""
            )
        update["trace"] = [self._event(
            state, "plan", summary,
            route=plan.route, rationale=plan.rationale,
            subqueries=subqueries, compare_documents=plan.compare_documents,
        )]
        return update

    def standard(self, state: AgentState) -> dict:
        """Simple route: identical to Standard RAG (one search, one answer call)."""
        if (stop := self._timed_out(state, "standard")):
            return stop
        question = state["question"]
        try:
            outcome = self.deps.tools.hybrid_search(question, k=self.s.simple_k)
        except ToolError as e:
            return self._failed(state, "standard", str(e))
        evidence = build_evidence(outcome.documents)
        try:
            result = self.deps.chat_model.complete(
                build_standard_messages(question, evidence), timeout_s=self._remaining(state)
            )
        except LLMError as e:
            return {"retrieval_calls": 1, "evidence": evidence, **self._failed(state, "standard", e)}
        answer, normalized = normalize_citations(result.text.strip())
        report = validate_citations(answer, evidence)
        if report.abstained:
            verification = assess_abstention(evidence, report, None, attempted=False)
        else:
            checkable, absence = _partition_claims(answer)
            verification = assess(
                answer, evidence, report, None, llm_skipped=True,
                claims=checkable[: self.s.max_verified_claims], total_checkable=len(checkable),
                absence_claims=absence,
            )
        return {
            **_accounting([result]),
            "retrieval_calls": 1,
            "evidence": evidence,
            "answer": answer,
            "citations": report,
            "verification": verification,
            "trace": [
                self._event(state, "retrieve", f"Retrieved {len(evidence)} passage(s)",
                            queries=[question], evidence=len(evidence)),
                self._event(state, "answer", "Answered with standard RAG; claim verification skipped for simple route",
                            offset=1, citation_ok=report.ok, cited=report.valid_ids,
                            citations_normalized=normalized),
            ],
        }

    def retrieve(self, state: AgentState) -> dict:
        if (stop := self._timed_out(state, "retrieve")):
            return stop
        tools = self.deps.tools
        existing = list(state.get("evidence", []))
        seen = {e.chunk_id for e in existing}
        compare = bool(state.get("plan", {}).get("compare_documents"))

        searches: list[tuple[str, list[str], int]] = []
        for sq in state.get("pending_queries", []):
            if compare and not sq.get("sources"):
                per_doc_k = max(2, self.s.k_per_query // 2)
                for doc in tools.list_documents()[: self.s.max_compare_docs]:
                    searches.append((sq["query"], [doc], per_doc_k))
            else:
                searches.append((sq["query"], sq.get("sources", []), self.s.k_per_query))

        new_docs = []
        log = []
        for query, sources, k in searches:
            try:
                outcome = tools.hybrid_search(query, k=k, sources=sources)
            except ToolError as e:
                log.append({"query": query[:120], "error": str(e)})
                continue
            fresh = [d for d in outcome.documents if d.metadata.get("chunk_id") not in seen]
            for d in fresh:
                seen.add(d.metadata.get("chunk_id"))
            new_docs.extend(fresh)
            log.append({
                "query": outcome.query, "sources": outcome.sources,
                "dropped_sources": outcome.dropped_sources,
                "results": len(outcome.documents), "new": len(fresh),
            })

        room = max(0, self.s.max_evidence - len(existing))
        added = build_evidence(new_docs[:room], start=len(existing) + 1)
        return {
            "evidence": existing + added,
            "pending_queries": [],
            "retrieval_calls": len(searches),
            "trace": [self._event(
                state, "retrieve",
                f"Ran {len(searches)} search(es); {len(added)} new passage(s), {len(existing) + len(added)} total",
                searches=log, added=[e.eid for e in added],
            )],
        }

    def synthesize(self, state: AgentState) -> dict:
        if (stop := self._timed_out(state, "synthesize")):
            return stop
        evidence = state.get("evidence", [])
        if not evidence:
            answer = ABSTAIN_TEXT
            return {
                "answer": answer,
                "citations": validate_citations(answer, evidence),
                "trace": [self._event(state, "synthesize", "No evidence found; abstaining")],
            }
        try:
            result = self.deps.chat_model.complete(
                synthesis_messages(state["question"], evidence, state.get("revision_feedback")),
                timeout_s=self._remaining(state),
            )
        except LLMError as e:
            return self._failed(state, "synthesize", e)
        answer, normalized = normalize_citations(result.text.strip())
        report = validate_citations(answer, evidence)
        revision = state.get("revision_count", 0)
        return {
            **_accounting([result]),
            "answer": answer,
            "citations": report,
            "trace": [self._event(
                state, "synthesize",
                ("Revised answer" if revision else "Drafted answer")
                + f" citing {len(report.valid_ids)} of {len(evidence)} passage(s)",
                cited=report.valid_ids, invalid=report.invalid_ids,
                coverage=round(report.citation_coverage, 2), abstained=report.abstained,
                citations_normalized=normalized,
            )],
        }

    def verify(self, state: AgentState) -> dict:
        answer, evidence, report = state["answer"], state.get("evidence", []), state["citations"]
        update: dict = {}
        out_of_time = self._remaining(state) <= 0

        if report.abstained:
            if not evidence or not self.s.verify_with_llm:
                v = assess_abstention(evidence, report, None, attempted=False)
            elif out_of_time:
                v = assess_abstention(evidence, report, None, attempted=True, llm_error="time budget exhausted")
            else:
                res = complete_structured(
                    self.deps.verifier, abstention_check_messages(state["question"], evidence),
                    AbstentionCheck, timeout_s=self._remaining(state), temperature=0.0,
                )
                update.update(_accounting(res.llm_results))
                v = assess_abstention(evidence, report, res.value, attempted=True, llm_error=res.error)
        else:
            checkable, absence = _partition_claims(answer)
            claims = checkable[: self.s.max_verified_claims]
            common = dict(
                claims=claims, total_checkable=len(checkable), absence_claims=absence,
                absence_searched=state.get("absence_searched", False),
            )
            if not self.s.verify_with_llm or not claims:
                v = assess(answer, evidence, report, None, llm_skipped=True, **common)
            elif out_of_time:
                v = assess(answer, evidence, report, None, llm_error="time budget exhausted", **common)
            else:
                res = complete_structured(
                    self.deps.verifier, verifier_messages(state["question"], evidence, claims),
                    VerifierOutput, timeout_s=self._remaining(state), temperature=0.0,
                )
                update.update(_accounting(res.llm_results))
                v = assess(answer, evidence, report, res.value, llm_error=res.error, **common)

        if v.grounding == "skipped":
            summary = {
                "no_evidence": "Abstained; no evidence was retrieved",
                "confirmed": "Abstention checked: retrieved evidence does not answer the question",
                "evidence_may_answer": "Abstention challenged: retrieved evidence may answer the question",
                "not_checked": "Abstained; not checked against the retrieved evidence",
                "failed": "Abstention check failed",
            }[v.abstention_check]
        elif v.llm_status == "failed":
            summary = "Claim verification failed; answer is not verified"
        elif v.grounding == "verified":
            summary = f"All {v.claims_checked} claim(s) supported by their cited evidence"
        elif v.grounding == "incomplete":
            summary = f"Verification incomplete: {v.claims_verified} of {v.claims_checked} claim(s) verified" + (
                f", {len(v.absence_claims)} missing-information statement(s) unconfirmed" if v.absence_claims else ""
            )
        else:
            summary = f"Verification found problems ({v.grounding.replace('_', ' ')})"
        update.update(
            verification=v,
            trace=[self._event(
                state, "verify", summary,
                citation_ok=v.citation_ok, grounding=v.grounding, llm_status=v.llm_status,
                decision=v.decision, claims_checked=v.claims_checked, claims_verified=v.claims_verified,
                unsupported=len(v.unsupported_claims), unverified=len(v.unverified_claims),
                absence=len(v.absence_claims), numeric_mismatches=len(v.numeric_mismatches),
                abstention_check=v.abstention_check, followup_query=v.followup_query,
            )],
        )
        return update

    def refine(self, state: AgentState) -> dict:
        v = state["verification"]
        count = state.get("revision_count", 0) + 1
        pending = [{"query": v.followup_query, "sources": []}] if v.followup_query else []
        absence_round = bool(pending and v.absence_claims and not v.absence_searched)
        return {
            "revision_count": count,
            "absence_searched": state.get("absence_searched", False) or absence_round,
            "revision_feedback": revision_feedback(v),
            "pending_queries": pending,
            "trace": [self._event(
                state, "refine",
                f"Refinement round {count}: "
                + ("searching for missing information" if pending else "revising with verifier feedback"),
                followup_query=v.followup_query,
            )],
        }

    def finalize(self, state: AgentState) -> dict:
        status = state.get("status", "running")
        answer = state.get("answer", "")
        v = state.get("verification")

        if status == "failed":
            answer = f"The agent could not complete this question: {state.get('error')}"
        elif status == "timeout" and not answer:
            answer = "The agent ran out of time before producing an answer."
        elif status == "timeout":
            pass
        elif v is None:
            status = "completed_unverified"
        elif v.grounding == "skipped":   # full abstention
            if v.abstention_check in ("no_evidence", "confirmed"):
                status = "completed"
            elif v.abstention_check == "evidence_may_answer":
                status = "qualified"
            else:
                status = "completed_unverified"
        elif v.decision == "accept" and v.grounding == "verified":
            status = "completed"
        elif (
            v.citation_ok
            and v.grounding in ("not_verified", "incomplete")
            and not v.numeric_mismatches
            and not v.unsupported_claims
        ):
            status = "completed_unverified"
        else:
            status = "qualified"

        if status == "qualified":
            problems = len(v.unsupported_claims) + len(v.numeric_mismatches) if v else 0
            if v is not None and v.abstention_check == "evidence_may_answer":
                note = "the retrieved passages may contain an answer to this question"
            elif problems:
                note = f"{problems} statement(s) could not be confirmed against the documents"
            else:
                note = "the answer's citations or grounding could not be fully confirmed"
            answer = f"{answer}\n\n> ⚠ Verification: {note}."
        if v is not None and v.absence_claims and status not in ("failed", "timeout"):
            searched = " and one targeted search" if v.absence_searched else ""
            answer = (
                f"{answer}\n\n> Note: statements that information is missing are based on the "
                f"passages retrieved{searched}; they do not establish that the documents lack it."
            )

        return {
            "status": status,
            "answer": answer,
            "trace": [self._event(state, "finalize", f"Finished: {status.replace('_', ' ')}",
                                  revisions=state.get("revision_count", 0))],
        }
