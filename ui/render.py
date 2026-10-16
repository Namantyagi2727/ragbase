"""
ui/render.py — Answer views: citations, verification status and agent execution summaries.

A *view* is a plain, JSON-serialisable dict built from either a Standard RAG
result or an AgentResult, so live answers and recorded showcase runs render
through the same code. The *_html functions return escaped HTML strings and
are tested without Streamlit; render_view() draws a view with Streamlit.

Wording rule: a citation or a verifier verdict is evidence about the answer,
never a guarantee that it is factually correct.
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:   # no runtime import: the public showcase must not load the agent or retrieval stack
    from citations import Evidence

# Same wording as agent.sources.METHOD_TEXT (kept equal by a test).
METHOD_TEXT = {
    "named": "named in the question",
    "named+relevance": "named in the question, plus a relevance search",
    "relevance": "relevance search",
    "reused": "reused from the first round",
}

STATUS_TEXT = {
    "completed": "completed",
    "completed_unverified": "completed — not fully checked",
    "qualified": "completed with warnings",
    "failed": "failed",
    "timeout": "stopped: time budget exhausted",
}


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _evidence_dicts(evidence: list["Evidence"]) -> list[dict]:
    return [{"eid": e.eid, "source": e.source, "loc": e.loc, "text": e.text} for e in evidence]


def _llm_info(result) -> dict:
    meta = result.metadata if hasattr(result, "metadata") else {}
    info = {"backend": getattr(result, "backend", None), "model": getattr(result, "model", None)}
    for key in ("provider_used", "inference_mode", "fallback_triggered", "cost_usd"):
        if meta.get(key) is not None:
            info[key] = meta[key]
    return info


def view_from_standard(result: dict, latency_s: float) -> dict:
    """View of a retriever.build_rag_chain() result."""
    report, llm = result["citations"], result.get("llm")
    return {
        "mode": "standard",
        "answer": result["answer"],
        "evidence": _evidence_dicts(result["evidence"]),
        "cited_ids": list(report.valid_ids),
        "invalid_ids": list(report.invalid_ids),
        "non_evidence_citations": len(report.non_evidence_citations),
        "abstained": report.abstained,
        "route": None,
        "status": None,
        "verification": None,
        "trace": [],
        "source_selections": [],
        "metrics": {
            "llm_calls": 1, "retrieval_calls": 1, "latency_s": round(latency_s, 2),
            "prompt_tokens": getattr(llm, "prompt_tokens", None),
            "completion_tokens": getattr(llm, "completion_tokens", None),
        },
        "llm": [_llm_info(llm)] if llm is not None else [],
    }


def view_from_agent(res) -> dict:
    """View of an agent.AgentResult."""
    report = res.citations
    return {
        "mode": "agentic",
        "answer": res.answer,
        "evidence": _evidence_dicts(res.evidence),
        "cited_ids": list(report.valid_ids) if report else [],
        "invalid_ids": list(report.invalid_ids) if report else [],
        "non_evidence_citations": len(report.non_evidence_citations) if report else 0,
        "abstained": bool(report and report.abstained),
        "route": res.route,
        "status": res.status,
        "verification": res.verification.to_dict() if res.verification else None,
        "trace": [dict(e) for e in res.trace],
        "source_selections": list(res.source_selections),
        "metrics": {
            "llm_calls": res.llm_calls, "retrieval_calls": res.retrieval_calls,
            "prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens,
            "latency_s": round(res.elapsed_s, 2), "revisions": res.revision_count,
        },
        "llm": [{k: m.get(k) for k in ("backend", "model", "provider_used", "inference_mode", "latency_s")
                 if m.get(k) is not None} for m in res.llm_meta],
        "error": res.error,
    }


# ---------------------------------------------------------------------------
# Answer text
# ---------------------------------------------------------------------------

def answer_html(text: str) -> str:
    """Model output as escaped text, never as Markdown or HTML.

    Rendering model output as Markdown would let a prompt-injected document make
    the browser fetch an attacker's image URL (leaking text in the query string)
    or show disguised links. Lines starting with "> " (RAGBase's own notes)
    are shown as notes.
    """
    parts = []
    for line in (text or "").split("\n"):
        if line.startswith("> "):
            parts.append(f'<div class="rb-answer-note">{_e(line[2:])}</div>')
        else:
            parts.append(_e(line) + "<br>")
    return f'<div class="rb-answer">{"".join(parts)}</div>'


# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------

def split_evidence(view: dict) -> tuple[list[dict], list[dict]]:
    """(cited passages in citation order, retrieved passages the answer did not cite)."""
    by_id = {e["eid"]: e for e in view["evidence"]}
    cited = [by_id[i] for i in view["cited_ids"] if i in by_id]
    cited_ids = {e["eid"] for e in cited}
    return cited, [e for e in view["evidence"] if e["eid"] not in cited_ids]


def citation_warnings(view: dict) -> list[str]:
    warnings = []
    if view["invalid_ids"]:
        warnings.append("The answer cites evidence that was not retrieved: " + ", ".join(view["invalid_ids"]))
    if view["non_evidence_citations"]:
        warnings.append("The answer contains references that are not evidence IDs; they may be invented.")
    if not view["abstained"] and not view["cited_ids"]:
        warnings.append("The answer has no citations to the retrieved evidence.")
    return warnings


def passage_html(e: dict, cited: bool, full: bool = False) -> str:
    text = e["text"] if full else e["text"][:220].replace("\n", " ") + ("…" if len(e["text"]) > 220 else "")
    loc = f'<div class="rb-source-loc">@ {_e(e["loc"])}</div>' if e.get("loc") else ""
    if not e.get("eid"):
        eid = ""
    elif cited:
        eid = f'<span class="rb-eid">[{_e(e["eid"])}]</span> '
    else:
        eid = f'<span class="rb-eid-dim">{_e(e["eid"])}</span> '
    return (f'<div class="rb-source{"" if cited else " uncited"}">'
            f'<div class="rb-source-path"><i class="rb-source-icon">◈</i>{eid}{_e(e["source"])}</div>'
            f'{loc}<div class="rb-source-snippet">{_e(text)}</div></div>')


# ---------------------------------------------------------------------------
# Verification status
# ---------------------------------------------------------------------------

def verification_summary(view: dict) -> dict:
    """{level: ok|warn|neutral|error, title, details[]} with wording that never implies proof."""
    v, status, details = view.get("verification"), view.get("status"), []
    citations_line = ("Citation IDs refer to retrieved passages." if not view["invalid_ids"]
                      and not view["non_evidence_citations"] else "Some citations do not refer to retrieved passages.")
    if view["mode"] == "standard":
        return {"level": "neutral", "title": "Not checked — Standard RAG does not verify claims",
                "details": [citations_line, "Citations show where an answer came from; they do not prove it is correct."]}
    if status in ("failed", "timeout"):
        return {"level": "error", "title": f"Agent {STATUS_TEXT[status]}", "details": [view.get("error") or ""]}
    if v is None:
        return {"level": "neutral", "title": "Not checked", "details": [citations_line]}
    if v["grounding"] == "skipped":
        title = {
            "no_evidence": "Declined to answer: no passages were retrieved",
            "confirmed": "Declined to answer: the verifier also found no answer in the retrieved passages",
            "evidence_may_answer": "Declined to answer, but the retrieved passages may contain an answer",
            "not_checked": "Declined to answer (not checked)",
            "failed": "Declined to answer (the check failed)",
        }.get(v["abstention_check"], "Declined to answer")
        details.append("This is judged from the retrieved passages only; it does not show the documents lack the answer.")
        return {"level": "warn" if v["abstention_check"] == "evidence_may_answer" else "neutral",
                "title": title, "details": details}
    if v["llm_status"] == "skipped":
        title, level = "Not checked — simple route answers without claim verification", "neutral"
    elif v["llm_status"] == "failed":
        title, level = "Claim check failed — the answer is not verified", "warn"
    elif v["grounding"] == "verified":
        title, level = (f"Checked: all {v['claims_checked']} claim(s) were judged supported by their cited "
                        "passages"), "ok"
    elif v["grounding"] in ("incomplete", "not_verified", "partially_verified"):
        title, level = f"Partly checked: {v['claims_verified']} of {v['claims_checked']} claim(s) confirmed", "warn"
    else:
        title, level = "Some statements could not be confirmed against the cited passages", "warn"
    for claim in v["unsupported_claims"]:
        details.append(f"Not supported by its citation: {claim}")
    for claim in v["numeric_mismatches"]:
        details.append(f"Number not found in the cited passage: {claim}")
    if v["unverified_claims"]:
        details.append(f"{len(v['unverified_claims'])} claim(s) received no usable verdict.")
    if v["absence_claims"]:
        details.append("Statements that information is missing are based on the retrieved passages"
                       + (" and one targeted search" if v["absence_searched"] else "")
                       + "; they do not establish that the documents lack it.")
    details.append(citations_line)
    if level == "ok":
        details.append("Automated check by a language model; it can be wrong and is not a guarantee of correctness.")
    return {"level": level, "title": title, "details": details}


def verification_html(view: dict) -> str:
    s = verification_summary(view)
    items = "".join(f"<li>{_e(d)}</li>" for d in s["details"] if d)
    return (f'<div class="rb-verify {s["level"]}"><div class="rb-verify-title">{_e(s["title"])}</div>'
            f'{"<ul>" + items + "</ul>" if items else ""}</div>')


# ---------------------------------------------------------------------------
# Agent execution summary (structured events only; no model reasoning text)
# ---------------------------------------------------------------------------

NODE_LABELS = {"plan": "Plan", "retrieve": "Retrieve", "answer": "Answer", "synthesize": "Write answer",
               "verify": "Verify", "refine": "Refine", "finalize": "Finish"}


def trace_rows(view: dict) -> list[dict]:
    """[{node, label, summary, details[], elapsed_s}] from the agent's trace events."""
    rows = []
    for ev in view.get("trace") or []:
        node, d, details = ev["node"], ev.get("data") or {}, []
        if node == "plan":
            details.append(f"Route: {d.get('route')}")
            for sq in d.get("subqueries") or []:
                where = ", ".join(sq.get("sources") or []) or "all documents"
                details.append(f"Sub-question: {sq['query']}  ({where})")
            if d.get("compare_documents"):
                details.append("Comparison across documents")
            if d.get("fallback"):
                details.append("Planner output was invalid; used the standard path")
        elif node == "retrieve":
            for q in d.get("queries") or []:
                details.append(f"hybrid_search: {q}  (all documents)")
            for s in d.get("searches") or []:
                if "error" in s:
                    details.append(f"hybrid_search: {s['query']} — rejected: {s['error']}")
                    continue
                where = ", ".join(s.get("sources") or []) or "all documents"
                details.append(f"hybrid_search: {s['query']}  ({where}) → {s['results']} result(s), {s['new']} new")
                if s.get("dropped_sources"):
                    details.append(f"Ignored unknown documents: {', '.join(s['dropped_sources'])}")
            sel = d.get("source_selection")
            if sel:
                how = METHOD_TEXT.get(sel["method"], sel["method"])
                details.append(f"Documents searched ({how}): {', '.join(sel['selected']) or 'none'}")
                if sel.get("limitation"):
                    details.append(f"Limitation: {sel['limitation']}")
            if d.get("added"):
                details.append(f"New evidence: {', '.join('[' + a + ']' for a in d['added'])}")
        elif node in ("synthesize", "answer"):
            if d.get("cited") is not None:
                details.append(f"Cited: {', '.join('[' + c + ']' for c in d['cited']) or 'none'}")
            if d.get("invalid"):
                details.append(f"Invalid citations: {', '.join(d['invalid'])}")
            if d.get("abstained"):
                details.append("Declined to answer")
        elif node == "verify":
            if d.get("claims_checked"):
                details.append(f"Claims confirmed: {d.get('claims_verified', 0)} of {d['claims_checked']}")
            for key, label in (("unsupported", "unsupported"), ("unverified", "without a usable verdict"),
                               ("numeric_mismatches", "with numbers missing from the citation"),
                               ("absence", "missing-information statement(s), not confirmable")):
                if d.get(key):
                    details.append(f"{d[key]} {label}")
            if d.get("followup_query"):
                details.append(f"Suggested follow-up search: {d['followup_query']}")
        elif node == "refine":
            if d.get("followup_query"):
                details.append(f"Follow-up search: {d['followup_query']}")
        elif node == "finalize":
            if d.get("revisions"):
                details.append(f"Revisions: {d['revisions']}")
        rows.append({"node": node, "label": NODE_LABELS.get(node, node), "summary": ev["summary"],
                     "details": details, "elapsed_s": ev.get("elapsed_s")})
    return rows


def trace_html(view: dict) -> str:
    out = ['<div class="rb-trace">']
    for r in trace_rows(view):
        items = "".join(f"<li>{_e(x)}</li>" for x in r["details"])
        out.append(f'<div class="rb-trace-step"><div class="rb-trace-head"><span class="rb-trace-node">'
                   f'{_e(r["label"])}</span><span class="rb-trace-summary">{_e(r["summary"])}</span>'
                   f'<span class="rb-trace-time">{_e(r["elapsed_s"])}s</span></div>'
                   f'{"<ul>" + items + "</ul>" if items else ""}</div>')
    out.append("</div>")
    return "".join(out)


def metrics_html(view: dict) -> str:
    m = view["metrics"]
    parts = [("model calls", m.get("llm_calls")), ("retrieval calls", m.get("retrieval_calls")),
             ("prompt tokens", m.get("prompt_tokens")), ("completion tokens", m.get("completion_tokens")),
             ("time", f"{m['latency_s']}s" if m.get("latency_s") is not None else None)]
    chips = "".join(f'<span class="rb-metric"><b>{_e(v)}</b> {_e(k)}</span>' for k, v in parts if v is not None)
    providers = sorted({x.get("provider_used") or x.get("backend") for x in view.get("llm") or [] if x})
    if providers:
        chips += f'<span class="rb-metric"><b>{_e(", ".join(p for p in providers if p))}</b> inference</span>'
    return f'<div class="rb-metrics">{chips}</div>'


# ---------------------------------------------------------------------------
# Streamlit
# ---------------------------------------------------------------------------

def render_view(view: dict) -> None:
    """Draw everything below an answer: status, citations, sources and (agentic) the execution summary."""
    import streamlit as st

    st.markdown(verification_html(view), unsafe_allow_html=True)
    for w in citation_warnings(view):
        st.markdown(f'<div class="rb-cite-warn">⚠ {_e(w)}</div>', unsafe_allow_html=True)
    cited, uncited = split_evidence(view)
    if cited:
        with st.expander(f"◈  {len(cited)} cited passage(s)"):
            st.caption("Citations show which passages the answer drew on; they do not prove it is correct.")
            for e in cited:
                st.markdown(passage_html(e, cited=True, full=True), unsafe_allow_html=True)
    if uncited:
        with st.expander(f"◇  {len(uncited)} retrieved but not cited"):
            for e in uncited:
                st.markdown(passage_html(e, cited=False), unsafe_allow_html=True)
    if view["mode"] == "agentic" and view.get("trace"):
        with st.expander(f"⬡  Agent execution — route: {view.get('route')}, status: "
                         f"{STATUS_TEXT.get(view.get('status'), view.get('status'))}"):
            st.markdown(trace_html(view), unsafe_allow_html=True)
    st.markdown(metrics_html(view), unsafe_allow_html=True)
