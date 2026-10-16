"""Answer views: citations, verification wording, execution summaries, providers, upload workspaces."""

import os
import time

import pytest

from agent import AgentSettings, run_agent
from citations import Evidence, validate_citations
from tests.unit.agent_helpers import MemoryRetriever, RoleModel
from ui import providers, uploads
from ui import render as rv

EVIDENCE = [Evidence("1", "c1", "policy_2023.txt", "p.1", "The 2023 meal allowance is 45 dollars."),
            Evidence("2", "c2", "policy_2024.txt", "", "The 2024 meal allowance is 60 dollars."),
            Evidence("3", "c3", "handbook.txt", "", "Unrelated <script>alert(1)</script> passage.")]


class _LLM:
    prompt_tokens, completion_tokens, backend, model, metadata = 100, 10, "ollama", "llama3.2", {}


def _standard(answer):
    return rv.view_from_standard({"answer": answer, "evidence": EVIDENCE,
                                  "citations": validate_citations(answer, EVIDENCE), "llm": _LLM()}, 0.8)


def _agent_view(plan, synth, verifier, question="Compare the meal allowance in 2023 and 2024."):
    model = RoleModel(planner=[plan], synth=[synth], verifier=[verifier])
    res = run_agent(question, MemoryRetriever(), model, settings=AgentSettings(k_per_query=1, max_revisions=0))
    return rv.view_from_agent(res)


COMPARE = {"route": "complex", "compare_documents": True, "rationale": "PRIVATE REASONING TEXT",
           "subqueries": [{"query": "meal allowance"}]}
ACCEPT = {"verdicts": [{"claim_number": 1, "status": "supported", "evidence_ids": ["1"]},
                       {"claim_number": 2, "status": "supported", "evidence_ids": ["2"]}], "recommendation": "accept"}
TWO = "In 2023 the meal allowance was 45 dollars [1]. In 2024 it rose to 60 dollars [2]."


# --- citations ------------------------------------------------------------------

def test_cited_and_uncited_passages_are_separated_in_citation_order():
    view = _standard("It rose to 60 dollars [2], from 45 dollars [1].")
    cited, uncited = rv.split_evidence(view)
    assert [e["eid"] for e in cited] == ["2", "1"] and [e["eid"] for e in uncited] == ["3"]


def test_all_document_and_model_text_is_escaped():
    html_out = rv.passage_html(EVIDENCE_DICT := {"eid": "3", "source": "a<b>.txt", "loc": "<i>",
                                                 "text": "x <script>alert(1)</script>"}, cited=False, full=True)
    assert "<script>" not in html_out and "&lt;script&gt;" in html_out and "a&lt;b&gt;.txt" in html_out
    assert EVIDENCE_DICT["text"].startswith("x")


def test_citation_warnings_report_invalid_and_missing_citations():
    assert any("not retrieved" in w for w in rv.citation_warnings(_standard("It is 60 dollars [7].")))
    assert any("no citations" in w for w in rv.citation_warnings(_standard("It is 60 dollars.")))
    assert rv.citation_warnings(_standard("It is 60 dollars [2].")) == []


# --- verification wording --------------------------------------------------------

def test_standard_mode_is_never_presented_as_verified():
    s = rv.verification_summary(_standard("It is 60 dollars [2]."))
    assert s["level"] == "neutral" and "Not checked" in s["title"]
    assert any("do not prove" in d for d in s["details"])


def test_verified_agent_answer_still_carries_a_no_guarantee_note():
    view = _agent_view(COMPARE, TWO, ACCEPT)
    s = rv.verification_summary(view)
    assert s["level"] == "ok" and "judged supported" in s["title"]
    assert any("not a guarantee" in d for d in s["details"])


def test_unsupported_claims_are_listed_and_flagged():
    reject = {"verdicts": [{"claim_number": 1, "status": "unsupported", "evidence_ids": ["1"]},
                           {"claim_number": 2, "status": "supported", "evidence_ids": ["2"]}],
              "recommendation": "accept"}
    s = rv.verification_summary(_agent_view(COMPARE, TWO, reject))
    assert s["level"] == "warn" and any(d.startswith("Not supported by its citation") for d in s["details"])


def test_abstention_wording_is_limited_to_retrieved_passages():
    model = RoleModel(planner=[{"route": "complex", "subqueries": [{"query": "meal allowance"}]}],
                      synth=["I don't have enough information in the provided documents to answer that."],
                      abscheck=[{"answerable": "no"}])
    res = run_agent("Can employees bring pets?", MemoryRetriever(), model,
                    settings=AgentSettings(k_per_query=1, max_revisions=0))
    s = rv.verification_summary(rv.view_from_agent(res))
    assert "Declined" in s["title"] and "verifier also found no answer" in s["title"]
    assert any("retrieved passages only" in d for d in s["details"])


# --- execution summary --------------------------------------------------------------

def test_trace_shows_route_subquestions_tools_documents_and_verification():
    view = _agent_view(COMPARE, TWO, ACCEPT)
    rows = rv.trace_rows(view)
    text = " ".join(r["summary"] + " " + " ".join(r["details"]) for r in rows)
    assert [r["node"] for r in rows][:2] == ["plan", "retrieve"]
    assert "Route: complex" in text and "Sub-question: meal allowance" in text
    assert "hybrid_search" in text and "policy_2023.txt" in text and "policy_2024.txt" in text
    assert "Documents searched (named in the question)" in text and "Claims confirmed: 2 of 2" in text


def test_trace_never_shows_the_planner_rationale():
    view = _agent_view(COMPARE, TWO, ACCEPT)
    assert "PRIVATE REASONING TEXT" not in rv.trace_html(view)


def test_metrics_and_view_are_json_serialisable():
    import json

    view = _agent_view(COMPARE, TWO, ACCEPT)
    json.dumps(view)
    m = rv.metrics_html(view)
    assert "model calls" in m and "retrieval calls" in m and "prompt tokens" in m


# --- providers ------------------------------------------------------------------------

def test_prism_is_offered_only_when_configured(monkeypatch):
    monkeypatch.delenv("PRISM_API_KEY", raising=False)
    assert providers.available_backends() == ["ollama"]
    monkeypatch.setenv("PRISM_API_KEY", "test-key")
    assert providers.available_backends() == ["ollama", "prism"]


def test_provider_notices_describe_where_text_goes():
    local = providers.notice("ollama", "llama3.2")
    assert "on this machine" in local["lines"][0]
    prism = " ".join(providers.notice("prism", "llama3.2")["lines"])
    assert "fail closed" in prism and "cache" in prism and "guardrails" in prism
    assert "test-key" not in providers.notice_html("prism", "llama3.2")


# --- upload workspaces ---------------------------------------------------------------------

def test_each_session_gets_its_own_workspace(tmp_path):
    a, b = uploads.create(tmp_path), uploads.create(tmp_path)
    assert a.path != b.path and os.path.isdir(a.docs_dir) and a.index_dir != b.index_dir


def test_workspace_removal_refuses_paths_outside_the_session_root(tmp_path):
    ws = uploads.create(tmp_path)
    uploads.remove(ws, tmp_path)
    assert not ws.path.exists()
    with pytest.raises(ValueError):
        uploads.remove(uploads.Workspace(tmp_path), tmp_path)


def test_sweep_removes_only_idle_session_workspaces(tmp_path):
    old, fresh = uploads.create(tmp_path), uploads.create(tmp_path)
    other = tmp_path / "keep-me"
    other.mkdir()
    past = time.time() - 10_000
    os.utime(old.path, (past, past))
    os.utime(other, (past, past))
    assert uploads.sweep(tmp_path, retention_s=3600) == 1
    assert not old.path.exists() and fresh.path.exists() and other.exists()
    uploads.touch(fresh)


def test_renderer_wording_matches_the_agent_and_needs_no_heavy_imports():
    import subprocess
    import sys

    from agent.sources import METHOD_TEXT

    assert rv.METHOD_TEXT == METHOD_TEXT
    code = ("import sys, ui.render; heavy = [m for m in ('agent', 'langgraph', 'torch', 'faiss', 'ingest', "
            "'retriever', 'llm', 'sentence_transformers') if m in sys.modules]; print(heavy)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"
