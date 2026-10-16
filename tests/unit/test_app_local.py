"""Local Streamlit app (AppTest): mode selection, UI state, rendering, providers. Scripted models only."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from streamlit.testing.v1 import AppTest

import config
from tests.unit.agent_helpers import MemoryRetriever, RoleModel

COMPARE = {"route": "complex", "compare_documents": True, "subqueries": [{"query": "meal allowance"}]}
TWO = "In 2023 the meal allowance was 45 dollars [1]. In 2024 it rose to 60 dollars [2]."
ACCEPT = {"verdicts": [{"claim_number": 1, "status": "supported", "evidence_ids": ["1"]},
                       {"claim_number": 2, "status": "supported", "evidence_ids": ["2"]}], "recommendation": "accept"}


class MemoryLCRetriever(BaseRetriever):
    """A LangChain retriever (for Standard RAG) that also offers search()/sources() (for the agent)."""

    mem: MemoryRetriever

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        return self.mem.search(query, k=3)

    def search(self, query, k=None, sources=None):
        return self.mem.search(query, k=k, sources=sources)

    def sources(self):
        return self.mem.sources()


@pytest.fixture
def app(monkeypatch, tmp_path):
    from ui import local_app

    model = RoleModel(planner=[COMPARE], synth=[TWO], verifier=[ACCEPT],
                      standard=["The 2024 meal allowance is 60 dollars [2]."])
    retriever = MemoryLCRetriever(mem=MemoryRetriever())
    monkeypatch.setattr(config, "INDEX_DIR", str(tmp_path / "no-library"))   # no shared library in tests
    monkeypatch.setattr(local_app, "_retriever", lambda: retriever)
    monkeypatch.setattr(local_app, "get_chat_model", lambda **kw: model)
    monkeypatch.setattr(local_app.uploads, "ROOT", tmp_path / "sessions")
    monkeypatch.delenv("PRISM_API_KEY", raising=False)
    at = AppTest.from_file(str(Path(__file__).resolve().parents[2] / "app.py"), default_timeout=30)
    at.session_state["index_built"] = True
    at.session_state["index_source"] = "session"
    at.session_state["docs_dir"] = str(tmp_path)
    at.model = model
    return at


def _markdown(at) -> str:
    return "\n".join(m.value for m in at.markdown)


def test_local_app_starts_with_standard_mode_and_both_tabs(app):
    app.run()
    assert not app.exception
    assert app.radio(key="mode_radio").value == "Standard RAG"
    assert [t.label for t in app.tabs] == ["Chat", "Methodology & evaluation"]
    assert len(app.chat_input) == 1


def test_standard_answer_renders_citations_and_unverified_status(app):
    app.run()
    app.chat_input[0].set_value("What is the 2024 meal allowance?").run()
    assert not app.exception and app.model.calls == ["standard"]
    md = _markdown(app)
    assert "Not checked — Standard RAG does not verify claims" in md
    assert "model calls" in md
    assert any("cited passage" in e.label for e in app.expander)


def test_agentic_mode_runs_the_agent_and_shows_the_execution_summary(app):
    app.run()
    app.radio(key="mode_radio").set_value("Agentic RAG").run()
    app.chat_input[0].set_value("Compare the meal allowance in 2023 and 2024.").run()
    assert not app.exception
    assert app.model.calls == ["planner", "synth", "verifier"]
    labels = [e.label for e in app.expander]
    assert any(label.startswith("⬡  Agent execution — route: complex") for label in labels)
    md = _markdown(app)
    assert "hybrid_search" in md and "Documents searched (named in the question)" in md and "Claims confirmed: 2 of 2" in md
    assert "not a guarantee of correctness" in md


def test_mode_choice_persists_across_reruns(app):
    app.run()
    app.radio(key="mode_radio").set_value("Agentic RAG").run()
    app.run()
    assert app.session_state["mode"] == "Agentic RAG"


def test_without_an_index_the_app_asks_for_documents(app):
    app.session_state["index_built"] = False
    app.run()
    app.chat_input[0].set_value("anything").run()
    assert not app.exception and app.model.calls == []
    assert any("build the index first" in w.value for w in app.warning)


def test_only_ollama_is_offered_when_prism_is_not_configured(app):
    app.run()
    assert not [s for s in app.selectbox if s.label == "Backend"]
    assert "Direct local inference · Ollama" in _markdown(app)


def test_prism_selector_appears_when_configured(app, monkeypatch):
    monkeypatch.setenv("PRISM_API_KEY", "test-only")
    app.run()
    backend = [s for s in app.selectbox if s.label == "Backend"]
    assert backend and backend[0].options == ["Ollama (direct, local)", "Prism gateway"]
    backend[0].set_value("prism").run()
    assert "Prism gateway" in _markdown(app) and "test-only" not in _markdown(app)


def test_methodology_section_has_no_numerical_results(app):
    import re

    from ui import methodology

    app.run()
    assert "Standard RAG outperformed Agentic RAG" in methodology.TEXT
    assert not re.search(r"\d+(\.\d+)?\s*%|\b0\.\d{2}\b", methodology.TEXT)


def test_overlong_questions_are_refused_before_any_model_call(app, monkeypatch):
    monkeypatch.setattr(config, "MAX_QUESTION_CHARS", 50)
    app.run()
    app.chat_input[0].set_value("x" * 51).run()
    assert not app.exception and app.model.calls == []
    assert any("limited to 50 characters" in e.value for e in app.error)


class _Upload:
    def __init__(self, name, data=b"Some policy text for indexing.", size=None):
        self.name, self._data, self.size = name, data, len(data) if size is None else size

    def getbuffer(self):
        return memoryview(self._data)


class _Report:
    def __init__(self, chunks):
        self.chunks, self.files_indexed, self.skipped, self.warnings = chunks, ["ok.txt"], [], []


def test_session_uploads_enforce_size_and_name_limits_and_replace_the_previous_workspace(monkeypatch, tmp_path):
    import streamlit as st

    from ui import local_app, uploads

    monkeypatch.setattr(local_app.uploads, "ROOT", tmp_path)
    written = {}

    def fake_ingest(ingest_dir, **kw):
        written["files"] = sorted(p.name for p in __import__("pathlib").Path(ingest_dir).iterdir())
        return _Report(chunks=3)

    monkeypatch.setattr(local_app.ingest, "ingest_documents", fake_ingest)
    monkeypatch.setattr(local_app, "_activate", lambda *a: None)
    monkeypatch.setattr(local_app._load_retriever, "clear", lambda: None)
    monkeypatch.setattr(config, "MAX_UPLOAD_MB", 1)
    st.session_state.clear()
    local_app._init_state()
    previous = uploads.create(tmp_path)
    st.session_state.workspace = previous
    local_app._build_session_index([_Upload("ok.txt"), _Upload("../../etc/passwd"),
                                    _Upload("big.pdf", size=2 * 1024 * 1024)])
    assert written["files"] == ["ok.txt"]                       # unsafe and oversized files never written
    assert not previous.path.exists()                           # previous session workspace removed
    assert st.session_state.workspace.path.parent == tmp_path


def test_a_failed_build_removes_its_new_workspace(monkeypatch, tmp_path):
    import streamlit as st

    from ui import local_app

    monkeypatch.setattr(local_app.uploads, "ROOT", tmp_path)
    monkeypatch.setattr(local_app.ingest, "ingest_documents", lambda **kw: _Report(chunks=0))
    st.session_state.clear()
    local_app._init_state()
    local_app._build_session_index([_Upload("empty.txt")])
    assert st.session_state.workspace is None and list(tmp_path.iterdir()) == []
