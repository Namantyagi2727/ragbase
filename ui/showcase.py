"""
ui/showcase.py — Public showcase: recorded runs over the curated synthetic corpus.

Read-only by construction:
  - imports nothing from ingestion, retrieval models, the agent, the LLM
    clients or the tabular module, so it cannot index, retrieve or generate;
  - offers no text input or upload widget: visitors pick a curated question
    and inspect recorded answers, citations, execution summaries and the
    source documents;
  - reads only the committed JSON files in showcase/ (no pickle).
Every answer shown was produced by scripts/record_showcase.py running the real
local pipeline; provenance is shown with each run.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import streamlit as st

from ui import methodology
from ui import render as rv

DATA = Path(__file__).resolve().parents[1] / "showcase"
REPO_URL = "https://github.com/Namantyagi2727/ragbase"


@st.cache_data(show_spinner=False)
def load() -> tuple[dict, dict, dict]:
    runs = json.loads((DATA / "data" / "recorded_runs.json").read_text())
    corpus = json.loads((DATA / "data" / "corpus.json").read_text())
    curation = json.loads((DATA / "curation.json").read_text())
    return runs, corpus, curation


def _e(x) -> str:
    return html.escape(str(x), quote=True)


def _sidebar(curation: dict) -> str:
    with st.sidebar:
        st.markdown("""
        <div class="rb-brand">
            <div class="rb-brand-logo-wrap">
                <span class="rb-brand-hex">⬡</span>
                <span class="rb-brand-logo">RAGBASE</span>
            </div>
            <div class="rb-brand-sub">// enterprise document intelligence</div>
        </div>
        """, unsafe_allow_html=True)
        st.divider()
        st.markdown('<div class="rb-label">Curated questions</div>', unsafe_allow_html=True)
        titles = {q["case_id"]: f'{q["title"]}' for q in curation["questions"]}
        choice = st.radio("Curated questions", list(titles), format_func=titles.get, key="case",
                          label_visibility="collapsed")
        st.divider()
        st.markdown(
            '<div class="rb-notice"><b>Recorded runs — no live inference</b><br>'
            "Every answer here was generated earlier by RAGBase running locally with Ollama. "
            "This page cannot run models, accept uploads or answer new questions.</div>",
            unsafe_allow_html=True)
        st.markdown(f'<div class="rb-notice" style="margin-top:0.6rem"><b>Run it yourself</b><br>'
                    f'Clone <a href="{REPO_URL}">the repository</a> for live Standard and Agentic RAG on '
                    "your own documents, fully local.</div>", unsafe_allow_html=True)
    return choice


def _provenance_html(prov: dict, view: dict) -> str:
    rec = view.get("recorded") or {}
    rows = [("Recorded", f'{rec.get("started")} → {rec.get("finished")}'),
            ("Commit", prov["git_commit"]), ("Model", f'{prov["model"]} (digest {prov["model_digest"]}) via Ollama '
                                                      f'{prov["ollama_version"]}'),
            ("Seed / temperature", f'{prov["seed"]} / {prov["temperature"]}'),
            ("Retrieval", ", ".join(f"{k} {v}" for k, v in prov["retrieval"].items())),
            ("Agent limits", ", ".join(f"{k} {v}" for k, v in prov["agent"].items())),
            ("Corpus", f'v{prov["corpus_version"]} ({prov["corpus_checksum"]})')]
    body = "<br>".join(f"<b>{_e(k)}:</b> {_e(v)}" for k, v in rows)
    return f'<div class="rb-notice">{body}</div>'


def _run_column(label: str, view: dict, prov: dict) -> None:
    st.markdown(f'<div class="rb-label">{_e(label)}</div>', unsafe_allow_html=True)
    st.markdown(rv.answer_html(view["answer"]), unsafe_allow_html=True)
    rv.render_view(view)
    with st.expander("Provenance of this recorded run"):
        st.markdown(_provenance_html(prov, view), unsafe_allow_html=True)


def _compare(case_id: str, runs: dict, curation: dict) -> None:
    item = next(q for q in runs["questions"] if q["case_id"] == case_id)
    notes = next(q for q in curation["questions"] if q["case_id"] == case_id)
    st.markdown(f'<div class="rb-label">{_e(item["category"].replace("_", " "))} · {_e(item["case_id"])}</div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="rb-question">{_e(item["question"])}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="rb-notice"><b>Reviewed reference answer:</b> {_e(item["reference_answer"])}<br>'
        f'<b>Why this question:</b> {_e(notes["why"])}<br>'
        f'<b>What happened:</b> {_e(notes.get("observation", ""))}</div>', unsafe_allow_html=True)
    st.write("")
    left, right = st.columns(2, gap="large")
    with left:
        _run_column("Standard RAG — recorded", item["runs"]["standard"], runs["provenance"])
    with right:
        _run_column("Agentic RAG — recorded", item["runs"]["agentic"], runs["provenance"])


def _documents(corpus: dict) -> None:
    st.caption("The curated synthetic corpus, as RAGBase's ingestion extracted it (two-column PDF pages "
               "are read column by column). Synthetic documents written for the benchmark; no real company.")
    names = [d["source"] for d in corpus["documents"]]
    name = st.selectbox("Document", names, key="doc")
    doc = next(d for d in corpus["documents"] if d["source"] == name)
    st.markdown(f'<div class="rb-notice"><b>{_e(name)}</b> · corpus v{_e(corpus["version"])} · sha256 '
                f'{_e(doc["sha256"][:16])}…</div>', unsafe_allow_html=True)
    for sec in doc["sections"]:
        st.markdown(rv.passage_html({"eid": "", "source": name, "loc": sec["loc"], "text": sec["text"]},
                                    cited=False, full=True), unsafe_allow_html=True)


def main() -> None:
    runs, corpus, curation = load()
    case_id = _sidebar(curation)
    st.markdown("""
    <div class="rb-title-wrap">
        <div class="rb-title-row">
            <span class="rb-title">RAGBASE</span>
            <span class="rb-badge">Showcase · recorded runs</span>
        </div>
    </div>
    <div class="rb-subtitle">
        <span>standard</span> vs <span>agentic</span> RAG
        &nbsp;·&nbsp; citations, verification and execution traces
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="rb-notice" style="margin-top:0.6rem"><b>Recorded runs — no live inference.</b> '
                "Every answer was generated earlier by RAGBase running locally; provenance is shown with each run. "
                "This page cannot answer new questions or accept documents. Choose one of ten curated questions in the "
                "sidebar (tap » on a phone).</div>", unsafe_allow_html=True)
    st.divider()
    tab_compare, tab_docs, tab_method = st.tabs(["Compare answers", "Source documents", "Methodology & evaluation"])
    with tab_compare:
        _compare(case_id, runs, curation)
    with tab_docs:
        _documents(corpus)
    with tab_method:
        methodology.render()
