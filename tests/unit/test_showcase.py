"""Public showcase: read-only by construction, recorded runs with provenance, no inference stack."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]
RUNS = json.loads((ROOT / "showcase" / "data" / "recorded_runs.json").read_text())
CURATION = json.loads((ROOT / "showcase" / "curation.json").read_text())
FORBIDDEN_MODULES = ("ingest", "retriever", "agent", "llm", "tabular", "evals", "langgraph", "langchain_core",
                     "torch", "faiss", "sentence_transformers", "sklearn", "pdfplumber", "httpx")


@pytest.fixture
def showcase():
    at = AppTest.from_file(str(ROOT / "showcase" / "streamlit_app.py"), default_timeout=30)
    at.run()
    assert not at.exception
    return at


def test_showcase_offers_no_text_input_or_upload(showcase):
    assert not showcase.chat_input and not showcase.text_input and not showcase.text_area
    assert not showcase.get("file_uploader")
    assert [t.label for t in showcase.tabs] == ["Compare answers", "Source documents", "Methodology & evaluation"]


def test_showcase_never_imports_ingestion_retrieval_inference_or_tabular_code():
    code = ("import sys; from streamlit.testing.v1 import AppTest; "
            f"at = AppTest.from_file({str(ROOT / 'showcase' / 'streamlit_app.py')!r}, default_timeout=30); at.run(); "
            "assert not at.exception, at.exception; "
            f"print([m for m in {FORBIDDEN_MODULES!r} if m in sys.modules])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=ROOT)
    assert out.stdout.strip().splitlines()[-1] == "[]"


def test_showcase_source_has_no_imports_of_the_live_stack():
    tree = ast.parse((ROOT / "ui" / "showcase.py").read_text())
    imported = {(n.module or "") for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    imported |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not {m.split(".")[0] for m in imported} & (set(FORBIDDEN_MODULES) | {"pickle", "joblib"})


def test_every_curated_question_shows_both_recorded_runs_with_provenance(showcase):
    for q in CURATION["questions"]:
        showcase.radio(key="case").set_value(q["case_id"]).run()
        assert not showcase.exception
        md = "\n".join(m.value for m in showcase.markdown)
        assert "Standard RAG — recorded" in md and "Agentic RAG — recorded" in md
        assert "Reviewed reference answer" in md
        assert sum("Provenance of this recorded run" in e.label for e in showcase.expander) == 2


def test_recorded_runs_are_genuine_recordings_with_complete_provenance():
    prov = RUNS["provenance"]
    assert prov["git_commit"] and "+uncommitted" not in prov["git_commit"]
    assert prov["model_digest"] and prov["ollama_version"] and prov["seed"] == 1001
    assert {"embed_model", "reranking", "top_k", "chunk_size", "chunk_overlap"} <= set(prov["retrieval"])
    assert [q["case_id"] for q in RUNS["questions"]] == [q["case_id"] for q in CURATION["questions"]]
    for q in RUNS["questions"]:
        for mode in ("standard", "agentic"):
            v = q["runs"][mode]
            assert v["mode"] == mode and v["recorded"]["started"] <= v["recorded"]["finished"]
            assert v["metrics"]["llm_calls"] >= 1
            if mode == "agentic":
                assert v["trace"] and v["trace"][0]["node"] == "plan" and v["status"]


def test_curated_set_is_balanced_between_successes_and_limitations():
    observations = " ".join(q["observation"] for q in CURATION["questions"])
    assert "Known limitation" in observations
    categories = {q["category"] for q in RUNS["questions"]}
    assert {"unanswerable", "false_premise", "multi_document", "partially_answerable"} <= categories


def test_documents_tab_lists_the_curated_corpus_only(showcase):
    corpus = json.loads((ROOT / "showcase" / "data" / "corpus.json").read_text())
    names = [d["source"] for d in corpus["documents"]]
    assert showcase.selectbox(key="doc").options == names
    manifest = json.loads((ROOT / "evals" / "corpus" / "MANIFEST.json").read_text())["files"]
    assert {d["sha256"] for d in corpus["documents"]} <= set(manifest.values())


def test_recorded_answers_are_rendered_as_escaped_text():
    from ui import render as rv

    out = rv.answer_html("See ![x](https://evil.example/?q=secret) <img src=x onerror=alert(1)>\n> note")
    assert "<img" not in out and "&lt;img" in out and 'class="rb-answer-note"' in out


def test_concurrent_sessions_do_not_share_mutable_state():
    a = AppTest.from_file(str(ROOT / "showcase" / "streamlit_app.py"), default_timeout=30)
    b = AppTest.from_file(str(ROOT / "showcase" / "streamlit_app.py"), default_timeout=30)
    a.run()
    b.run()
    a.radio(key="case").set_value("V02").run()
    assert a.session_state["case"] == "V02" and b.session_state["case"] == CURATION["questions"][0]["case_id"]
    md_b = "\n".join(m.value for m in b.markdown)
    assert "hotel approval" not in md_b.lower()


def test_cached_recordings_are_copies_so_one_session_cannot_alter_another():
    from ui import showcase

    runs, _, _ = showcase.load()
    runs["questions"][0]["runs"]["standard"]["answer"] = "tampered"
    again, _, _ = showcase.load()
    assert again["questions"][0]["runs"]["standard"]["answer"] != "tampered"


def test_recorded_disclosure_is_on_the_main_page_not_only_the_sidebar(showcase):
    main = "\n".join(m.value for m in showcase.main.markdown)
    assert "Recorded runs — no live inference" in main


def test_theme_keeps_the_sidebar_reopen_control_visible():
    css = (ROOT / "ui" / "theme.css").read_text()
    assert '[data-testid="stExpandSidebarButton"]' in css and "visibility: visible" in css
