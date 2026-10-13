"""Standard RAG against a real local Ollama model.

Marked `live`: skipped unless Ollama is reachable. Never runs in default CI.
Asserts structure (valid evidence citations or abstention), not exact wording.
"""

import httpx
import pytest

import config
import ingest
import retriever as r
from llm import get_chat_model

pytestmark = [pytest.mark.live, pytest.mark.integration]


def _ollama_up() -> bool:
    try:
        return httpx.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


@pytest.fixture(scope="module", autouse=True)
def _require_ollama():
    if not _ollama_up():
        pytest.skip("Ollama is not running")


@pytest.fixture
def chain(sample_corpus, tmp_path):
    idx = str(tmp_path / "idx")
    ingest.ingest_documents(ingest_dir=str(sample_corpus), index_dir=idx)
    return r.build_rag_chain(r.get_retriever(index_dir=idx, top_k=3), chat_model=get_chat_model())


def test_answer_cites_only_retrieved_evidence(chain):
    result = chain.invoke("What is the daily meal allowance?")
    report = result["citations"]
    assert not report.invalid_ids and not report.non_evidence_citations
    assert report.valid_ids, result["answer"]
    assert "45" in result["answer"]
    assert not result["llm"].is_mock


def test_unanswerable_question_abstains(chain):
    result = chain.invoke("What is the company's policy on bringing pets to the office?")
    assert result["citations"].abstained, result["answer"]


def test_tabular_quantity_question_is_summed():
    """Regression for the count-vs-sum planning error seen with llama3.2."""
    import pandas as pd

    import tabular as tb

    df = pd.DataFrame({"region": ["EU", "US", "EU", "APAC"], "units": [10, 20, 5, 7]})
    tables = [tb.Table("sales.csv", "sales.csv", None, df)]
    resp = tb.answer_tabular_question("How many units were sold in the EU?", tables, get_chat_model())
    assert resp.ok, resp.error
    assert resp.query["operation"] == "sum" and resp.query["column"] == "units"
    assert "15" in resp.answer
