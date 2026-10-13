"""End-to-end retrieval with the real embedding and reranker models.

Marked `integration`: downloads all-MiniLM-L6-v2 and ms-marco-MiniLM-L-6-v2
from Hugging Face on first run (cached afterwards). No LLM required.
"""

import pytest

import ingest
import retriever as r

pytestmark = pytest.mark.integration


def _build_real(sample_corpus, tmp_path):
    idx = str(tmp_path / "idx")
    report = ingest.ingest_documents(ingest_dir=str(sample_corpus), index_dir=idx)
    assert report.chunks > 0
    return idx


@pytest.mark.parametrize("reranking", [False, True])
def test_semantic_question_finds_the_right_document(sample_corpus, tmp_path, reranking):
    idx = _build_real(sample_corpus, tmp_path)
    ret = r.get_retriever(index_dir=idx, top_k=2, reranking=reranking)
    docs = ret.invoke("How much can I spend on food when travelling?")
    assert docs[0].metadata["source"] == "policy.txt"


def test_source_filter_with_real_faiss(sample_corpus, tmp_path):
    idx = _build_real(sample_corpus, tmp_path)
    ret = r.get_retriever(index_dir=idx, top_k=3, reranking=False)
    docs = ret.search("encryption and leave policy", k=3, sources=["handbook.pdf"])
    assert docs and {d.metadata["source"] for d in docs} == {"handbook.pdf"}


def test_index_manifest_model_overrides_requested_model(sample_corpus, tmp_path):
    idx = _build_real(sample_corpus, tmp_path)
    assert r.resolve_embed_model(idx, "BAAI/bge-small-en-v1.5") == ingest.config.EMBED_MODEL
