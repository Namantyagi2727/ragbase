"""Hybrid retrieval behaviour, tested without embedding models.

FAISS is replaced by a stub returning a fixed ranking, so the tests isolate
the fusion logic from embedding quality.
"""

import pytest
from langchain_core.documents import Document
from sklearn.feature_extraction.text import TfidfVectorizer

import retriever as r
from ingest import assign_chunk_ids

RAW = [
    ("Meal allowance is 45 dollars per day.", "travel.txt", "txt"),
    ("Hotel rooms above 200 dollars need approval.", "travel.txt", "txt"),
    ("Remote work is allowed two days a week.", "hr.pdf", "p.1"),
    ("Annual leave accrues monthly.", "hr.pdf", "p.2"),
    ("Laptops require disk encryption.", "sec.docx", "docx"),
    ("The zygomorphic widget clause applies to suppliers.", "contract.txt", "txt"),
]
IDS = assign_chunk_ids((s, loc, t) for t, s, loc in RAW)
CHUNKS = [
    {"chunk_id": cid, "text": t, "source": s, "loc": loc}
    for cid, (t, s, loc) in zip(IDS, RAW, strict=True)
]


def _doc(i: int) -> Document:
    c = CHUNKS[i]
    return Document(page_content=c["text"], metadata={k: c[k] for k in ("chunk_id", "source", "loc")})


class StubVectorStore:
    """Returns chunks in a fixed order regardless of the query; honours callable filters."""

    def __init__(self, order):
        self.order = order
        self.calls = []

    def similarity_search(self, query, k=4, filter=None, **kwargs):
        self.calls.append({"k": k, "filter": filter})
        docs = [_doc(i) for i in self.order]
        if filter is not None:
            docs = [d for d in docs if filter(d.metadata)]
        return docs[:k]


def _hybrid(order, top_k=3, **kw):
    texts = [c["text"] for c in CHUNKS]
    vec = TfidfVectorizer().fit(texts)
    return r.HybridRetriever(
        vectorstore=StubVectorStore(order),
        vectorizer=vec,
        tfidf_matrix=vec.transform(texts),
        chunks_meta=CHUNKS,
        top_k=top_k,
        **kw,
    )


# --- behaviour preserved from the original implementation -----------------

def test_results_capped_at_top_k():
    assert len(_hybrid([0, 1, 2, 3, 4], top_k=3).invoke("meal allowance")) == 3


def test_results_preserve_source_and_location():
    for d in _hybrid([2, 3, 4], top_k=3).invoke("remote work"):
        assert d.metadata["source"] in {"travel.txt", "hr.pdf", "sec.docx", "contract.txt"}
        assert d.metadata["loc"]


def test_no_duplicate_chunks_when_both_retrievers_agree():
    docs = _hybrid([0, 1, 2], top_k=3).invoke("meal allowance dollars")
    ids = [d.metadata["chunk_id"] for d in docs]
    assert len(ids) == len(set(ids))


# --- fusion fix -------------------------------------------------------------

def test_keyword_only_match_reaches_results():
    """A chunk FAISS ranks last but TF-IDF ranks first must survive fusion.

    Original bug: FAISS filled all top_k slots before TF-IDF was considered.
    """
    docs = _hybrid([0, 1, 2, 3, 4], top_k=3).invoke("zygomorphic widget clause")
    assert any("zygomorphic" in d.page_content for d in docs)


def test_results_carry_chunk_ids_and_rank_provenance():
    docs = _hybrid([5, 0, 1], top_k=2).invoke("zygomorphic widget clause")
    top = docs[0]
    assert top.metadata["chunk_id"] == IDS[5]
    assert top.metadata["semantic_rank"] == 1
    assert top.metadata["keyword_rank"] == 1


def test_candidate_pool_is_larger_than_top_k():
    h = _hybrid([0, 1, 2, 3, 4], top_k=2, candidate_pool=5)
    h.invoke("anything")
    assert h.vectorstore.calls[0]["k"] == 5


def test_source_filter_applies_to_both_retrievers():
    docs = _hybrid([0, 1, 2, 3, 4, 5], top_k=5).search("dollars approval", k=5, sources=["hr.pdf"])
    assert docs and {d.metadata["source"] for d in docs} == {"hr.pdf"}


def test_sources_lists_indexed_files():
    assert _hybrid([0]).sources() == ["contract.txt", "hr.pdf", "sec.docx", "travel.txt"]


def test_legacy_chunks_without_ids_get_deterministic_ids():
    legacy = [{k: c[k] for k in ("text", "source", "loc")} for c in CHUNKS]
    texts = [c["text"] for c in legacy]
    vec = TfidfVectorizer().fit(texts)
    h = r.HybridRetriever(
        vectorstore=StubVectorStore([0, 1]), vectorizer=vec,
        tfidf_matrix=vec.transform(texts), chunks_meta=legacy, top_k=2,
    )
    assert [c["chunk_id"] for c in h.chunks_meta] == IDS


# --- reciprocal rank fusion (pure function) --------------------------------

def test_rrf_rewards_agreement_between_retrievers():
    fused = r.reciprocal_rank_fusion([["a", "b", "c"], ["c", "d"]], k=60)
    assert fused[0][0] == "c"  # appears in both lists


def test_rrf_scores_match_formula():
    fused = dict(r.reciprocal_rank_fusion([["a", "b"], ["b"]], k=10))
    assert fused["a"] == pytest.approx(1 / 11)
    assert fused["b"] == pytest.approx(1 / 12 + 1 / 11)


def test_rrf_ties_are_deterministic():
    fused = r.reciprocal_rank_fusion([["x"], ["y"]], k=60)
    assert [i for i, _ in fused] == ["x", "y"]


def test_rrf_handles_empty_lists():
    assert r.reciprocal_rank_fusion([[], []]) == []


# --- reranker ----------------------------------------------------------------

class ScoreByLength:
    def predict(self, pairs):
        return [len(text) for _, text in pairs]


def test_reranker_scores_a_candidate_pool_larger_than_top_k(monkeypatch):
    monkeypatch.setattr(r, "_get_cross_encoder", lambda name: ScoreByLength())
    base = _hybrid([0, 1, 2, 3, 4, 5], top_k=2, candidate_pool=6)
    rr = r.CrossEncoderReranker(base_retriever=base, top_k=2, candidate_k=6)
    docs = rr.invoke("dollars")
    assert len(docs) == 2
    assert all("rerank_score" in d.metadata for d in docs)
    assert docs[0].metadata["rerank_score"] >= docs[1].metadata["rerank_score"]


def test_reranker_falls_back_to_fused_order_on_failure(monkeypatch):
    def boom(name):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(r, "_get_cross_encoder", boom)
    base = _hybrid([0, 1, 2, 3], top_k=2)
    rr = r.CrossEncoderReranker(base_retriever=base, top_k=2)
    assert len(rr.invoke("dollars")) == 2
