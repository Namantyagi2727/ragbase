"""Chunk identity, upload-name safety and ingestion robustness."""

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

import ingest
import retriever as r


def test_chunk_ids_are_deterministic():
    recs = [("a.txt", "txt", "hello"), ("b.txt", "txt", "world")]
    assert ingest.assign_chunk_ids(recs) == ingest.assign_chunk_ids(list(recs))
    assert len(ingest.assign_chunk_ids(recs)[0]) == 16


def test_chunk_ids_depend_on_source_location_and_text():
    a, b, c, d = ingest.assign_chunk_ids([
        ("a.txt", "txt", "x"), ("b.txt", "txt", "x"), ("a.txt", "p.2", "x"), ("a.txt", "txt", "y"),
    ])
    assert len({a, b, c, d}) == 4


def test_duplicate_chunks_get_ordered_suffixes():
    ids = ingest.assign_chunk_ids([("a", "p.1", "same")] * 3)
    assert ids[1] == f"{ids[0]}-2" and ids[2] == f"{ids[0]}-3"


@pytest.mark.parametrize("raw,expected", [
    ("report.pdf", "report.pdf"),
    ("../../etc/passwd.txt", "passwd.txt"),
    ("..\\..\\win\\evil.docx", "evil.docx"),
    ("my file (v2).csv", "my file (v2).csv"),
    ("<script>.txt", "_script_.txt"),
    (".hidden.txt", None),
    ("noextension", None),
    ("payload.exe", None),
    ("archive.tar.gz", None),
    ("", None),
])
def test_safe_filename(raw, expected):
    assert ingest.safe_filename(raw) == expected


def test_safe_filename_truncates_long_names():
    name = ingest.safe_filename("a" * 300 + ".pdf")
    assert len(name) <= 128 and name.endswith(".pdf")


def _build(docs_dir, index_dir):
    return ingest.ingest_documents(
        ingest_dir=str(docs_dir), index_dir=str(index_dir),
        embed_model="fake-test", embeddings=DeterministicFakeEmbedding(size=32),
    )


def test_ingest_writes_ids_and_manifest(sample_corpus, tmp_path):
    report = _build(sample_corpus, tmp_path / "idx")
    assert report.chunks > 0 and not report.skipped
    assert sorted(report.files_indexed) == ["handbook.pdf", "policy.txt", "sales.csv", "security.docx"]
    manifest = ingest.read_manifest(str(tmp_path / "idx"))
    assert manifest["embed_model"] == "fake-test"
    assert manifest["chunk_count"] == report.chunks
    assert manifest["format_version"] == ingest.INDEX_FORMAT_VERSION


def test_rebuild_produces_identical_chunk_ids(sample_corpus, tmp_path):
    _build(sample_corpus, tmp_path / "one")
    _build(sample_corpus, tmp_path / "two")
    ids_one = [c["chunk_id"] for c in r.load_tfidf(str(tmp_path / "one"))[2]]
    ids_two = [c["chunk_id"] for c in r.load_tfidf(str(tmp_path / "two"))[2]]
    assert ids_one == ids_two


def test_faiss_and_tfidf_agree_on_chunk_ids_after_reload(sample_corpus, tmp_path):
    idx = str(tmp_path / "idx")
    _build(sample_corpus, idx)
    vs = r.load_vectorstore(idx, embeddings=DeterministicFakeEmbedding(size=32))
    faiss_ids = {d.metadata["chunk_id"] for d in vs.docstore._dict.values()}
    meta_ids = {c["chunk_id"] for c in r.load_tfidf(idx)[2]}
    assert faiss_ids == meta_ids


def test_unreadable_file_is_skipped_not_fatal(sample_corpus, tmp_path):
    (sample_corpus / "broken.pdf").write_bytes(b"%PDF-1.4 this is not a real pdf")
    report = _build(sample_corpus, tmp_path / "idx")
    assert report.chunks > 0
    assert [fn for fn, _ in report.skipped] == ["broken.pdf"]


def test_oversized_file_is_skipped(sample_corpus, tmp_path, monkeypatch):
    monkeypatch.setattr(ingest.config, "MAX_UPLOAD_MB", 0)
    report = _build(sample_corpus, tmp_path / "idx")
    assert report.chunks == 0 and len(report.skipped) == 4


def test_pdf_page_limit_is_enforced_and_reported(tmp_path, monkeypatch):
    from tests.conftest import make_pdf

    monkeypatch.setattr(ingest.config, "MAX_PDF_PAGES", 1)
    pdf = make_pdf(tmp_path / "long.pdf", ["page one text", "page two text"])
    warnings = []
    entries = ingest.load_doc(str(pdf), warnings=warnings)
    assert [loc for _, _, loc in entries] == ["p.1"]
    assert warnings and "only the first 1" in warnings[0]


def test_main_keeps_backward_compatible_int_return(sample_corpus, tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "HuggingFaceEmbeddings", lambda model_name: DeterministicFakeEmbedding(size=16))
    n = ingest.main(ingest_dir=str(sample_corpus), index_dir=str(tmp_path / "idx"))
    assert isinstance(n, int) and n > 0


def test_legacy_index_without_ids_loads_with_consistent_ids(sample_corpus, tmp_path):
    """Indexes built before chunk IDs existed (no chunk_id, random docstore IDs, no manifest)."""
    import os
    import pickle

    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document

    idx = tmp_path / "legacy"
    _build(sample_corpus, idx)
    meta = r.load_tfidf(str(idx))[2]
    legacy_meta = [{k: c[k] for k in ("text", "source", "loc")} for c in meta]
    emb = DeterministicFakeEmbedding(size=32)
    FAISS.from_documents(
        [Document(page_content=c["text"], metadata={"source": c["source"], "loc": c["loc"]}) for c in legacy_meta],
        emb,
    ).save_local(str(idx))
    with open(idx / "chunks_meta.pkl", "wb") as f:
        pickle.dump(legacy_meta, f)
    os.remove(idx / ingest.MANIFEST_FILE)

    vs = r.load_vectorstore(str(idx), embeddings=emb)
    _, _, loaded_meta = r.load_tfidf(str(idx))
    assert [c["chunk_id"] for c in loaded_meta] == [c["chunk_id"] for c in meta]
    assert {d.metadata["chunk_id"] for d in vs.docstore._dict.values()} == {c["chunk_id"] for c in meta}


def test_batch_ingestion_has_no_file_cap_by_default(tmp_path):
    """Regression: an upload-only file cap once truncated large-corpus ingestion."""
    docs = tmp_path / "many"
    docs.mkdir()
    for i in range(ingest.config.MAX_UPLOAD_FILES + 5):
        (docs / f"d{i:03d}.txt").write_text(f"Document number {i} has unique content {i * 7}.")
    report = _build(docs, tmp_path / "idx")
    assert len(report.files_indexed) == ingest.config.MAX_UPLOAD_FILES + 5


def test_explicit_file_cap_is_reported(sample_corpus, tmp_path):
    report = ingest.ingest_documents(
        ingest_dir=str(sample_corpus), index_dir=str(tmp_path / "idx"),
        embed_model="fake-test", embeddings=DeterministicFakeEmbedding(size=32), max_files=2,
    )
    assert len(report.files_indexed) == 2 and "only the first 2" in report.warnings[0]
