"""Ingestion: every supported text format loads with source + location metadata."""

import pandas as pd

import ingest


def test_txt_loads_whole_file(sample_corpus):
    entries = ingest.load_doc(str(sample_corpus / "policy.txt"))
    assert len(entries) == 1
    text, source, loc = entries[0]
    assert "meal allowance of 45 dollars" in text
    assert (source, loc) == ("policy.txt", "txt")


def test_pdf_loads_one_entry_per_page(sample_corpus):
    entries = ingest.load_doc(str(sample_corpus / "handbook.pdf"))
    assert [loc for _, _, loc in entries] == ["p.1", "p.2"]
    assert "two days per week" in entries[0][0]
    assert "1.5 days per month" in entries[1][0]


def test_docx_loads_paragraphs(sample_corpus):
    entries = ingest.load_doc(str(sample_corpus / "security.docx"))
    assert entries == [
        ("Security guideline: laptops must use full disk encryption.", "security.docx", "docx")
    ]


def test_csv_produces_metadata_and_row_blocks(sample_corpus):
    entries = ingest.load_doc(str(sample_corpus / "sales.csv"))
    locs = [loc for _, _, loc in entries]
    assert locs[0] == "metadata"
    assert locs[1] == "rows.1-3"
    meta_text = entries[0][0]
    assert "Column names: ['region', 'units', 'price']" in meta_text
    assert "Rows: 3, Columns: 3" in meta_text


def test_tabular_row_blocks_respect_row_chunk_size(monkeypatch):
    monkeypatch.setattr(ingest.config, "ROW_CHUNK_SIZE", 2)
    df = pd.DataFrame({"a": range(5)})
    entries = ingest._ingest_tabular(df, "t.csv")
    assert [loc for _, _, loc in entries] == ["metadata", "rows.1-2", "rows.3-4", "rows.5-5"]


def test_unsupported_extension_is_skipped(tmp_path):
    p = tmp_path / "notes.md"
    p.write_text("# hi")
    assert ingest.load_doc(str(p)) == []
