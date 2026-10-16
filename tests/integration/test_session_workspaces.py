"""Session-isolated upload workspaces with the real embedding model (no LLM).

Two sessions upload different documents; each index only contains its own
session's documents, and removing one workspace leaves the other intact.
"""

import pytest

import ingest
import retriever as r
from ui import uploads

pytestmark = pytest.mark.integration


def _session(root, name, text):
    ws = uploads.create(root)
    (ws.path / "docs" / name).write_text(text)
    report = ingest.ingest_documents(ingest_dir=ws.docs_dir, index_dir=ws.index_dir)
    assert report.chunks > 0
    return ws


def test_sessions_never_see_each_others_documents(tmp_path):
    a = _session(tmp_path, "alpha.txt", "Project Alpha's launch date is 3 March. The budget is 40,000 dollars.")
    b = _session(tmp_path, "beta.txt", "Project Beta's launch date is 9 July. The budget is 75,000 dollars.")
    ra = r.get_retriever(index_dir=a.index_dir, top_k=3, reranking=False)
    rb = r.get_retriever(index_dir=b.index_dir, top_k=3, reranking=False)
    assert {d.metadata["source"] for d in ra.invoke("launch date budget")} == {"alpha.txt"}
    assert {d.metadata["source"] for d in rb.invoke("launch date budget")} == {"beta.txt"}
    uploads.remove(a, tmp_path)
    assert not a.path.exists() and b.path.exists()
    assert rb.invoke("launch date")[0].metadata["source"] == "beta.txt"
