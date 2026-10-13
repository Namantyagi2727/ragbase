"""Shared test fixtures and helpers."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def make_pdf(path: Path, pages: list[str]) -> Path:
    """Write a minimal single-font PDF with one line of text per page.

    Avoids a PDF-generation dependency; pdfplumber reads the text layer.
    """
    n = len(pages)
    page_ids = [4 + 2 * i for i in range(n)]
    objs = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{' '.join(f'{p} 0 R' for p in page_ids)}] /Count {n} >>",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for i, text in enumerate(pages):
        safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 72 720 Td ({safe}) Tj ET"
        objs.append(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {page_ids[i] + 1} 0 R >>"
        )
        objs.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")

    out = b"%PDF-1.4\n"
    offsets = []
    for i, obj in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{obj}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    ).encode()
    path.write_bytes(out)
    return path


@pytest.fixture
def sample_corpus(tmp_path: Path) -> Path:
    """A tiny mixed-format corpus covering every text loader."""
    from docx import Document as DocxDocument

    d = tmp_path / "docs"
    d.mkdir()
    (d / "policy.txt").write_text(
        "Travel Policy. Employees may claim a daily meal allowance of 45 dollars.\n\n"
        "Hotel bookings above 200 dollars per night require manager approval.",
        encoding="utf-8",
    )
    make_pdf(d / "handbook.pdf", [
        "Remote work is permitted two days per week.",
        "Annual leave accrues at 1.5 days per month.",
    ])
    doc = DocxDocument()
    doc.add_paragraph("Security guideline: laptops must use full disk encryption.")
    doc.save(d / "security.docx")
    (d / "sales.csv").write_text(
        "region,units,price\nEU,10,2.5\nUS,20,3.0\nEU,5,4.0\n", encoding="utf-8"
    )
    return d
