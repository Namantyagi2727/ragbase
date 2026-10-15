"""
evals/pdfgen.py — Minimal deterministic PDF writer for synthetic test documents.

Places each line of text at explicit coordinates with Helvetica, so tests and
the evaluation corpus can build single-column, two-column and mixed layouts
without a PDF library. Output contains no timestamps or random IDs, so the same
input produces the same bytes.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from pathlib import Path

PAGE_W, PAGE_H = 612, 792


@dataclass(frozen=True)
class TextLine:
    x: float
    y: float          # baseline, measured from the bottom of the page
    text: str
    size: float = 10


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def write_pdf(path: Path, pages: list[list[TextLine]]) -> Path:
    """Write pages of positioned text lines (ASCII) to `path`."""
    n = len(pages)
    page_ids = [4 + 2 * i for i in range(n)]
    objs = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{' '.join(f'{p} 0 R' for p in page_ids)}] /Count {n} >>",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for i, lines in enumerate(pages):
        ops = "\n".join(
            f"BT /F1 {ln.size:g} Tf {ln.x:g} {ln.y:g} Td ({_escape(ln.text)}) Tj ET" for ln in lines
        )
        objs.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_W} {PAGE_H}] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {page_ids[i] + 1} 0 R >>"
        )
        objs.append(f"<< /Length {len(ops.encode('latin-1'))} >>\nstream\n{ops}\nendstream")

    out = b"%PDF-1.4\n"
    offsets = []
    for i, obj in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{obj}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    Path(path).write_bytes(out)
    return Path(path)


def flow(text: str, x: float, top: float, width_chars: int, leading: float = 13, size: float = 10) -> list[TextLine]:
    """Wrap paragraphs (blank-line separated) into positioned lines starting at `top`."""
    lines: list[TextLine] = []
    y = top
    for para in text.strip().split("\n\n"):
        for row in textwrap.wrap(" ".join(para.split()), width_chars):
            lines.append(TextLine(x, y, row, size))
            y -= leading
        y -= leading * 0.6
    return lines
