"""Two-column PDF extraction: reading order, and no change where layout is not clearly two-column.

Regression for the driver's-manual failure: pdfplumber read two-column pages
line by line across both columns, interleaving unrelated sentences (e.g. the
over-21 and under-21 alcohol rules), which contributed to a verifier false positive.
"""

import pdfplumber

import ingest
from evals.pdfgen import PAGE_H, TextLine, flow, write_pdf

LEFT = (
    "Drivers aged 21 and over commit an offence if their blood alcohol concentration is 0.08 percent "
    "or higher. A first offence carries a licence revocation of at least six months and a fine."
)
RIGHT = (
    "Drivers under age 21 are covered by the zero tolerance rule. A blood alcohol concentration from "
    "0.02 to 0.07 percent leads to a six month suspension of the licence."
)


def _two_column_page(heading: bool = False, table: bool = False) -> list[TextLine]:
    top = PAGE_H - 72
    lines: list[TextLine] = []
    if heading:
        lines.append(TextLine(72, top, "Fleet Driver Safety Handbook - Alcohol and Driving", 14))
        top -= 30
    lines += flow(LEFT, 54, top, 42) + flow(RIGHT, 318, top, 42)
    if table:
        y = top - 140
        for row in [("Vehicle class", "Inspection", "Interval"), ("Van", "Brakes", "12 months"),
                    ("Truck", "Tyres", "6 months")]:
            lines += [TextLine(72, y, row[0]), TextLine(250, y, row[1]), TextLine(430, y, row[2])]
            y -= 14
    return lines


def _single_column_page() -> list[TextLine]:
    text = ("Passwords must be at least 14 characters long and must not reuse any of the previous five "
            "passwords. Multi-factor authentication is required for email, the VPN and all administrative "
            "consoles. Lost or stolen devices must be reported to the IT service desk within 24 hours.")
    return flow(text, 72, PAGE_H - 72, 95)


def _extract(path):
    return "\n".join(t for t, _, _ in ingest.load_doc(str(path)))


def test_two_column_page_is_read_column_by_column(tmp_path):
    pdf = write_pdf(tmp_path / "two.pdf", [_two_column_page()])
    text = " ".join(_extract(pdf).split())
    assert " ".join(LEFT.split()) in text
    assert " ".join(RIGHT.split()) in text
    assert text.index("Drivers aged 21") < text.index("Drivers under age 21")


def test_full_width_heading_is_kept_whole_above_the_columns(tmp_path):
    pdf = write_pdf(tmp_path / "heading.pdf", [_two_column_page(heading=True)])
    text = " ".join(_extract(pdf).split())
    assert text.startswith("Fleet Driver Safety Handbook - Alcohol and Driving")
    assert " ".join(LEFT.split()) in text and " ".join(RIGHT.split()) in text


def test_table_rows_are_not_split_into_columns(tmp_path):
    pdf = write_pdf(tmp_path / "table.pdf", [_two_column_page(table=True)])
    text = " ".join(_extract(pdf).split())
    assert "Vehicle class Inspection Interval Van Brakes 12 months Truck Tyres 6 months" in text
    assert " ".join(LEFT.split()) in text and " ".join(RIGHT.split()) in text


def test_single_column_page_is_unchanged(tmp_path):
    pdf = write_pdf(tmp_path / "single.pdf", [_single_column_page()])
    with pdfplumber.open(pdf) as doc:
        raw = doc.pages[0].extract_text()
    import re
    expected = re.sub(r"\n+", " ", re.sub(r"-\n", "", raw)).strip()
    assert _extract(pdf) == expected


def test_mixed_document_handles_each_page_independently(tmp_path):
    pdf = write_pdf(tmp_path / "mixed.pdf", [_single_column_page(), _two_column_page()])
    entries = ingest.load_doc(str(pdf))
    assert [loc for _, _, loc in entries] == ["p.1", "p.2"]
    assert "Passwords must be at least 14 characters" in entries[0][0]
    p2 = " ".join(entries[1][0].split())
    assert " ".join(LEFT.split()) in p2 and " ".join(RIGHT.split()) in p2


def test_table_only_page_is_not_treated_as_columns(tmp_path):
    rows = [(f"Item {i}", f"{i * 3} units") for i in range(1, 8)]
    lines = []
    y = PAGE_H - 72
    for a, b in rows:
        lines += [TextLine(72, y, a), TextLine(340, y, b)]
        y -= 14
    pdf = write_pdf(tmp_path / "pairs.pdf", [lines])
    joined = " ".join(_extract(pdf).split())
    assert "Item 1 3 units Item 2 6 units" in joined  # rows stay paired, not column-major
