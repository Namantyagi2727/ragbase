"""
evals/review_export.py — Human-readable review tables for the benchmark labels.

    python -m evals.review_export

Writes evals/review/*.csv and *.md from the JSONL sources. Edit labels in the
JSONL files (not in these exports), then re-run.
"""

from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

from evals import dataset as d

REVIEW_DIR = d.EVALS / "review"


def _csv(header: list[str], rows: list[list[str]]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue()


def _md(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", "<br>")


def _cases():
    cases = d.load_cases()
    header = ["id", "category", "question", "answerable", "reference_answer", "expected_facts",
              "evidence (source: quote)", "forbidden_facts", "missing_information", "required", "review_status", "notes"]
    rows = []
    for c in cases:
        facts = "\n".join(f"{f.id}: {f.statement}" for f in c.expected_facts)
        evidence = "\n".join(f"{f.id} - {e.source}: \"{e.quote}\"" for f in c.expected_facts for e in f.evidence)
        forbidden = "\n".join(f.statement for f in c.forbidden_facts)
        rows.append([c.id, c.category, c.question, c.answerable, c.reference_answer, facts, evidence, forbidden,
                     "; ".join(c.missing_information), "yes" if c.required else "", c.review_status, c.notes])
    md = ["# Benchmark cases — label review", "",
          f"{len(cases)} cases. Every case is `proposed` until reviewed. For each case, check that the "
          "reference answer is correct and complete, that each expected fact is supported by its quoted "
          "evidence, and that forbidden facts describe genuinely wrong answers.", "",
          "| ID | Category | Question | Reference answer | Evidence | Forbidden | Status |",
          "|---|---|---|---|---|---|---|"]
    for row in rows:
        md.append(f"| {row[0]}{' ⚑' if row[9] else ''} | {row[1]} | {_md(row[2])} | {_md(row[4])}"
                  f"{'<br>*missing:* ' + _md(row[8]) if row[8] else ''} | {_md(row[6]) or '—'} | {_md(row[7]) or '—'} | {row[10]} |")
    md += ["", "⚑ = required regression case."]
    return _csv(header, rows), "\n".join(md) + "\n"


def _verifier():
    vcs = d.load_verifier_cases()
    header = ["id", "kind", "question", "evidence", "claim", "supported_by_cited_evidence", "factually_true",
              "answer_label", "required", "review_status", "notes"]
    rows = []
    for v in vcs:
        evidence = "\n".join(f"[{e.id}] {e.source}: {e.text}" for e in v.evidence)
        for i, c in enumerate(v.claims):
            rows.append([v.id if i == 0 else "", v.kind if i == 0 else "", v.question if i == 0 else "",
                         evidence if i == 0 else "", c.text, c.label, "yes" if c.factually_true else "no",
                         v.answer_label if i == 0 else "",
                         ("yes" if v.required else "") if i == 0 else "", v.review_status if i == 0 else "",
                         v.notes if i == 0 else ""])
    md = ["# Verifier challenge set — label review", "",
          f"{len(vcs)} fixed answer/evidence pairs. Each claim has two labels: whether the evidence it "
          "*cites* supports it (what the verifier should conclude), and whether it is factually true "
          "according to the corpus.", "",
          "| ID | Kind | Evidence | Claim | Supported by cited evidence | Factually true | Answer | Status |",
          "|---|---|---|---|---|---|---|---|"]
    for row in rows:
        md.append(f"| {row[0]} | {row[1]} | {_md(row[3]) or ''} | {_md(row[4])} | {row[5]} | {row[6]} | {row[7]} | {row[9]} |")
    return _csv(header, rows), "\n".join(md) + "\n"


def _tabular():
    tcs = d.load_tabular_cases()
    header = ["id", "kind", "question", "expected_operation", "expected_column", "expected_filters",
              "expected_group_by", "expected_value", "acceptable_outcomes", "security_checks", "review_status", "notes"]
    rows = [[t.id, t.kind, t.question, t.expected_operation or "", t.expected_column or "",
             "; ".join(f"{f.column} {f.operator} {f.value}" for f in t.expected_filters),
             t.expected_group_by or "", t.expected_value or "", ", ".join(t.acceptable_outcomes),
             ", ".join(t.security_checks), t.review_status, t.notes] for t in tcs]
    md = ["# Tabular regression suite — label review", "",
          "Expected values were computed from `evals/tabular/fleet_usage.csv` with the allowlisted executor. "
          "Every case also checks, through sandboxed file state and patched execution boundaries, that no "
          "Python execution, shell command, file deletion or filesystem change occurs.", "",
          "| ID | Kind | Question | Expected query | Expected value | Passing outcomes | Notes | Status |",
          "|---|---|---|---|---|---|---|---|"]
    for row in rows:
        query = f"{row[3]}({row[4]})" + (f" where {row[5]}" if row[5] else "") + (f" by {row[6]}" if row[6] else "")
        md.append(f"| {row[0]} | {row[1]} | {_md(row[2])} | {_md(query) if row[3] else 'any validated query'} | "
                  f"{_md(row[7]) or '—'} | {row[8]} | {_md(row[11])} | {row[10]} |")
    return _csv(header, rows), "\n".join(md) + "\n"


def render_all() -> dict[str, str]:
    out = {}
    for name, fn in (("benchmark", _cases), ("verifier_challenge", _verifier), ("tabular", _tabular)):
        csv_text, md_text = fn()
        out[str(REVIEW_DIR / f"{name}_review.csv")] = csv_text
        out[str(REVIEW_DIR / f"{name}_review.md")] = md_text
    return out


def main() -> int:
    REVIEW_DIR.mkdir(exist_ok=True)
    for path, content in render_all().items():
        Path(path).write_text(content)
        print("wrote", Path(path).relative_to(d.EVALS.parent))
    return 0


if __name__ == "__main__":
    sys.exit(main())
