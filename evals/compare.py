"""
evals/compare.py — Compare benchmark runs (e.g. initial, post-fix, final), scored with one scoring version.

    python -m evals compare --run initial=evals/results/official-2026-10-15 \
        --run postfix=evals/results/postfix-2026-10-15 --run final=evals/results/final-2026-10-15 \
        --labels 1.3 --scoring 1.2 --out evals/results/compare-final

Every run is re-scored here with the same label and scoring version, so
differences between runs come from the application, not the scorer. A
separate table changes only the scorer, one step at a time, for each fixed set
of answers. Original-mode records are taken from the first run. Run
directories are only read.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from evals import metrics as m
from evals.dataset import load_cases
from evals.report import _fmt, load_rows

MODES = ("standard", "agentic")
# Scorer steps: each row changes one thing relative to the row above.
SCORER_STEPS = (
    ("1.0", "1.1", "labels 1.0 (as approved), scoring 1.1 (= official 1.0 scores)"),
    ("1.1", "1.1", "+ labels 1.1: FP01 active-voice pattern"),
    ("1.2", "1.1", "+ labels 1.2: T01/V05/C01 contextual facts optional"),
    ("1.2", "1.2", "+ scoring 1.2: abstention re-derived with the clause-level classifier"),
    ("1.3", "1.2", "+ labels 1.3: comparison-consistency check (V01-V04, V06, FP01)"),
)


def _per_case(rows: list[dict], mode: str) -> dict[str, list[bool]]:
    per = defaultdict(list)
    for r in rows:
        if r["mode"] == mode:
            per[r["case_id"]].append(bool(r["keyword_correct"]))
    return per


def source_selection_traces(rows: list[dict]) -> list[dict]:
    """Comparison-mode document choices in agentic records, checked against the expected sources."""
    cases = {c.id: c for c in load_cases()}
    out = []
    for r in rows:
        for sel in r.get("source_selections") or []:
            if sel.get("method") == "reused":
                continue
            expected = sorted({e.source for f in cases[r["case_id"]].expected_facts for e in f.evidence})
            out.append({"case_id": r["case_id"], "repeat": r["repeat"], "method": sel["method"],
                        "selected": sel["selected"], "expected_sources": expected,
                        "missing_expected": [s for s in expected if s not in sel["selected"]],
                        "limitation": sel.get("limitation"), "keyword_correct": r.get("keyword_correct")})
    return sorted(out, key=lambda t: (t["case_id"], t["repeat"]))


def _grounding(rows: list[dict]) -> dict:
    return dict(Counter((r.get("verification") or {}).get("grounding") for r in rows))


def build(runs: list[tuple[str, Path]], out_dir: Path, labels: str = "1.3",
          scoring: str = m.SCORING_VERSION) -> dict:
    names = [n for n, _ in runs]
    loaded = {n: load_rows(Path(d), labels, scoring=scoring) for n, d in runs}
    tagged = []
    for i, (n, _) in enumerate(runs):
        for r in loaded[n][1]:
            if r["mode"] in MODES or (i == 0 and r["mode"] == "original"):
                tagged.append({**r, "mode": f"{r['mode']}@{n}"})
    cols = [f"{mo}@{n}" for mo in MODES for n in names]
    orig = f"original@{names[0]}"
    all_cols = [orig] + cols
    summary = {c: m.aggregate([r for r in tagged if r["mode"] == c]) for c in all_cols}
    categories = sorted({r["category"] for r in tagged})
    by_cat = {cat: {c: m.aggregate([r for r in tagged if r["mode"] == c and r["category"] == cat])["keyword_correct"]
                    for c in all_cols} for cat in categories}
    steps = {mo: {f"{b}_vs_{a}": m.paired_comparison(tagged, f"{mo}@{b}", f"{mo}@{a}")
                  for a, b in zip(names, names[1:], strict=False)} for mo in MODES}
    agentic_vs_standard = {n: m.paired_comparison(tagged, f"agentic@{n}", f"standard@{n}") for n in names}
    case_detail = {mo: {} for mo in MODES}
    for mo in MODES:
        per = {n: _per_case(tagged, f"{mo}@{n}") for n in names}
        for cid in sorted(per[names[0]]):
            case_detail[mo][cid] = {n: f"{sum(per[n][cid])}/{len(per[n][cid])}" for n in names}
    agent_rows = {n: [r for r in loaded[n][1] if r["mode"] == "agentic"] for n in names}
    traces = source_selection_traces(agent_rows[names[-1]])

    scorer_effect = {}
    for lab, sc, label in SCORER_STEPS:
        scorer_effect[label] = {}
        for i, (n, d) in enumerate(runs):
            _, rs = load_rows(Path(d), lab, scoring=sc)
            for mo in (("original",) if i == 0 else ()) + MODES:
                sub = [r for r in rs if r["mode"] == mo]
                if sub:
                    scorer_effect[label][f"{mo}@{n}"] = m.aggregate(sub)["keyword_correct"]

    out = {"runs": {n: str(d) for n, d in runs}, "scoring": {"labels": labels, "scoring_version": scoring},
           "summary": summary, "by_category": by_cat, "steps": steps, "agentic_vs_standard": agentic_vs_standard,
           "case_detail": case_detail, "source_selection": traces, "scorer_effect": scorer_effect,
           "agent_routes": {n: dict(Counter(r.get("route") for r in rs)) for n, rs in agent_rows.items()},
           "agent_statuses": {n: dict(Counter(r.get("status") for r in rs)) for n, rs in agent_rows.items()},
           "agent_grounding": {n: _grounding(rs) for n, rs in agent_rows.items()},
           "verifier_confusion": {n: m.verifier_confusion(rs) for n, rs in agent_rows.items()},
           "environments": {n: loaded[n][0]["environment"] for n in names},
           "runtime": {n: loaded[n][0].get("runtime") for n in names}}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison.json").write_text(json.dumps(out, indent=2, default=str))
    (out_dir / "comparison.md").write_text(_markdown(out, names, all_cols, categories))
    return out


def _markdown(out: dict, names: list[str], cols: list[str], categories: list[str]) -> str:
    sc = out["scoring"]
    md = ["# RAGBase evaluation — run comparison", "",
          f"All runs scored with labels v{sc['labels']} and scoring implementation v{sc['scoring_version']}. "
          "Correctness is a keyword proxy, not semantic ground truth.", ""]
    for n in names:
        env = out["environments"][n]
        md.append(f"- {n}: `{out['runs'][n]}` · commit `{env['git_commit']}` · {env['timestamp']}")
    md += [f"- Original RAGBase: measurements from `{names[0]}` (same code snapshot, corpus, model and settings).",
           "", "## Summary", "", "| Metric | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    rows = [("keyword_correct", "Correct (keyword proxy)"), ("false_abstention", "False abstention (answerable)"),
            ("correct_abstention", "Correct abstention (unanswerable)"), ("forbidden_hit_rate", "Asserted a known-wrong fact"),
            ("citation_ids_valid", "Citation IDs valid"), ("citation_coverage", "Claims carrying a citation"),
            ("cited_precision", "Cited passages contain expected evidence"),
            ("fact_evidence_cited", "Expected facts' evidence cited"),
            ("mode_recall_at_k", "Mode recall@k (first k passages)"), ("pool_recall", "Evidence recall, final pool"),
            ("pool_size", "Passages gathered"), ("llm_calls", "Model calls"), ("retrieval_calls", "Retrieval calls"),
            ("prompt_tokens", "Prompt tokens"), ("completion_tokens", "Completion tokens"),
            ("latency_p50_s", "Latency p50 (s)"), ("latency_p95_s", "Latency p95 (s)"), ("errors", "Errors")]
    for key, label in rows:
        md.append(f"| {label} | " + " | ".join(_fmt(out["summary"][c].get(key)) for c in cols) + " |")
    md += ["", "## Correctness by category", "", "| Category | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for cat in categories:
        md.append(f"| {cat} | " + " | ".join(_fmt(out["by_category"][cat][c]) for c in cols) + " |")
    md += ["", "## Run-to-run changes per case (majority over repeats)", ""]
    for mo, comps in out["steps"].items():
        for step, comp in comps.items():
            md.append(f"- **{mo}, {step.replace('_', ' ')}**: improved {len(comp['wins'])} "
                      f"({', '.join(comp['wins']) or '—'}) · regressed {len(comp['losses'])} "
                      f"({', '.join(comp['losses']) or '—'}) · both right {len(comp['ties_both_right'])} · "
                      f"both wrong {len(comp['ties_both_wrong'])}")
    md += ["", "Correct repeats per case (version comparison, multi-document, conflicting, two-step):", "",
           "| Case | " + " | ".join(f"{mo} {n}" for mo in MODES for n in names) + " |",
           "|---|" + "---|" * (len(MODES) * len(names))]
    for cid in out["case_detail"]["agentic"]:
        if cid[0] in "VMCT":
            md.append(f"| {cid} | " + " | ".join(out["case_detail"][mo][cid][n] for mo in MODES for n in names) + " |")
    md += ["", "## Agentic vs Standard (majority over repeats)", ""]
    for n, comp in out["agentic_vs_standard"].items():
        md.append(f"- {n}: agentic wins {len(comp['wins'])} ({', '.join(comp['wins']) or '—'}) · losses "
                  f"{len(comp['losses'])} ({', '.join(comp['losses']) or '—'}) · both right "
                  f"{len(comp['ties_both_right'])} · both wrong {len(comp['ties_both_wrong'])} · sign test p = "
                  f"{_fmt(comp['sign_test_p'])} (reference only)")
    md += ["", "## Agentic workflow", ""]
    for n in names:
        cm = out["verifier_confusion"][n]
        md.append(f"- {n}: routes {out['agent_routes'][n]} · statuses {out['agent_statuses'][n]} · grounding "
                  f"{out['agent_grounding'][n]} · verified correct/incorrect {cm['verified_correct']}/"
                  f"{cm['verified_incorrect (false positive)']} · flagged correct/incorrect "
                  f"{cm['flagged_correct (false negative)']}/{cm['flagged_incorrect']} · undecided correct/incorrect "
                  f"{cm['undecided_correct']}/{cm['undecided_incorrect']}")
    traces = out["source_selection"]
    ok = sum(1 for t in traces if not t["missing_expected"])
    md += ["", f"## Source selection in comparison mode ({names[-1]} agentic runs)", "",
           f"{len(traces)} comparison selections; all expected sources selected in {ok}/{len(traces)}. "
           f"Limitations recorded: {sum(1 for t in traces if t['limitation'])}.", "",
           "| Case | Rep | Method | Documents searched | Expected sources missing | Correct |", "|---|---|---|---|---|---|"]
    for t in traces:
        md.append(f"| {t['case_id']} | {t['repeat']} | {t['method']} | {', '.join(t['selected'])} | "
                  f"{', '.join(t['missing_expected']) or 'none'} | {'yes' if t['keyword_correct'] else 'no'} |")
    md += ["", "## Scorer changes alone (fixed answers; one change per row)", "",
           "Each column is one fixed set of saved answers; only the scoring changes down the rows.", ""]
    keys = list(next(iter(out["scorer_effect"].values())).keys())
    md += ["| Scoring | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    for lab, vals in out["scorer_effect"].items():
        md.append(f"| {lab} | " + " | ".join(_fmt(vals.get(k)) for k in keys) + " |")
    md.append("")
    return "\n".join(md)
