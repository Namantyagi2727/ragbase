"""
evals/compare.py — Initial (pre-fix) versus post-fix benchmark, scored with one scoring version.

    python -m evals compare --initial evals/results/official-2026-10-15 \
        --postfix evals/results/postfix-2026-10-15 --labels 1.1 --out evals/results/compare-...

Both result sets are re-scored here with the same label and scoring version, so
differences come from the application, not the scorer. A separate table shows
how much the scorer revision alone changes each result set. Both run
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
SCORINGS = (("1.0", False), ("1.1", False), ("1.1", True))


def _tag(rows: list[dict], suffix: str) -> list[dict]:
    return [{**r, "mode": f"{r['mode']}@{suffix}"} for r in rows]


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
                        "limitation": sel.get("limitation"), "retrieval_calls": r.get("retrieval_calls"),
                        "keyword_correct": r.get("keyword_correct")})
    return sorted(out, key=lambda t: (t["case_id"], t["repeat"]))


def build(initial: Path, postfix: Path, out_dir: Path, labels: str = "1.1", include_proposed: bool = False) -> dict:
    meta_i, rows_i = load_rows(initial, labels, include_proposed)
    meta_p, rows_p = load_rows(postfix, labels, include_proposed)
    rows_i = [r for r in rows_i if r["mode"] in MODES + ("original",)]
    rows_p = [r for r in rows_p if r["mode"] in MODES]
    tagged = _tag(rows_i, "initial") + _tag(rows_p, "postfix")
    cols = [f"{mo}@{s}" for mo in MODES for s in ("initial", "postfix")]
    summary = {c: m.aggregate([r for r in tagged if r["mode"] == c]) for c in cols}
    summary["original@initial"] = m.aggregate([r for r in tagged if r["mode"] == "original@initial"])
    categories = sorted({r["category"] for r in tagged})
    by_cat = {cat: {c: m.aggregate([r for r in tagged if r["mode"] == c and r["category"] == cat])["keyword_correct"]
                    for c in cols + ["original@initial"]} for cat in categories}
    per_mode = {mo: m.paired_comparison(tagged, f"{mo}@postfix", f"{mo}@initial") for mo in MODES}
    agentic_vs_standard = {
        "initial": m.paired_comparison(tagged, "agentic@initial", "standard@initial"),
        "postfix": m.paired_comparison(tagged, "agentic@postfix", "standard@postfix"),
    }
    case_detail = {}
    for mo in MODES:
        a, b = _per_case(tagged, f"{mo}@initial"), _per_case(tagged, f"{mo}@postfix")
        case_detail[mo] = {cid: {"initial": f"{sum(a[cid])}/{len(a[cid])}", "postfix": f"{sum(b[cid])}/{len(b[cid])}"}
                           for cid in sorted(a)}
    traces = source_selection_traces([r for r in rows_p if r["mode"] == "agentic"])
    agent_p = [r for r in rows_p if r["mode"] == "agentic"]
    agent_i = [r for r in rows_i if r["mode"] == "agentic"]

    scorer_effect = {}
    for name, d in (("initial", initial), ("postfix", postfix)):
        for lab, prop in SCORINGS:
            _, rs = load_rows(d, lab, prop)
            key = f"labels {lab}" + (" + proposed" if prop else "")
            scorer_effect.setdefault(key, {})
            for mo in ("original", *MODES):
                sub = [r for r in rs if r["mode"] == mo]
                if sub:
                    scorer_effect[key][f"{mo}@{name}"] = m.aggregate(sub)["keyword_correct"]

    out = {"initial": str(initial), "postfix": str(postfix),
           "scoring": {"labels": labels, "include_proposed": include_proposed, "scoring_version": m.SCORING_VERSION},
           "summary": summary, "by_category": by_cat, "postfix_vs_initial": per_mode,
           "agentic_vs_standard": agentic_vs_standard, "case_detail": case_detail,
           "source_selection": traces, "scorer_effect": scorer_effect,
           "agent_routes": {"initial": dict(Counter(r.get("route") for r in agent_i)),
                            "postfix": dict(Counter(r.get("route") for r in agent_p))},
           "agent_statuses": {"initial": dict(Counter(r.get("status") for r in agent_i)),
                              "postfix": dict(Counter(r.get("status") for r in agent_p))},
           "environments": {"initial": meta_i["environment"], "postfix": meta_p["environment"]},
           "configs": {"initial": meta_i["config"], "postfix": meta_p["config"]}}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison.json").write_text(json.dumps(out, indent=2, default=str))
    (out_dir / "comparison.md").write_text(_markdown(out, cols, categories))
    return out


def _markdown(out: dict, cols: list[str], categories: list[str]) -> str:
    sc = out["scoring"]
    env_i, env_p = out["environments"]["initial"], out["environments"]["postfix"]
    md = ["# RAGBase evaluation — initial (pre-fix) vs post-fix", "",
          f"Both result sets scored with labels v{sc['labels']}"
          + (" including **proposed** revisions" if sc["include_proposed"] else "")
          + f" and scoring implementation v{sc['scoring_version']}. Correctness is a keyword proxy.", "",
          f"- Initial: `{out['initial']}` · commit `{env_i['git_commit']}` · {env_i['timestamp']}",
          f"- Post-fix: `{out['postfix']}` · commit `{env_p['git_commit']}` · {env_p['timestamp']}",
          "- Original RAGBase: initial measurements reused (same code snapshot, corpus, model and settings).", "",
          "## Summary", "", "| Metric | original | " + " | ".join(cols) + " |", "|---|" + "---|" * (len(cols) + 1)]
    rows = [("keyword_correct", "Correct (keyword proxy)"), ("false_abstention", "False abstention (answerable)"),
            ("correct_abstention", "Correct abstention (unanswerable)"), ("forbidden_hit_rate", "Asserted a known-wrong fact"),
            ("citation_ids_valid", "Citation IDs valid"), ("cited_precision", "Cited passages contain expected evidence"),
            ("fact_evidence_cited", "Expected facts' evidence cited"),
            ("mode_recall_at_k", "Mode recall@k (first k passages)"), ("pool_recall", "Evidence recall, final pool"),
            ("pool_size", "Passages gathered"), ("llm_calls", "Model calls"), ("retrieval_calls", "Retrieval calls"),
            ("prompt_tokens", "Prompt tokens"), ("completion_tokens", "Completion tokens"),
            ("latency_p50_s", "Latency p50 (s)"), ("latency_p95_s", "Latency p95 (s)"), ("errors", "Errors")]
    for key, label in rows:
        md.append(f"| {label} | {_fmt(out['summary']['original@initial'].get(key))} | "
                  + " | ".join(_fmt(out["summary"][c].get(key)) for c in cols) + " |")
    md += ["", "## Correctness by category", "", "| Category | original | " + " | ".join(cols) + " |",
           "|---|" + "---|" * (len(cols) + 1)]
    for cat in categories:
        md.append(f"| {cat} | {_fmt(out['by_category'][cat]['original@initial'])} | "
                  + " | ".join(_fmt(out["by_category"][cat][c]) for c in cols) + " |")
    md += ["", "## Post-fix vs initial, per case (majority over 3 repeats)", ""]
    for mo, comp in out["postfix_vs_initial"].items():
        md += [f"**{mo}**: improved {len(comp['wins'])} · regressed {len(comp['losses'])} · both right "
               f"{len(comp['ties_both_right'])} · both wrong {len(comp['ties_both_wrong'])}",
               f"Improved: {', '.join(comp['wins']) or '—'} · Regressed: {', '.join(comp['losses']) or '—'}", ""]
    md += ["Per-case correct repeats for version-comparison and multi-document questions:", "",
           "| Case | standard initial | standard post-fix | agentic initial | agentic post-fix |", "|---|---|---|---|---|"]
    for cid, d in out["case_detail"]["agentic"].items():
        if cid[0] in "VM" or cid in ("C04",):
            s = out["case_detail"]["standard"][cid]
            md.append(f"| {cid} | {s['initial']} | {s['postfix']} | {d['initial']} | {d['postfix']} |")
    md += ["", "## Agentic vs Standard (majority over repeats)", ""]
    for when, comp in out["agentic_vs_standard"].items():
        md.append(f"- {when}: agentic wins {len(comp['wins'])} ({', '.join(comp['wins']) or '—'}) · losses "
                  f"{len(comp['losses'])} ({', '.join(comp['losses']) or '—'}) · both right {len(comp['ties_both_right'])}"
                  f" · both wrong {len(comp['ties_both_wrong'])} · sign test p = {_fmt(comp['sign_test_p'])} (reference only)")
    md += ["", f"Agent routes: initial {out['agent_routes']['initial']} · post-fix {out['agent_routes']['postfix']}",
           f"Agent statuses: initial {out['agent_statuses']['initial']} · post-fix {out['agent_statuses']['postfix']}"]
    traces = out["source_selection"]
    ok = sum(1 for t in traces if not t["missing_expected"])
    md += ["", "## Source selection in comparison mode (post-fix agentic runs)", "",
           f"{len(traces)} comparison selections; all expected sources selected in {ok}/{len(traces)}. "
           f"Limitations recorded: {sum(1 for t in traces if t['limitation'])}.", "",
           "| Case | Rep | Method | Documents searched | Expected sources missing | Correct |", "|---|---|---|---|---|---|"]
    for t in traces:
        md.append(f"| {t['case_id']} | {t['repeat']} | {t['method']} | {', '.join(t['selected'])} | "
                  f"{', '.join(t['missing_expected']) or 'none'} | {'yes' if t['keyword_correct'] else 'no'} |")
    md += ["", "## Scorer revision alone (same answers, different label versions)", "",
           "Each column is one fixed set of answers; only the scoring changes across rows.", ""]
    keys = list(next(iter(out["scorer_effect"].values())).keys())
    md += ["| Labels | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    for lab, vals in out["scorer_effect"].items():
        md.append(f"| {lab} | " + " | ".join(_fmt(vals.get(k)) for k in keys) + " |")
    md.append("")
    return "\n".join(md)
