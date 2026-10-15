"""
evals/report.py — Score run records and write summary.csv, scored_rows.jsonl and report.md.

    python -m evals report <run_dir>
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from citations import _NON_EVIDENCE_RE, is_abstention
from evals import metrics as m
from evals.dataset import load_cases
from evals.verifier_eval import summarize_unique

MODES = ["original", "standard", "agentic"]


def _fixed_retrieval(run_dir: Path) -> dict:
    fixed = {}
    for line in (run_dir / "fixed_retrieval.jsonl").read_text().splitlines():
        entry = json.loads(line)
        fixed[entry["case_id"]] = [{"source": src, "text": txt} for src, txt in zip(entry["sources"], entry["texts"], strict=True)]
    return fixed


def load_rows(run_dir: Path, labels: str = "1.0", include_proposed: bool = False,
              original_from: Path | None = None, scoring: str = m.SCORING_VERSION) -> tuple[dict, list[dict]]:
    """Score a run's records. Reads only; nothing in run_dir is modified.

    original_from: take the original-mode records from another run (e.g. the
    pre-fix baseline, when the original code and its conditions are unchanged).
    """
    meta = json.loads((run_dir / "config.json").read_text())
    cases = {c.id: c for c in load_cases(labels=labels, include_proposed=include_proposed)}
    k = meta["config"]["top_k"]
    sources = [(run_dir, lambda rec: original_from is None or rec["mode"] != "original")]
    if original_from is not None:
        sources.append((Path(original_from), lambda rec: rec["mode"] == "original"))
    rows = []
    for src_dir, keep in sources:
        fixed = _fixed_retrieval(src_dir)
        for line in (src_dir / "records.jsonl").read_text().splitlines():
            rec = json.loads(line)
            if not keep(rec):
                continue
            case = cases[rec["case_id"]]
            if rec["mode"] == "original":
                rec["abstained"] = is_abstention(rec.get("answer") or "")
                rec["free_form_references"] = len(_NON_EVIDENCE_RE.findall(rec.get("answer") or ""))
            scores = m.score_record(case, rec, fixed.get(case.id, []), k, scoring=scoring)
            rows.append({**rec, **scores, "answerable": case.answerable, "required": case.required})
    meta = {**meta, "scoring": {"labels": labels, "include_proposed_revisions": include_proposed,
                                "scoring_version": scoring,
                                "original_from": str(original_from) if original_from else None}}
    return meta, rows


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def build(run_dir: Path, out_dir: Path | None = None, labels: str = "1.0", include_proposed: bool = False,
          original_from: Path | None = None, scoring: str = m.SCORING_VERSION) -> dict:
    """Score run_dir and write the report to out_dir (default: run_dir itself)."""
    meta, rows = load_rows(run_dir, labels, include_proposed, original_from, scoring)
    out_dir = Path(out_dir) if out_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    modes = [mo for mo in MODES if any(r["mode"] == mo for r in rows)]
    by_mode = {mo: [r for r in rows if r["mode"] == mo] for mo in modes}
    summary = {mo: {**m.aggregate(rs), "repeat_spread": m.repeat_spread(rs)} for mo, rs in by_mode.items()}
    categories = sorted({r["category"] for r in rows})
    by_cat = {c: {mo: m.aggregate([r for r in by_mode[mo] if r["category"] == c])["keyword_correct"] for mo in modes}
              for c in categories}
    comparisons = {}
    if "standard" in modes and "agentic" in modes:
        comparisons["agentic_vs_standard"] = m.paired_comparison(rows, "agentic", "standard")
        complex_cases = {r["case_id"] for r in by_mode["agentic"] if r.get("route") == "complex"}
        comparisons["agentic_vs_standard (complex-routed cases only)"] = m.paired_comparison(
            [r for r in rows if r["case_id"] in complex_cases], "agentic", "standard")
    if "original" in modes and "standard" in modes:
        comparisons["standard_vs_original"] = m.paired_comparison(rows, "standard", "original")
    agent = by_mode.get("agentic", [])
    agent_info = {
        "routes": dict(Counter(r.get("route") for r in agent)),
        "planner_fallbacks": sum(1 for r in agent if r.get("planner_fallback")),
        "statuses": dict(Counter(r.get("status") for r in agent)),
        "by_route": {route: m.aggregate([r for r in agent if r.get("route") == route]) for route in ("simple", "complex")},
        "verifier_confusion_keyword_proxy": m.verifier_confusion(agent),
    }
    required = {mo: {r["case_id"]: [] for r in rs if r["required"]} for mo, rs in by_mode.items()}
    for mo, rs in by_mode.items():
        for r in rs:
            if r["required"]:
                required[mo][r["case_id"]].append(bool(r["keyword_correct"]) and not r["forbidden_hits"])

    out = {"meta": meta, "summary": summary, "by_category": by_cat, "comparisons": comparisons,
           "agent": agent_info, "required_regressions": required}
    _write(out_dir, out, rows, modes, categories, supplementary_from=run_dir)
    return out


def _write(run_dir: Path, out: dict, rows: list[dict], modes: list[str], categories: list[str],
           supplementary_from: Path | None = None) -> None:
    keep = ["mode", "repeat", "case_id", "category", "answerable", "required", "keyword_correct", "facts_covered",
            "facts_total", "forbidden_hits", "comparison_inconsistencies", "abstained", "citation_ids_valid", "citation_coverage", "cited_precision",
            "fact_evidence_cited", "shared_recall_at_k", "mode_hit_at_k", "mode_recall_at_k", "pool_recall", "pool_size",
            "numeric_mismatches",
            "route", "planner_fallback", "status", "revision_count", "llm_calls", "retrieval_calls", "prompt_tokens",
            "completion_tokens", "latency_s", "seed", "error"]
    with open(run_dir / "scored_rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps({k: r.get(k) for k in keep} | {"answer": r.get("answer"),
                                "verification": r.get("verification")}) + "\n")
    with open(run_dir / "summary.csv", "w", newline="") as f:
        cols = ["mode", "n", "keyword_correct", "correct_abstention", "false_abstention", "forbidden_hit_rate",
                "citation_ids_valid", "citation_coverage", "cited_precision", "fact_evidence_cited",
                "shared_recall_at_k", "mode_hit_at_k", "mode_recall_at_k", "pool_recall", "pool_size", "llm_calls",
                "retrieval_calls", "prompt_tokens",
                "completion_tokens", "revision_count", "latency_p50_s", "latency_p95_s", "errors"]
        w = csv.writer(f)
        w.writerow(cols)
        for mo in modes:
            w.writerow([mo] + [out["summary"][mo].get(c) for c in cols[1:]])

    meta = out["meta"]
    cfg, env = meta["config"], meta["environment"]
    md = [f"# RAGBase evaluation — {cfg['label']} run", ""]
    for err in meta.get("mode_errors", []):
        md += [f"> **{err['mode']} mode did not run:** {err['error']}.", ""]
    if cfg["label"] != "official":
        md += ["> **Not an official benchmark result.** Labels may be unapproved or the run is for development.", ""]
    md += [f"Dataset review status: {', '.join(meta['dataset_review_status'])} · corpus {env['corpus_version']} "
           f"(`{env['corpus_checksum']}`) · commit `{env['git_commit']}` · {env['timestamp']}",
           f"Generator `{cfg['generator_model']}` via {env['backend']} (Ollama {env.get('ollama_version')}) · "
           f"embeddings `{cfg['embed_model']}` · reranking {cfg['reranking']} · top-k {cfg['top_k']} · "
           f"{cfg['repeats']} repeat(s), base seed {cfg['base_seed']} · temperature "
           f"{cfg['temperature'] if cfg['temperature'] is not None else 'application defaults'}", "",
           f"Scoring: labels v{meta['scoring']['labels']}"
           + (" including **proposed** (unapproved) revisions" if meta["scoring"]["include_proposed_revisions"] else "")
           + f" · scoring implementation v{meta['scoring']['scoring_version']}"
           + (f" · original-mode records reused from `{meta['scoring']['original_from']}`"
              if meta["scoring"]["original_from"] else ""), "",
           "Correctness below is a **keyword proxy** against reviewed reference facts; it is not semantic ground truth.", "",
           "## Summary by mode", "",
           "| Metric | " + " | ".join(modes) + " |", "|---|" + "---|" * len(modes)]
    labels = [("keyword_correct", "Correct (keyword proxy)"), ("correct_abstention", "Correct abstention (unanswerable)"),
              ("false_abstention", "False abstention (answerable)"), ("forbidden_hit_rate", "Asserted a known-wrong fact"),
              ("citation_ids_valid", "Citation IDs valid"), ("cited_precision", "Cited passages contain expected evidence"),
              ("fact_evidence_cited", "Expected facts' evidence cited"), ("citation_coverage", "Claims carrying a citation"),
              ("shared_recall_at_k", "Reference: shared corrected-retriever recall@k (same for all modes)"),
              ("mode_hit_at_k", "This mode's hit@k (first k passages it retrieved)"),
              ("mode_recall_at_k", "This mode's recall@k (first k passages it retrieved)"),
              ("pool_recall", "Evidence recall, final pool"), ("pool_size", "Passages gathered"),
              ("llm_calls", "Model calls"), ("retrieval_calls", "Retrieval calls"), ("prompt_tokens", "Prompt tokens"),
              ("completion_tokens", "Completion tokens"), ("revision_count", "Revisions"),
              ("latency_p50_s", "Latency p50 (s)"), ("latency_p95_s", "Latency p95 (s)"), ("errors", "Errors")]
    for key, label in labels:
        md.append(f"| {label} | " + " | ".join(_fmt(out["summary"][mo].get(key)) for mo in modes) + " |")
    md += ["", "Repeat spread of keyword-proxy correctness: " + "; ".join(
        f"{mo} {out['summary'][mo]['repeat_spread']['per_repeat']}" for mo in modes), "",
        "## Correctness by category (keyword proxy)", "", "| Category | " + " | ".join(modes) + " |",
        "|---|" + "---|" * len(modes)]
    for c in categories:
        md.append(f"| {c} | " + " | ".join(_fmt(out["by_category"][c][mo]) for mo in modes) + " |")
    for name, comp in out["comparisons"].items():
        md += ["", f"## {name}", "",
               f"Wins {len(comp['wins'])} · losses {len(comp['losses'])} · both right {len(comp['ties_both_right'])} · "
               f"both wrong {len(comp['ties_both_wrong'])} (per-case majority over repeats).",
               f"Sign test on {comp['discordant']} discordant cases: p = {_fmt(comp['sign_test_p'])} "
               "(small sample; for reference only, not a claim of significance).",
               f"Wins: {', '.join(comp['wins']) or '—'} · Losses: {', '.join(comp['losses']) or '—'}"]
    if out["agent"]["routes"]:
        a = out["agent"]
        md += ["", "## Agentic workflow", "",
               f"Routes: {a['routes']} · planner fallbacks: {a['planner_fallbacks']} · statuses: {a['statuses']}", "",
               "Simple-routed questions run the Standard RAG path inside the agent; they are not evidence for the "
               "agentic architecture.", "",
               "| Route | n | Correct (proxy) | Model calls | Latency p50 (s) |", "|---|---|---|---|---|"]
        for route, agg in a["by_route"].items():
            md.append(f"| {route} | {agg['n']} | {_fmt(agg['keyword_correct'])} | {_fmt(agg['llm_calls'])} | {_fmt(agg['latency_p50_s'])} |")
        cm = a["verifier_confusion_keyword_proxy"]
        md += ["", "Verifier outcome vs. keyword-proxy correctness (abstentions excluded):", "",
               "| | Proxy-correct | Proxy-incorrect |", "|---|---|---|",
               f"| Verified | {cm['verified_correct']} | {cm['verified_incorrect (false positive)']} |",
               f"| Flagged | {cm['flagged_correct (false negative)']} | {cm['flagged_incorrect']} |",
               f"| Undecided | {cm['undecided_correct']} | {cm['undecided_incorrect']} |"]
    md += ["", "## Required regression cases", ""]
    for mo, cases_ in out["required_regressions"].items():
        md.append(f"- {mo}: " + ", ".join(f"{cid} {sum(v)}/{len(v)}" for cid, v in sorted(cases_.items())))
    md += _supplementary_sections(supplementary_from or run_dir)
    md += ["", "## Limitations", "",
           "- Keyword matching is a proxy; paraphrases can be missed and lucky matches counted.",
           "- Small sample (about 40 cases); differences of a few cases are within run-to-run variation.",
           "- The original-code baseline uses its own ingestion (no column-aware PDF extraction) and free-form "
           "citations, so citation metrics do not apply to it.",
           ""]
    (run_dir / "report.md").write_text("\n".join(md))


def _supplementary_sections(run_dir: Path) -> list[str]:
    """Verifier challenge, tabular suite and judge results, when present in the run directory."""
    md: list[str] = []
    vc_path = run_dir / "verifier_challenge.json"
    if vc_path.exists():
        vc = json.loads(vc_path.read_text())
        u = summarize_unique(vc["results"])
        md += ["", "## Verifier challenge set (fixed answer/evidence pairs)", "",
               f"Model `{vc['model']}`, {vc['repeats']} repeat(s) at temperature 0. Counts are per distinct claim "
               f"({u['unique_claims']} claims: {u['should_accept']} should be accepted, {u['should_reject']} rejected); "
               "repeats are not independent cases. A claim whose repeats disagree is counted as inconsistent.", "",
               "| | Should accept | Should reject |", "|---|---|---|",
               f"| Accepted | {u['accepted_supported']} | {u['accepted_unsupported (false positive)']} |",
               f"| Rejected | {u['rejected_supported (false negative)']} | {u['rejected_unsupported']} |",
               f"| Undecided | {u['undecided_supported']} | {u['undecided_unsupported']} |",
               f"| Inconsistent across repeats | {u['inconsistent_supported']} | {u['inconsistent_unsupported']} |", "",
               f"False positives {u['false_positive_rate']} · false negatives {u['false_negative_rate']}. "
               "Errors: " + (", ".join(f"{case} ({outcome})" for case, outcome in u["errors"]) or "none") + ".",
               "Answer-level (verified / not verified) by label, over all repeats: " + "; ".join(
                   f"{k}: {v}" for k, v in vc["summary"]["answers_by_label"].items()),
               "Required cases: " + ", ".join(f"{k} → {v}" for k, v in vc["summary"]["required"].items())]
    tab_runs = sorted(run_dir.glob("tabular_run*.json"))
    if tab_runs:
        runs = [json.loads(p.read_text()) for p in tab_runs]
        md += ["", "## Tabular regression suite", "",
               f"{len(runs)} run(s) with `{runs[0]['model']}`. Security checks are based on sandboxed state and "
               "patched execution boundaries.", "", "| Case | Passed per run | Outcome (run 1) | Security violations |",
               "|---|---|---|---|"]
        for res in runs[0]["results"]:
            per = [next(r for r in run["results"] if r["case_id"] == res["case_id"]) for run in runs]
            md.append(f"| {res['case_id']} | {' '.join('✓' if p['passed'] else '✗' for p in per)} | {res['outcome']} | "
                      f"{sum(len(p['security_violations']) for p in per)} |")
    judge_path = run_dir / "judge_summary.json"
    if judge_path.exists():
        j = json.loads(judge_path.read_text())
        md += ["", "## LLM judge (supplementary, not ground truth)", "",
               f"`{j['judge_model']}` (digest `{j.get('judge_model_digest')}`), {j['items']} items × {j['judgments_per_item']} "
               f"judgment(s), blind to mode. Agreement with the keyword proxy: {_fmt(j['agreement_with_keyword_proxy'])}; "
               f"self-consistency {_fmt(j['self_consistency'])} (temperature 0, so repeats are expected to agree). "
               f"Failed judgments: {j['failed_judgments']}.", "",
               "| Item | Mode | Case | Judge | Keyword proxy | First issue |", "|---|---|---|---|---|---|"]
        seen = set()
        for d_ in j["disagreements"]:
            key = (d_["mode"], d_["case_id"])
            if key in seen:
                continue
            seen.add(key)
            issue = (d_["issues"] or [""])[0].replace("|", "\\|")
            md.append(f"| {d_['item']} | {d_['mode']} | {d_['case_id']} | {'pass' if d_['judge_passed'] else 'fail'} | "
                      f"{'correct' if d_['keyword_correct'] else 'incorrect'} | {issue} |")
    return md
