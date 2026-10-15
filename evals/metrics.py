"""
evals/metrics.py — Deterministic evaluation metrics.

Four things are measured separately and never merged into one score:

  1. Citation ID validity   — cited IDs exist in the evidence actually provided.
  2. Citation relevance     — the cited passages are the ones that contain the
                              expected evidence.
  3. Factual correctness    — keyword proxy: expected facts present, no known
                              wrong facts asserted. A proxy, not semantic truth.
  4. Evidence faithfulness  — deterministic proxy: numbers in cited claims
                              appear in the cited evidence; share of claims cited.

Retrieval is reported three ways, all source-aware: a shared reference (the
corrected retriever's top-k for the question, identical for every mode); each
mode's own fixed budget (the first k passages it actually retrieved, in order);
and each mode's final evidence pool.
"""

from __future__ import annotations

import math
import re
import statistics
from collections import Counter, defaultdict
from typing import Iterable

from citations import CITE_RE, is_absence_statement, is_abstention
from evals.dataset import Case, normalize

NEGATION_CUE = re.compile(
    r"\b(not|no|never|cannot|can't|isn't|aren't|doesn't|don't|didn't|won't|wasn't|weren't|"
    r"incorrect|false|wrong|rather than|instead of|unlike|neither|nor|premise)\b",
    re.I,
)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text or "") if s.strip()]


def _alt_pattern(alt: str) -> re.Pattern:
    """Left word boundary always; right boundary only after a digit ("2" must not match "2024",
    but "revo" should match "revocation")."""
    core = re.escape(normalize(alt))
    tail = r"(?!\d)" if re.search(r"\d$", alt) else ""
    return re.compile(rf"(?<![\w]){core}{tail}")


def keyword_group_matched(group: list[str], text: str) -> bool:
    norm = normalize(CITE_RE.sub(" ", text))
    return any(_alt_pattern(alt).search(norm) for alt in group)


def fact_covered(fact_keywords: list[list[str]], answer: str) -> bool:
    return all(keyword_group_matched(g, answer) for g in fact_keywords)


def forbidden_hits(patterns: Iterable[tuple[str, str]], answer: str) -> list[str]:
    """Statements of known-wrong facts asserted in the answer.

    Each pattern is applied per sentence; sentences with a negation or
    rejection cue are skipped, so rejecting a wrong value is not penalised.
    """
    hits = []
    for statement, pattern in patterns:
        rx = re.compile(pattern, re.I)
        for sentence in sentences(answer):
            clean = normalize(CITE_RE.sub(" ", sentence))
            if not NEGATION_CUE.search(clean) and rx.search(clean):
                hits.append(statement)
                break
    return hits


def acknowledges_missing(answer: str) -> bool:
    return is_abstention(answer) or any(is_absence_statement(s) for s in sentences(answer))


Item = dict  # {"source": filename, "text": passage}


def _expected(case: Case) -> list[tuple[str, str]]:
    return [(e.source, e.quote) for f in case.expected_facts for e in f.evidence]


def evidence_present(expected: list[tuple[str, str]], items: list[Item]) -> list[bool]:
    """For each expected (source, quote): is it in some item from that same source?

    Matching the text alone would credit an identical sentence found in a
    different document; the source must match too.
    """
    norm = [(it.get("source"), normalize(it.get("text") or "")) for it in items]
    return [any(src == source and normalize(quote) in text for src, text in norm) for source, quote in expected]


def retrieval_metrics(case: Case, ranked: list[Item], k: int) -> dict:
    """Hit@k, Recall@k and MRR of the expected evidence within the first k ranked items."""
    expected = _expected(case)
    if not expected:
        return {"hit_at_k": None, "recall_at_k": None, "mrr": None}
    top = ranked[:k]
    found = evidence_present(expected, top)
    first = next((i + 1 for i, it in enumerate(top) if any(evidence_present(expected, [it]))), None)
    return {"hit_at_k": float(any(found)), "recall_at_k": sum(found) / len(found),
            "mrr": 1 / first if first else 0.0}


def pool_recall(case: Case, items: list[Item]) -> float | None:
    expected = _expected(case)
    if not expected:
        return None
    found = evidence_present(expected, items)
    return sum(found) / len(found)


def citation_relevance(case: Case, cited: list[Item]) -> dict:
    """Do the cited passages contain the expected evidence, from the expected source?

    Independent of citation ID validity.
    """
    expected = _expected(case)
    if not expected or not cited:
        return {"cited_precision": None if not cited else 0.0 if expected else None,
                "fact_evidence_cited": None if not expected else 0.0}
    relevant = [any(evidence_present(expected, [it])) for it in cited]
    facts_cited = [any(evidence_present([(e.source, e.quote) for e in f.evidence], cited)) for f in case.expected_facts]
    return {"cited_precision": sum(relevant) / len(relevant),
            "fact_evidence_cited": sum(facts_cited) / len(facts_cited)}


def score_record(case: Case, record: dict, shared_ranked: list[Item], k: int) -> dict:
    """All deterministic metrics for one answer.

    shared_ranked: the corrected retriever's top-k for the question (the same
    reference for every mode). Mode-specific retrieval is measured on the
    record's own evidence, in the order it was retrieved.
    """
    answer = record.get("answer") or ""
    evidence = record.get("evidence") or []
    items = [{"source": e.get("source"), "text": e.get("text") or ""} for e in evidence]
    cited_ids = set(record.get("cited_ids") or [])
    cited = [{"source": e.get("source"), "text": e.get("text") or ""} for e in evidence if e.get("eid") in cited_ids]
    abstained = bool(record.get("abstained"))

    coverage = [fact_covered(f.keywords, answer) for f in case.expected_facts]
    wrong = forbidden_hits([(f.statement, f.pattern) for f in case.forbidden_facts], answer)
    if record.get("error"):
        correct = False
    elif case.answerable == "yes":
        correct = all(coverage) and not wrong and not abstained
    elif case.answerable == "partial":
        correct = all(coverage) and acknowledges_missing(answer) and not wrong
    else:
        correct = (abstained or acknowledges_missing(answer)) and not wrong

    report = record.get("citations") or {}
    out = {
        "keyword_correct": correct,
        "facts_covered": sum(coverage), "facts_total": len(coverage),
        "forbidden_hits": wrong,
        "abstained": abstained,
        "correct_abstention": abstained if case.answerable == "no" else None,
        "false_abstention": abstained if case.answerable == "yes" else None,
        # 1. citation ID validity
        "citation_ids_valid": report.get("ok"),
        "invalid_citation_ids": len(report.get("invalid_ids", [])) if report else None,
        "non_evidence_citations": len(report.get("non_evidence_citations", [])) if report else None,
        "citation_coverage": report.get("citation_coverage"),
        # 4. faithfulness proxies
        "numeric_mismatches": record.get("numeric_mismatches"),
        # retrieval: shared corrected-retriever reference (same for every mode) ...
        **{f"shared_{k_}": v for k_, v in retrieval_metrics(case, shared_ranked, k).items()},
        # ... and this mode's own evidence: first k passages, then the final pool
        **{f"mode_{k_}": v for k_, v in retrieval_metrics(case, items, k).items()},
        "pool_recall": pool_recall(case, items),
        "pool_size": len(items),
        # 2. citation relevance (source-aware); not applicable to answers without evidence-ID citations
        **(citation_relevance(case, cited) if record.get("citations") is not None
           else {"cited_precision": None, "fact_evidence_cited": None}),
    }
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _mean(values) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    return round(s[min(len(s) - 1, max(0, math.ceil(p / 100 * len(s)) - 1))], 3)


def aggregate(rows: list[dict]) -> dict:
    """Mean metrics over rows (one row = one case x mode x repeat, metrics merged with record fields)."""
    lat = [r["latency_s"] for r in rows if r.get("latency_s") is not None]
    keys = ["keyword_correct", "correct_abstention", "false_abstention", "citation_ids_valid", "citation_coverage",
            "shared_recall_at_k", "mode_hit_at_k", "mode_recall_at_k", "mode_mrr", "pool_recall", "pool_size",
            "cited_precision", "fact_evidence_cited", "llm_calls", "retrieval_calls", "prompt_tokens",
            "completion_tokens", "revision_count"]
    out = {k: _mean(r.get(k) for r in rows) for k in keys}
    out["forbidden_hit_rate"] = _mean(bool(r.get("forbidden_hits")) for r in rows)
    out["errors"] = sum(1 for r in rows if r.get("error"))
    out["latency_p50_s"] = percentile(lat, 50)
    out["latency_p95_s"] = percentile(lat, 95)
    out["n"] = len(rows)
    return out


def repeat_spread(rows: list[dict], metric: str = "keyword_correct") -> dict:
    by_repeat = defaultdict(list)
    for r in rows:
        if r.get(metric) is not None:
            by_repeat[r["repeat"]].append(float(r[metric]))
    means = [sum(v) / len(v) for v in by_repeat.values() if v]
    return {"per_repeat": [round(m, 4) for m in means],
            "mean": round(statistics.mean(means), 4) if means else None,
            "stdev": round(statistics.stdev(means), 4) if len(means) > 1 else None}


def majority(values: list[bool]) -> bool:
    return sum(values) * 2 > len(values)


def paired_comparison(rows: list[dict], mode_a: str, mode_b: str, metric: str = "keyword_correct") -> dict:
    """Per-case wins/losses/ties using the majority outcome over repeats."""
    per = defaultdict(lambda: defaultdict(list))
    cats = {}
    for r in rows:
        if r["mode"] in (mode_a, mode_b) and r.get(metric) is not None:
            per[r["case_id"]][r["mode"]].append(bool(r[metric]))
            cats[r["case_id"]] = r["category"]
    result = {"wins": [], "losses": [], "ties_both_right": [], "ties_both_wrong": []}
    by_category = defaultdict(Counter)
    for cid, modes in per.items():
        if mode_a not in modes or mode_b not in modes:
            continue
        a, b = majority(modes[mode_a]), majority(modes[mode_b])
        key = "wins" if a and not b else "losses" if b and not a else "ties_both_right" if a else "ties_both_wrong"
        result[key].append(cid)
        by_category[cats[cid]][key] += 1
    n = len(result["wins"]) + len(result["losses"])
    # Exact two-sided sign test on discordant cases; reported for reference only.
    k = min(len(result["wins"]), len(result["losses"]))
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else None
    return {**{key: sorted(v) for key, v in result.items()},
            "discordant": n, "sign_test_p": round(p, 4) if p is not None else None,
            "by_category": {c: dict(v) for c, v in sorted(by_category.items())}}


def verifier_confusion(rows: list[dict], truth_key: str = "keyword_correct") -> dict:
    """Agent verification outcome vs. answer correctness (from `truth_key`).

    verified = grounding "verified"; flagged = unsupported / partially_verified;
    undecided = incomplete / not_verified; abstentions are excluded.
    """
    cells = Counter()
    for r in rows:
        if r.get("mode") != "agentic" or r.get(truth_key) is None or r.get("abstained"):
            continue
        g = (r.get("verification") or {}).get("grounding")
        pred = "verified" if g == "verified" else "flagged" if g in ("unsupported", "partially_verified") else "undecided"
        cells[(pred, "correct" if r[truth_key] else "incorrect")] += 1
    tp, fp = cells[("verified", "correct")], cells[("verified", "incorrect")]
    fn, tn = cells[("flagged", "correct")], cells[("flagged", "incorrect")]
    return {
        "verified_correct": tp, "verified_incorrect (false positive)": fp,
        "flagged_correct (false negative)": fn, "flagged_incorrect": tn,
        "undecided_correct": cells[("undecided", "correct")], "undecided_incorrect": cells[("undecided", "incorrect")],
        "precision_of_verified": round(tp / (tp + fp), 4) if tp + fp else None,
        "truth_source": truth_key,
    }
