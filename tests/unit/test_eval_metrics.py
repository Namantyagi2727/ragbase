"""Deterministic evaluation metrics."""

import pytest

from evals import metrics as m
from evals.dataset import Case


def _case(**kw):
    base = dict(id="X", category="direct_factual", question="q", answerable="yes", reference_answer="r",
                expected_facts=[{"id": "f1", "statement": "s", "keywords": [["60", "sixty"], ["meal"]],
                                 "evidence": [{"source": "a.txt", "quote": "The meal allowance is 60 dollars."}]}])
    base.update(kw)
    return Case.model_validate(base)


@pytest.mark.parametrize("text,group,expected", [
    ("The allowance is 60 dollars [1].", ["60"], True),
    ("Updated in 2024.", ["2"], False),               # no digit-prefix false match
    ("It is 600 dollars.", ["60"], False),
    ("Revocation for one year.", ["revo"], True),      # word-prefix match allowed
    ("A BAC of .02 percent", ["0.02", ".02"], True),
    ("The limit is $250 per night", ["250"], True),
])
def test_keyword_matching_boundaries(text, group, expected):
    assert m.keyword_group_matched(group, text) is expected


def test_citation_markers_do_not_satisfy_number_keywords():
    assert not m.keyword_group_matched(["2"], "Something is stated [2].")


def test_forbidden_skips_negated_sentences():
    pats = [("wrong 45", r"\b45 dollars")]
    assert m.forbidden_hits(pats, "The allowance is 45 dollars.") == ["wrong 45"]
    assert m.forbidden_hits(pats, "It is not 45 dollars. It is 60 dollars.") == []
    assert m.forbidden_hits(pats, "Rather than 45 dollars, it is now 60.") == []


def test_score_answerable_correct_and_wrong():
    c = _case(forbidden_facts=[{"statement": "45 as current", "pattern": r"\b45 dollars"}])
    ok = m.score_record(c, {"answer": "The meal allowance is 60 dollars [1]."}, [], 5)
    bad = m.score_record(c, {"answer": "The meal allowance is 45 dollars, previously 60 [1]."}, [], 5)
    assert ok["keyword_correct"] and not bad["keyword_correct"] and bad["forbidden_hits"]


def test_score_unanswerable_requires_abstention():
    c = _case(answerable="no", expected_facts=[], missing_information=["x"])
    assert m.score_record(c, {"answer": "I don't have enough information in the provided documents to answer that.",
                              "abstained": True}, [], 5)["keyword_correct"]
    assert not m.score_record(c, {"answer": "Pets are welcome on Fridays [1]."}, [], 5)["keyword_correct"]


def test_score_partial_requires_fact_and_missing_acknowledgement():
    c = _case(answerable="partial", missing_information=["vesting"])
    full = "The meal allowance is 60 dollars [1]. The documents do not mention a vesting schedule."
    assert m.score_record(c, {"answer": full}, [], 5)["keyword_correct"]
    assert not m.score_record(c, {"answer": "The meal allowance is 60 dollars [1]."}, [], 5)["keyword_correct"]


def test_errors_are_never_correct():
    c = _case()
    assert not m.score_record(c, {"answer": "The meal allowance is 60 dollars.", "error": "boom"}, [], 5)["keyword_correct"]


def test_retrieval_metrics_fixed_budget():
    c = _case()
    ranked = [{"source": "a.txt", "text": "other text"},
              {"source": "a.txt", "text": "Policy: The meal allowance is 60 dollars. More."},
              {"source": "a.txt", "text": "x"}]
    r = m.retrieval_metrics(c, ranked, k=5)
    assert r == {"hit_at_k": 1.0, "recall_at_k": 1.0, "mrr": 0.5}
    assert m.retrieval_metrics(c, ranked, k=1)["hit_at_k"] == 0.0


def test_citation_relevance_is_separate_from_validity():
    c = _case()
    rel = m.citation_relevance(c, [{"source": "a.txt", "text": "Unrelated passage about hotels."}])
    assert rel == {"cited_precision": 0.0, "fact_evidence_cited": 0.0}
    rel = m.citation_relevance(c, [{"source": "a.txt", "text": "The meal allowance is 60 dollars."},
                                   {"source": "a.txt", "text": "Other."}])
    assert rel == {"cited_precision": 0.5, "fact_evidence_cited": 1.0}


def test_paired_comparison_uses_majority_over_repeats():
    rows = []
    for rep in (1, 2, 3):
        rows += [{"case_id": "A", "category": "c", "mode": "standard", "repeat": rep, "keyword_correct": rep != 3},
                 {"case_id": "A", "category": "c", "mode": "agentic", "repeat": rep, "keyword_correct": False},
                 {"case_id": "B", "category": "c", "mode": "standard", "repeat": rep, "keyword_correct": True},
                 {"case_id": "B", "category": "c", "mode": "agentic", "repeat": rep, "keyword_correct": True}]
    pc = m.paired_comparison(rows, "standard", "agentic")
    assert pc["wins"] == ["A"] and pc["ties_both_right"] == ["B"] and pc["discordant"] == 1


def test_verifier_confusion_counts_false_positives():
    rows = [
        {"mode": "agentic", "keyword_correct": False, "verification": {"grounding": "verified"}},
        {"mode": "agentic", "keyword_correct": True, "verification": {"grounding": "verified"}},
        {"mode": "agentic", "keyword_correct": True, "verification": {"grounding": "unsupported"}},
        {"mode": "agentic", "keyword_correct": True, "verification": {"grounding": "incomplete"}},
        {"mode": "standard", "keyword_correct": False, "verification": None},
    ]
    cm = m.verifier_confusion(rows)
    assert cm["verified_incorrect (false positive)"] == 1 and cm["flagged_correct (false negative)"] == 1
    assert cm["undecided_correct"] == 1 and cm["precision_of_verified"] == 0.5


def test_repeat_spread():
    rows = [{"repeat": 1, "keyword_correct": True}, {"repeat": 1, "keyword_correct": False},
            {"repeat": 2, "keyword_correct": True}, {"repeat": 2, "keyword_correct": True}]
    s = m.repeat_spread(rows)
    assert s["per_repeat"] == [0.5, 1.0] and s["mean"] == 0.75


# --- source-aware matching and mode-specific retrieval (regressions) ----------

QUOTE = "Expense reports must be submitted within 30 days."


def _sourced_case():
    return _case(expected_facts=[{"id": "f1", "statement": "30 days", "keywords": [["30"]],
                                  "evidence": [{"source": "policy_a.txt", "quote": QUOTE}]}])


def test_identical_quote_from_the_wrong_source_gets_no_retrieval_credit():
    c = _sourced_case()
    wrong = [{"source": "policy_b.txt", "text": f"Rules. {QUOTE} More."}]
    right = [{"source": "policy_a.txt", "text": f"Rules. {QUOTE} More."}]
    assert m.retrieval_metrics(c, wrong, k=5)["recall_at_k"] == 0.0
    assert m.retrieval_metrics(c, right, k=5)["recall_at_k"] == 1.0
    assert m.pool_recall(c, wrong) == 0.0 and m.pool_recall(c, right) == 1.0


def test_identical_quote_from_the_wrong_source_gets_no_citation_credit():
    c = _sourced_case()
    rel = m.citation_relevance(c, [{"source": "policy_b.txt", "text": QUOTE}])
    assert rel == {"cited_precision": 0.0, "fact_evidence_cited": 0.0}
    rel = m.citation_relevance(c, [{"source": "policy_a.txt", "text": QUOTE}])
    assert rel == {"cited_precision": 1.0, "fact_evidence_cited": 1.0}


def test_mode_specific_retrieval_uses_each_modes_own_evidence():
    c = _sourced_case()
    shared = [{"source": "policy_a.txt", "text": QUOTE}]
    found = {"answer": "Within 30 days [1].", "citations": {"ok": True}, "evidence": [
        {"eid": "1", "source": "policy_a.txt", "text": QUOTE}], "cited_ids": ["1"]}
    missed = {"answer": "Within 30 days.", "evidence": [
        {"eid": "1", "source": "policy_b.txt", "text": QUOTE},
        {"eid": "2", "source": "policy_a.txt", "text": "Unrelated text."}], "cited_ids": []}
    a, b = m.score_record(c, found, shared, 5), m.score_record(c, missed, shared, 5)
    assert a["shared_recall_at_k"] == b["shared_recall_at_k"] == 1.0   # same reference for both
    assert a["mode_recall_at_k"] == 1.0 and b["mode_recall_at_k"] == 0.0
    assert a["mode_hit_at_k"] == 1.0 and b["mode_hit_at_k"] == 0.0


def test_mode_fixed_budget_counts_only_the_first_k_passages():
    c = _sourced_case()
    late = {"answer": "x", "cited_ids": [], "evidence": [
        {"eid": str(i), "source": "other.txt", "text": f"filler {i}"} for i in range(1, 6)
    ] + [{"eid": "6", "source": "policy_a.txt", "text": QUOTE}]}
    s = m.score_record(c, late, [], 5)
    assert s["mode_recall_at_k"] == 0.0     # outside the fixed budget of 5
    assert s["pool_recall"] == 1.0          # but recovered in the final pool
    assert s["pool_size"] == 6


def test_citation_relevance_is_not_applicable_without_evidence_id_citations():
    c = _sourced_case()
    original_style = {"answer": "Within 30 days. Sources: [policy @ p.1]", "citations": None,
                      "evidence": [{"eid": "1", "source": "policy_a.txt", "text": QUOTE}], "cited_ids": []}
    s = m.score_record(c, original_style, [], 5)
    assert s["cited_precision"] is None and s["fact_evidence_cited"] is None
    assert s["mode_recall_at_k"] == 1.0     # its own retrieval is still scored
