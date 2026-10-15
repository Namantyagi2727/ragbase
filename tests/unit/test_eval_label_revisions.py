"""Versioned label revisions: v1.0 stays as approved; v1.1 changes are explicit and reviewable."""

import pytest

from evals import dataset as d
from evals import metrics as m


def _by_id(**kw):
    return {c.id: c for c in d.load_cases(**kw)}


def test_version_1_0_is_the_approved_file_unchanged():
    raw = [d.Case.model_validate_json(line) for line in d.CASES_FILE.read_text().splitlines() if line.strip()]
    assert d.load_cases() == raw == d.load_cases(labels="1.0")
    assert all(f.required for c in raw for f in c.expected_facts)


def test_each_label_version_adds_only_its_own_revisions():
    v10, v11, v12, v13 = _by_id(), _by_id(labels="1.1"), _by_id(labels="1.2"), _by_id(labels="1.3")
    assert [i for i in v10 if v10[i] != v11[i]] == ["FP01"]
    assert sorted(i for i in v11 if v11[i] != v12[i]) == ["C01", "T01", "V05"]
    assert sorted(i for i in v12 if v12[i] != v13[i]) == ["FP01", "V01", "V02", "V03", "V04", "V06"]
    assert all(r.review_status == "approved" for r in d.load_revisions())


def test_optional_facts_keep_their_evidence_for_retrieval_metrics():
    for cid in ("T01", "V05", "C01"):
        before, after = _by_id(labels="1.1")[cid], _by_id(labels="1.2")[cid]
        assert [f.evidence for f in before.expected_facts] == [f.evidence for f in after.expected_facts]
    ranked = [{"source": e.source, "text": e.quote} for f in _by_id(labels="1.2")["V05"].expected_facts
              for e in f.evidence][:1]
    assert m.retrieval_metrics(_by_id(labels="1.2")["V05"], ranked, 5)["recall_at_k"] < 1.0


def test_revised_cases_still_pass_the_corpus_consistency_checks():
    assert d.case_problems(d.load_cases(labels="1.3")) == []


def test_unknown_label_version_is_refused():
    with pytest.raises(ValueError):
        d.load_cases(labels="9.9")


def test_every_revision_has_a_rationale():
    assert all(len(r.rationale) > 40 for r in d.load_revisions())


# --- FP01: active and passive acceptance of the false premise ---------------------

FP01_SHOULD_FLAG = [
    "The 2024 travel policy reduced the domestic meal allowance from $45 to $60.",
    "The allowance was reduced to save costs.",
    "The policy cut the meal allowance.",
    "The 2024 policy lowered the domestic meal allowance from 45 to 60 dollars.",
    "It decreased the allowance from 45 to 60.",
    "The reduction from $45 to $60 was made to simplify claims.",
]
FP01_SHOULD_NOT_FLAG = [
    "The 2024 policy did not reduce the allowance; it increased it from $45 to $60.",
    "The allowance was increased from 45 to 60 dollars, so the premise is wrong.",
    "The allowance rose from 45 to 60 dollars per day.",
    "The 2024 policy raised the domestic meal allowance from 45 to 60 dollars [1].",
    "There was no reduction; it went up to 60 dollars.",
    "The question assumes a reduction, but the allowance actually increased to 60 dollars.",
]


def _fp01_patterns(labels):
    case = _by_id(labels=labels)["FP01"]
    return [(f.statement, f.pattern) for f in case.forbidden_facts]


@pytest.mark.parametrize("sentence", FP01_SHOULD_FLAG)
def test_fp01_revised_pattern_flags_acceptance_of_the_false_premise(sentence):
    assert m.forbidden_hits(_fp01_patterns("1.1"), sentence)


@pytest.mark.parametrize("sentence", FP01_SHOULD_NOT_FLAG)
def test_fp01_revised_pattern_does_not_flag_correct_answers(sentence):
    assert not m.forbidden_hits(_fp01_patterns("1.1"), sentence)


def test_fp01_v10_pattern_missed_the_active_voice_answer_seen_in_the_benchmark():
    answer = "The 2024 travel policy reduced the domestic meal allowance from $45 to $60."
    assert not m.forbidden_hits(_fp01_patterns("1.0"), answer)
    assert m.forbidden_hits(_fp01_patterns("1.1"), answer)


# --- optional facts ------------------------------------------------------------------

def test_optional_facts_are_reported_but_not_needed_for_correctness():
    case = _by_id(labels="1.2")["V05"]
    s = m.score_record(case, {"answer": "The current rate is 67 cents per mile [1]."}, [], 5)
    assert s["keyword_correct"] and (s["facts_covered"], s["facts_total"]) == (1, 2)
    v10 = m.score_record(_by_id()["V05"], {"answer": "The current rate is 67 cents per mile [1]."}, [], 5)
    assert not v10["keyword_correct"]


def test_superseded_value_still_fails_after_facts_become_optional():
    case = _by_id(labels="1.2")["V05"]
    assert not m.score_record(case, {"answer": "The rate is 58 cents per mile [1]."}, [], 5)["keyword_correct"]


def test_t01_synonym_accepts_no_approval_but_not_a_wrong_threshold():
    case = _by_id(labels="1.2")["T01"]
    ok = "Stays up to 250 dollars per night do not need pre-approval, so a 230-dollar night does not require approval [1]."
    assert m.score_record(case, {"answer": ok}, [], 5)["keyword_correct"]
    wrong = "Stays above 200 dollars need a manager's approval, so it needs approval [1]."
    assert not m.score_record(case, {"answer": wrong}, [], 5)["keyword_correct"]

