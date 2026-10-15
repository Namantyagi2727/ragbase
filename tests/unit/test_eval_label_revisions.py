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


def test_approved_revisions_apply_and_proposed_ones_only_on_request():
    v10, v11, v11p = _by_id(), _by_id(labels="1.1"), _by_id(labels="1.1", include_proposed=True)
    assert [i for i in v10 if v10[i] != v11[i]] == ["FP01"]
    assert sorted(i for i in v10 if v10[i] != v11p[i]) == ["C01", "FP01", "T01", "V05"]
    assert {r.case_id: r.review_status for r in d.load_revisions()} == {
        "FP01": "approved", "T01": "proposed", "V05": "proposed", "C01": "proposed"}


def test_revised_cases_still_pass_the_corpus_consistency_checks():
    assert d.case_problems(d.load_cases(labels="1.1", include_proposed=True)) == []


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
    case = _by_id(labels="1.1", include_proposed=True)["V05"]
    s = m.score_record(case, {"answer": "The current rate is 67 cents per mile [1]."}, [], 5)
    assert s["keyword_correct"] and (s["facts_covered"], s["facts_total"]) == (1, 2)
    v10 = m.score_record(_by_id()["V05"], {"answer": "The current rate is 67 cents per mile [1]."}, [], 5)
    assert not v10["keyword_correct"]


def test_superseded_value_still_fails_after_facts_become_optional():
    case = _by_id(labels="1.1", include_proposed=True)["V05"]
    assert not m.score_record(case, {"answer": "The rate is 58 cents per mile [1]."}, [], 5)["keyword_correct"]


def test_t01_synonym_accepts_no_approval_but_not_a_wrong_threshold():
    case = _by_id(labels="1.1", include_proposed=True)["T01"]
    ok = "Stays up to 250 dollars per night do not need pre-approval, so a 230-dollar night does not require approval [1]."
    assert m.score_record(case, {"answer": ok}, [], 5)["keyword_correct"]
    wrong = "Stays above 200 dollars need a manager's approval, so it needs approval [1]."
    assert not m.score_record(case, {"answer": wrong}, [], 5)["keyword_correct"]

