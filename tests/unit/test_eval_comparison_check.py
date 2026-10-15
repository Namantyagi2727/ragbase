"""Comparison-consistency check (scoring 1.2, labels 1.3) — regression for benchmark case V06.

A narrow deterministic check; passing it does not show an answer is semantically correct.
"""

import pytest

from evals import metrics as m
from evals.dataset import ComparisonSpec, load_cases

INC = ComparisonSpec(old="75", new="90", direction="increase")
DEC = ComparisonSpec(old="30", new="15", direction="decrease")
HOURS = ComparisonSpec(old="6", new="8", direction="increase")


@pytest.mark.parametrize("spec,answer", [
    (INC, "The allowance increased from 75 dollars to 90 dollars [1]."),
    (INC, "It rose from $75 to $90 per day."),
    (INC, "The 2024 policy sets 90 dollars, while the 2023 policy set 75 dollars."),
    (DEC, "The deadline was shortened from 30 days to 15 days [2]."),
    (DEC, "Reports are now due within 15 days instead of 30."),
    (HOURS, "Business class now needs flights of eight hours, up from six hours."),
    (HOURS, "The threshold changed from six hours to eight hours."),
    (INC, "It increased from 75 to 90; it did not decrease."),        # negated wrong direction
    (INC, "It changed from 75 to 90, not from 75 dollars to 60 dollars."),  # negated wrong pair
])
def test_consistent_comparisons_pass(spec, answer):
    assert m.comparison_inconsistencies(spec, answer) == []


@pytest.mark.parametrize("spec,answer,kind", [
    (INC, "The international daily meal allowance increased from 75 dollars to 60 dollars [1].", "wrong new value"),
    (INC, "The allowance went from 90 dollars to 75 dollars.", "reversed"),
    (INC, "The allowance was reduced to 90 dollars.", "opposite direction"),
    (INC, "It rose from 60 dollars to 90 dollars.", "wrong old value"),
    (DEC, "The deadline was extended from 30 days to 15 days.", "opposite direction"),
    (DEC, "The deadline changed from 15 days to 30 days.", "reversed"),
    (HOURS, "The threshold dropped from six hours to eight hours.", "opposite direction"),
])
def test_inconsistent_comparisons_are_flagged(spec, answer, kind):
    problems = m.comparison_inconsistencies(spec, answer)
    assert problems and any(kind in p for p in problems)


def test_v06_answer_from_the_benchmark_fails_only_under_the_revised_scoring():
    v06 = ("According to the evidence, the international daily meal allowance increased from 75 dollars to 60 "
           "dollars between the 2023 and 2024 policies, not from 75 dollars to 90 dollars as previously stated.")
    v12, v13 = ({c.id: c for c in load_cases(labels=v)}["V06"] for v in ("1.2", "1.3"))
    assert m.score_record(v12, {"answer": v06}, [], 5)["keyword_correct"]            # the old weakness
    s = m.score_record(v13, {"answer": v06}, [], 5)
    assert not s["keyword_correct"] and s["comparison_inconsistencies"] and not s["forbidden_hits"]
    assert m.score_record(v13, {"answer": v06}, [], 5, scoring="1.1")["keyword_correct"]   # 1.1 ignores the spec


def test_years_in_from_to_phrases_are_not_flagged():
    spec = ComparisonSpec(old="45", new="60", direction="increase")
    assert m.comparison_inconsistencies(spec, "From 2023 to 2024 the allowance rose from 45 to 60 dollars.") == []


def test_scoring_1_2_rederives_abstention_from_the_answer_text():
    case = {c.id: c for c in load_cases()}["M03"]
    answer = ("The primary caregiver receives 16 weeks of fully paid parental leave, but the evidence does not "
              "specify more. The company pays 50 percent of the premium for dependants.")
    rec = {"answer": answer, "abstained": True}   # as recorded by the earlier classifier
    assert not m.score_record(case, rec, [], 5, scoring="1.1")["keyword_correct"]
    s = m.score_record(case, rec, [], 5, scoring="1.2")
    assert not s["abstained"] and s["keyword_correct"]
