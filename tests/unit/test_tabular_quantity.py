"""Quantity questions must total a numeric column, not count rows.

Regression: with llama3.2, "How many units were sold in the EU?" was planned
as count(units) -> 2 rows instead of sum(units) -> 15.
"""

import json

import pandas as pd
import pytest

import tabular as tb
from llm.fake import FakeChatModel

EU = [{"column": "region", "operator": "==", "value": "EU"}]


@pytest.fixture
def tables():
    df = pd.DataFrame({
        "region": ["EU", "US", "EU", "APAC"],
        "units": [10, 20, 5, 7],
        "revenue": [25.0, 60.0, 20.0, 7.0],
        "customer_id": [101, 102, 103, 104],
    })
    return [tb.Table("sales.csv", "sales.csv", None, df)]


def _ask(question, plan, tables):
    return tb.answer_tabular_question(question, tables, FakeChatModel([json.dumps(plan)]))


def test_regression_how_many_units_sums_instead_of_counting(tables):
    resp = _ask("How many units were sold in the EU?",
                {"operation": "count", "column": "units", "filters": EU}, tables)
    assert resp.ok
    assert resp.query["operation"] == "sum" and resp.query["column"] == "units"
    assert "15" in resp.answer.split("\n\n")[1]
    assert resp.notes and "sum(units)" in resp.answer  # the correction is shown, not silent


def test_count_without_column_is_corrected_when_question_names_quantity(tables):
    resp = _ask("How many units were sold in the EU?", {"operation": "count", "filters": EU}, tables)
    assert resp.query["operation"] == "sum" and "15" in resp.answer


@pytest.mark.parametrize("question", [
    "What is the total revenue in the EU?",
    "What is the number of units sold in the EU?",
    "How many total units were sold in the EU?",
])
def test_other_quantity_phrasings(question, tables):
    plan = {"operation": "count", "filters": EU}
    resp = _ask(question, plan, tables)
    assert resp.query["operation"] == "sum"


def test_row_count_phrasing_stays_a_count(tables):
    plan = {"operation": "count", "filters": [{"column": "units", "operator": ">", "value": 6}]}
    resp = _ask("How many rows have units above 6?", plan, tables)
    assert resp.query["operation"] == "count" and resp.rows_matched == 3 and not resp.notes


def test_question_without_a_quantity_column_stays_a_count(tables):
    resp = _ask("How many sales were made in the EU?", {"operation": "count", "filters": EU}, tables)
    assert resp.query["operation"] == "count" and "\n\n2" in resp.answer


def test_identifier_columns_are_not_summed(tables):
    resp = _ask("How many customer ids are there?", {"operation": "count", "column": "customer_id"}, tables)
    assert resp.query["operation"] == "count"


def test_ambiguous_quantity_question_is_left_unchanged(tables):
    resp = _ask("How many units and how much revenue in the EU?", {"operation": "count", "filters": EU}, tables)
    assert resp.query["operation"] == "count"


def test_count_of_a_different_column_is_not_rewritten(tables):
    resp = _ask("How many units were sold?", {"operation": "count", "column": "region"}, tables)
    assert resp.query == {**resp.query, "operation": "count", "column": "region"}


def test_sum_plan_is_untouched(tables):
    resp = _ask("How many units were sold in the EU?",
                {"operation": "sum", "column": "units", "filters": EU}, tables)
    assert resp.query["operation"] == "sum" and not resp.notes
