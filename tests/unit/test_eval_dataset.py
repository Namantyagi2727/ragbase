"""Benchmark dataset consistency (runs in CI).

These checks confirm the labels are internally consistent with the corpus —
sources exist, quotes appear in the extracted text, computed values match the
data. They do not establish that a proposed answer is correct; that requires
human review (review_status "approved").
"""

from collections import Counter
from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

import ingest
import retriever as r
import tabular as tb
from citations import split_claims
from evals import build_corpus
from evals import dataset as d


@pytest.fixture(scope="module")
def cases():
    return d.load_cases()


def test_cases_load_with_expected_size_and_categories(cases):
    assert 40 <= len(cases) <= 44
    counts = Counter(c.category for c in cases)
    assert set(counts) == set(d.Category.__args__)
    assert counts["unanswerable"] >= 5 and counts["regression"] >= 2


def test_review_status_values_are_valid(cases):
    assert {c.review_status for c in cases} <= {"proposed", "approved", "rejected"}


def test_cases_are_consistent_with_the_corpus(cases):
    assert d.case_problems(cases) == []


def test_required_regression_cases_are_present(cases):
    required = {c.id for c in cases if c.required}
    assert {"R01", "R02", "R03"} <= required
    r01 = next(c for c in cases if c.id == "R01")
    assert r01.forbidden_facts and "0?\\.08" in r01.forbidden_facts[0].pattern


def test_corpus_matches_manifest_and_rebuilds_identically(tmp_path):
    assert d.manifest_problems() == []
    rebuilt = build_corpus.build(tmp_path / "corpus", tmp_path / "tabular")
    assert rebuilt["files"] == d.corpus_manifest()["files"]


def test_two_column_handbook_yields_separate_age_passages_after_chunking(tmp_path):
    src = tmp_path / "docs"
    src.mkdir()
    (src / "driver_safety_handbook.pdf").write_bytes((d.CORPUS_DIR / "driver_safety_handbook.pdf").read_bytes())
    ingest.ingest_documents(ingest_dir=str(src), index_dir=str(tmp_path / "idx"),
                            embed_model="fake", embeddings=DeterministicFakeEmbedding(size=16))
    chunks = [d.normalize(c["text"]) for c in r.load_tfidf(str(tmp_path / "idx"))[2]]
    over = d.normalize("A driver aged 21 or over commits an offence if their blood alcohol concentration is 0.08 percent or higher.")
    under = d.normalize("A blood alcohol concentration from 0.02 to 0.07 percent leads to a six month licence suspension.")
    assert any(over in c for c in chunks) and any(under in c for c in chunks)
    assert not any("under age 21" in c and "aged 21 or over commits" in c and c.index("under age 21") < c.index("commits")
                   for c in chunks), "columns interleaved"


def test_verifier_challenge_claims_split_exactly_as_labelled():
    vcs = d.load_verifier_cases()
    assert len(vcs) >= 18
    for v in vcs:
        assert split_claims(v.answer) == [c.text for c in v.claims], v.id
        assert [e.id for e in v.evidence] == [str(i) for i in range(1, len(v.evidence) + 1)], v.id
    assert {"VC01", "VC02", "VC03"} <= {v.id for v in vcs if v.required}
    assert {v.answer_label for v in vcs} == {"correct", "incorrect", "correct_fact_wrong_citation"}


def test_answer_labels_are_consistent_with_claim_labels():
    for v in d.load_verifier_cases():
        all_true = all(c.factually_true for c in v.claims)
        all_supported = all(c.label == "supported" for c in v.claims)
        if v.answer_label == "correct":
            assert all_true and all_supported, v.id
        elif v.answer_label == "correct_fact_wrong_citation":
            assert all_true and not all_supported, v.id
        else:
            assert not all_true, v.id


def test_subset_truth_is_distinguished_from_false_maximum():
    """VC04/VC21 regression: 'up to 200 is auto-approved' is true when the limit is 250;
    'the maximum auto-approved rate is 200' is false. Labels must reflect that."""
    vcs = {v.id: v for v in d.load_verifier_cases()}
    false_max, true_subset = vcs["VC04"].claims[0], vcs["VC21"].claims[0]
    assert "maximum" in false_max.text and false_max.label == "unsupported" and not false_max.factually_true
    assert vcs["VC04"].answer_label == "incorrect"
    assert true_subset.label == "supported" and true_subset.factually_true
    assert vcs["VC21"].answer_label == "correct"
    assert vcs["VC04"].evidence == vcs["VC21"].evidence


def test_correct_fact_with_wrong_citation_is_labelled_separately():
    vc15 = next(v for v in d.load_verifier_cases() if v.id == "VC15")
    claim = vc15.claims[0]
    assert claim.factually_true and claim.label == "unsupported"
    assert vc15.answer_label == "correct_fact_wrong_citation"


def test_every_evidence_quote_sits_inside_one_indexed_chunk(cases):
    assert d.chunk_problems(cases, d.indexed_chunks()) == []


def test_under_21_evidence_states_the_age_group_itself(cases):
    for cid in ("R01", "R03"):
        case = next(c for c in cases if c.id == cid)
        for f in case.expected_facts:
            assert all("under age 21" in e.quote for e in f.evidence), cid


def test_tabular_expected_values_match_the_data():
    tables = tb.load_tables(str(d.TABULAR_DIR))
    for c in d.load_tabular_cases():
        if c.expected_operation is None:
            continue
        q = tb.TabularQuery(operation=c.expected_operation, column=c.expected_column,
                            filters=[tb.TableFilter(**f.model_dump()) for f in c.expected_filters],
                            group_by=c.expected_group_by)
        rendered = tb.execute_query(q, tables).answer.split("\n\n", 1)[1].strip()
        assert rendered == c.expected_value, c.id


def test_review_materials_are_up_to_date():
    from evals import review_export

    for path, content in review_export.render_all().items():
        assert Path(path).read_text() == content, f"{path} is stale; run python -m evals.review_export"


@pytest.mark.parametrize("case_id,sentence,should_match", [
    ("R01", "Drivers under 21 have a blood alcohol limit of 0.08 percent [1].", True),
    ("R01", "The 0.08 percent limit applies to drivers 21 and over, while drivers under 21 face zero tolerance [1].", False),
    ("R01", "For drivers under age 21, a BAC from 0.02 to 0.07 percent leads to a six month suspension [1].", False),
    ("R02", "A driver aged 21 or over commits an offence at 0.02 percent [1].", True),
    ("R02", "Drivers aged 21 or over commit an offence at 0.08 percent, unlike drivers under 21 at 0.02 [1].", False),
    ("F01", "Under the 2024 policy the domestic allowance is 45 dollars [1].", True),
    ("F01", "The 2023 allowance was 45 dollars, and the 2024 allowance is 60 dollars [1].", False),
    ("V05", "The current mileage rate is 58 cents per mile [1].", True),
    ("V05", "The rate rose from the previous 58 cents to 67 cents per mile [1].", False),
    ("C02", "Employees still in probation may work remotely three days per week [1].", True),
    ("C02", "After probation the limit is three days; in probation it stays at two days [1].", False),
    ("T01", "A manager must approve the stay before booking [1].", True),
])
def test_forbidden_patterns_flag_wrong_claims_but_not_correct_contrasts(cases, case_id, sentence, should_match):
    import re

    case = next(c for c in cases if c.id == case_id)
    matched = any(re.search(f.pattern, sentence.lower()) for f in case.forbidden_facts)
    assert matched is should_match
