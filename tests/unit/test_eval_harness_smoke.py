"""Evaluation harness smoke test — MOCKED.

Runs the real runner, scorer, report, verifier evaluator and judge plumbing with
a scripted model and fake embeddings. It checks that artifacts are produced and
consistent; the numbers it produces say nothing about answer quality.
"""

import json
import re

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from citations import ABSTAIN_TEXT
from evals import judge, report, runner, verifier_eval
from evals.__main__ import main as cli
from evals.dataset import load_cases, load_verifier_cases
from llm.base import LLMResult

_DOC_RE = re.compile(r'<doc source="[^"]*" location="[^"]*" id="(\d+)">\n(.*?)\n</doc>', re.S)


class ScriptedModel:
    """Answers each prompt type with a plausible, deterministic reply."""

    backend, model, seed = "fake", "scripted", None

    def complete(self, messages, *, json_schema=None, temperature=None, timeout_s=None):
        system, user = messages[0]["content"], messages[-1]["content"]
        docs = _DOC_RE.findall(user)
        if "You plan how to answer" in system:
            text = json.dumps({"route": "complex", "subqueries": [{"query": "policy"}]} if " and " in user
                              else {"route": "simple"})
        elif "You check whether an answer" in system:
            n = len(re.findall(r"^\d+\. ", user.split("Claims to check:")[-1], re.M))
            text = json.dumps({"verdicts": [{"claim_number": i, "status": "supported", "evidence_ids": ["1"]}
                                            for i in range(1, n + 1)], "recommendation": "accept"})
        elif "An assistant said it could not answer" in system:
            text = json.dumps({"answerable": "no"})
        elif "You grade an answer" in system:
            text = json.dumps({"correctness": 4, "faithfulness": 4, "completeness": 4,
                               "abstention": "not_applicable", "passed": True, "issues": []})
        elif "pets" in user.lower() or not docs:
            text = ABSTAIN_TEXT
        else:
            first = docs[0][1].split(". ")[0].strip().rstrip(".")
            text = f"{first} [1]."
        return LLMResult(text=text, model=self.model, backend=self.backend, is_mock=True,
                         prompt_tokens=10, completion_tokens=5)


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("smoke_run")
    cfg = runner.EvalConfig(out_dir=str(out), modes=["standard", "agentic"], repeats=2,
                            case_ids=["F02", "U01", "M01", "R01"], reranking=False, embed_model="fake",
                            label="development")
    runner.run(cfg, model_factory=lambda repeat: ScriptedModel(),
               embeddings=DeterministicFakeEmbedding(size=32), progress=lambda _: None)
    return out


def test_run_writes_records_config_and_fixed_retrieval(run_dir):
    records = [json.loads(line) for line in (run_dir / "records.jsonl").read_text().splitlines()]
    assert len(records) == 4 * 2 * 2  # cases x modes x repeats
    assert {r["mode"] for r in records} == {"standard", "agentic"}
    meta = json.loads((run_dir / "config.json").read_text())
    assert meta["config"]["label"] == "development"
    assert meta["environment"]["corpus_checksum"] and meta["environment"]["git_commit"]
    assert meta["dataset_review_status"] == ["approved"]
    assert len((run_dir / "fixed_retrieval.jsonl").read_text().splitlines()) == 4


def test_rerun_resumes_without_duplicating_records(run_dir):
    cfg = runner.EvalConfig(out_dir=str(run_dir), modes=["standard", "agentic"], repeats=2,
                            case_ids=["F02", "U01", "M01", "R01"], reranking=False, embed_model="fake")
    runner.run(cfg, model_factory=lambda repeat: ScriptedModel(),
               embeddings=DeterministicFakeEmbedding(size=32), progress=lambda _: None)
    assert len((run_dir / "records.jsonl").read_text().splitlines()) == 16


def test_report_scores_and_labels_the_run(run_dir):
    out = report.build(run_dir)
    md = (run_dir / "report.md").read_text()
    assert "Not an official benchmark result" in md and "keyword proxy" in md
    assert {"standard", "agentic"} <= set(out["summary"])
    assert "agentic_vs_standard" in out["comparisons"]
    assert out["agent"]["routes"]  # routes recorded
    rows = [json.loads(line) for line in (run_dir / "scored_rows.jsonl").read_text().splitlines()]
    u01 = [r for r in rows if r["case_id"] == "U01"]
    assert all(r["abstained"] and r["keyword_correct"] for r in u01)
    assert (run_dir / "summary.csv").read_text().startswith("mode,n,keyword_correct")


def test_judge_is_blind_and_bounded(run_dir):
    out = judge.run(run_dir, ScriptedModel(), max_items=3, judgments=2, progress=lambda _: None)
    assert out["items"] == 3 and out["judgments_per_item"] == 2 and out["self_consistency"] == 1.0
    lines = (run_dir / "judge.jsonl").read_text()
    assert '"mode"' not in lines  # mode lives only in the separate mapping
    assert set(json.loads((run_dir / "judge_items.json").read_text())) == {"J001", "J002", "J003"}
    est = judge.run(run_dir, ScriptedModel(), max_items=0, estimate_only=True, progress=lambda _: None)
    assert est["estimate"]["items_selected"] == 8


def test_verifier_challenge_plumbing():
    out = verifier_eval.run(load_verifier_cases()[:4], ScriptedModel())
    s = out["summary"]["claims"]
    assert sum(v for k, v in s.items() if not k.endswith("rate")) == 4
    assert out["summary"]["n_runs"] == 4


def test_official_run_is_refused_when_any_selected_case_is_unapproved(tmp_path, capsys, monkeypatch):
    from evals import dataset

    cases = load_cases()
    cases[0] = cases[0].model_copy(update={"review_status": "proposed"})
    monkeypatch.setattr(dataset, "load_cases", lambda *a, **k: cases)
    code = cli(["run", "--out", str(tmp_path / "x"), "--label", "official", "--repeats", "1",
                "--cases", f"{cases[0].id},{cases[1].id}"])
    assert code == 2 and "Refusing an official run" in capsys.readouterr().err


def test_official_run_passes_the_gate_when_selected_cases_are_approved(tmp_path, capsys, monkeypatch):
    assert all(c.review_status == "approved" for c in load_cases())
    monkeypatch.setattr(runner, "preflight", lambda cfg: "stopped after the approval gate (test)")
    code = cli(["run", "--out", str(tmp_path / "y"), "--label", "official", "--repeats", "1", "--cases", "F01"])
    err = capsys.readouterr().err
    assert code == 3 and "Refusing an official run" not in err and "stopped after the approval gate" in err


def test_preflight_reports_unreachable_ollama(monkeypatch):
    import httpx

    def down(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(runner.httpx, "get", down)
    problem = runner.preflight(runner.EvalConfig(out_dir="x"))
    assert problem and "not reachable" in problem


def test_report_to_a_separate_directory_leaves_the_run_untouched(run_dir, tmp_path):
    before = {p.name: p.read_bytes() for p in run_dir.iterdir() if p.is_file()}
    out = report.build(run_dir, tmp_path / "rescored", labels="1.1", include_proposed=True)
    assert {p.name: p.read_bytes() for p in run_dir.iterdir() if p.is_file()} == before
    assert (tmp_path / "rescored" / "report.md").exists()
    assert out["meta"]["scoring"]["labels"] == "1.1" and out["meta"]["scoring"]["include_proposed_revisions"]
    assert "proposed" in (tmp_path / "rescored" / "report.md").read_text()


def test_unique_claim_summary_counts_each_claim_once_across_repeats():
    results = [{"id": "A", "claims": [{"claim": "x", "expected": "accept", "predicted": "accepted"},
                                      {"claim": "y", "expected": "reject", "predicted": p}]}
               for p in ("accepted", "rejected", "accepted")]
    u = verifier_eval.summarize_unique(results)
    assert u["unique_claims"] == 2 and u["accepted_supported"] == 1 and u["inconsistent_unsupported"] == 1
    assert u["false_positive_rate"] == "0/1"


@pytest.mark.parametrize("scores,abstention,expected", [
    ((5, 5, 5), "not_applicable", True),
    ((5, 5, 3), "not_applicable", False),      # incomplete answers cannot pass on correct claims alone
    ((5, 5, 1), "inappropriate", False),
    ((3, 5, 5), "not_applicable", False),
    ((5, 3, 5), "not_applicable", False),
    ((5, 5, 5), "appropriate", True),
])
def test_rubric_v2_pass_rule_requires_every_dimension(scores, abstention, expected):
    c, f, comp = scores
    assert judge.passes({"correctness": c, "faithfulness": f, "completeness": comp, "abstention": abstention},
                        "2") is expected


def test_rubric_v2_prompt_carries_expected_answerability_and_reference_facts():
    case = next(c for c in load_cases() if c.id == "V02")
    msgs = judge.judge_messages(case, "answer", ["evidence"], rubric="2")
    assert "Expected answerability: yes" in msgs[1]["content"] and "director" in msgs[1]["content"]
    assert "not with the retrieved evidence" in msgs[0]["content"]


def test_judge_writes_to_a_separate_directory(run_dir, tmp_path):
    before = {p.name: p.read_bytes() for p in run_dir.iterdir() if p.is_file()}
    out = judge.run(run_dir, ScriptedModel(), max_items=2, out_dir=tmp_path / "j", progress=lambda _: None)
    assert out["rubric"] == "3" and (tmp_path / "j" / "judge_summary.json").exists()
    assert {p.name: p.read_bytes() for p in run_dir.iterdir() if p.is_file()} == before


def test_calibration_plumbing_reports_false_passes_and_fails():
    out = judge.calibrate(ScriptedModel(), progress=lambda _: None)
    assert out["items"] == 22 and out["failed_judgments"] == 0
    # the scripted judge passes everything, so every known-incorrect item is a false pass
    expected_fail = [r["id"] for r in out["results"] if not r["expected_pass"]]
    assert out["false_passes"] == expected_fail and not out["false_fails"]


def test_compare_scores_all_runs_with_one_scoring_version(run_dir, tmp_path):
    from evals import compare

    before = {p.name: p.read_bytes() for p in run_dir.iterdir() if p.is_file()}
    out = compare.build([("a", run_dir), ("b", run_dir), ("c", run_dir)], tmp_path / "cmp", labels="1.3")
    assert {p.name: p.read_bytes() for p in run_dir.iterdir() if p.is_file()} == before
    assert out["scoring"]["labels"] == "1.3"
    s = out["summary"]
    assert s["agentic@a"]["keyword_correct"] == s["agentic@c"]["keyword_correct"]
    assert not out["steps"]["agentic"]["c_vs_b"]["wins"] and not out["steps"]["agentic"]["c_vs_b"]["losses"]
    assert len(out["scorer_effect"]) == len(compare.SCORER_STEPS)
    assert "Source selection" in (tmp_path / "cmp" / "comparison.md").read_text()


def test_rubric_v3_only_changes_partial_answer_handling():
    v2, v3 = judge.RUBRICS["2"][0], judge.RUBRICS["3"][0]
    assert v3.startswith(v2.split("- completeness")[0])
    assert "never expect that information to be supplied" in v3 and "never expect" not in v2
    assert judge.RUBRICS["1"][0] == judge.RUBRIC_V1          # earlier rubrics kept for reproducibility


def test_untracked_outputs_do_not_mark_the_commit_as_uncommitted(monkeypatch):
    calls = []

    def fake_git(*args):
        calls.append(args)
        if args[:2] == ("rev-parse", "--short"):
            return "abc1234"
        if args[0] == "status":
            assert "--untracked-files=no" in args       # untracked files are not "uncommitted changes"
            return ""
        if args[:2] == ("ls-files", "--others"):
            return "eval-out/records.jsonl"
        return ""

    monkeypatch.setattr(runner, "_git", fake_git)
    monkeypatch.setattr(runner.httpx, "get", lambda *a, **k: (_ for _ in ()).throw(runner.httpx.ConnectError("x")))
    env = runner.environment(runner.EvalConfig(out_dir="x"))
    assert env["git_commit"] == "abc1234" and env["untracked_files"] == ["eval-out/records.jsonl"]


def test_cli_rejects_unknown_label_and_scoring_versions(capsys):
    with pytest.raises(SystemExit):
        cli(["run", "--out", "x", "--labels", "9.9"])
    with pytest.raises(SystemExit):
        cli(["report", "x", "--scoring", "0.1"])
