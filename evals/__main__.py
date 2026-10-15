"""
RAGBase evaluation command line.

    python -m evals run --out evals/results/dev1 --modes standard,agentic --repeats 1 --tags ci_subset
    python -m evals report evals/results/dev1
    python -m evals verifier --out evals/results/verifier.json
    python -m evals tabular --out evals/results/tabular.json
    python -m evals judge evals/results/dev1 --max-items 10 [--estimate]

Official runs (--label official) are refused unless every selected case has
review_status "approved".
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import config


def _csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()] if value else []


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m evals")
    sub = ap.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="run benchmark cases")
    run.add_argument("--out", required=True)
    run.add_argument("--modes", default="standard,agentic", help="any of original,standard,agentic")
    run.add_argument("--repeats", type=int, default=3)
    run.add_argument("--tags", default="")
    run.add_argument("--cases", default="")
    run.add_argument("--label", default="development", choices=["development", "official"])
    run.add_argument("--temperature", type=float, default=None, help="override generation temperature (e.g. 0 for debugging)")
    run.add_argument("--seed", type=int, default=1000, help="base seed; repeat i uses seed+i")
    run.add_argument("--no-rerank", action="store_true")
    run.add_argument("--model", default=config.OLLAMA_MODEL)

    rep = sub.add_parser("report", help="score a run and write report.md / summary.csv")
    rep.add_argument("run_dir")

    ver = sub.add_parser("verifier", help="verifier challenge set")
    ver.add_argument("--out", required=True)
    ver.add_argument("--model", default=config.OLLAMA_MODEL)
    ver.add_argument("--repeats", type=int, default=1)

    tab = sub.add_parser("tabular", help="tabular regression suite (with security checks)")
    tab.add_argument("--out", required=True)
    tab.add_argument("--model", default=config.OLLAMA_MODEL)

    jud = sub.add_parser("judge", help="optional LLM judge over a run")
    jud.add_argument("run_dir")
    jud.add_argument("--model", default="gemma3:12b")
    jud.add_argument("--max-items", type=int, default=10, help="0 = all")
    jud.add_argument("--judgments", type=int, default=1)
    jud.add_argument("--modes", default="standard,agentic")
    jud.add_argument("--cases", default="")
    jud.add_argument("--estimate", action="store_true", help="time 3 items and project the full workload")

    args = ap.parse_args(argv)
    logging.disable(logging.WARNING)

    if args.cmd == "run":
        from evals import runner
        from evals.dataset import load_cases

        cfg = runner.EvalConfig(out_dir=args.out, modes=_csv(args.modes), repeats=args.repeats,
                                tags=_csv(args.tags), case_ids=_csv(args.cases), label=args.label,
                                temperature=args.temperature, base_seed=args.seed,
                                reranking=not args.no_rerank, generator_model=args.model)
        selected = runner.select_cases(cfg, load_cases())
        unapproved = [c.id for c in selected if c.review_status != "approved"]
        if args.label == "official" and unapproved:
            print(f"Refusing an official run: {len(unapproved)} selected case(s) are not approved "
                  f"(e.g. {', '.join(unapproved[:5])}).", file=sys.stderr)
            return 2
        runner.run(cfg)
        from evals import report

        report.build(Path(args.out))
        print(f"Report: {Path(args.out) / 'report.md'}")
        return 0

    if args.cmd == "report":
        from evals import report

        report.build(Path(args.run_dir))
        print(f"Report: {Path(args.run_dir) / 'report.md'}")
        return 0

    from llm.ollama import OllamaChatModel

    if args.cmd == "verifier":
        from evals import verifier_eval
        from evals.dataset import load_verifier_cases

        out = verifier_eval.run(load_verifier_cases(), OllamaChatModel(model=args.model), args.repeats)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(json.dumps(out["summary"], indent=2))
        return 0

    if args.cmd == "tabular":
        from evals import tabular_eval
        from evals.dataset import load_tabular_cases

        model = OllamaChatModel(model=args.model)
        results = [tabular_eval.run_case(c, model).to_dict() for c in load_tabular_cases()]
        summary = {"passed": sum(r["passed"] for r in results), "total": len(results),
                   "security_violations": sum(bool(r["security_violations"]) for r in results)}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"model": args.model, "summary": summary, "results": results}, indent=2))
        print(json.dumps(summary, indent=2))
        return 0

    if args.cmd == "judge":
        from evals import judge

        out = judge.run(Path(args.run_dir), OllamaChatModel(model=args.model, timeout_s=600), args.max_items,
                        args.judgments, tuple(_csv(args.modes)), tuple(_csv(args.cases)), estimate_only=args.estimate)
        print(json.dumps({k: v for k, v in out.items() if k != "disagreements"}, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
