"""
Run the agent from the command line against a built index.

    python -m agent "Compare the 2023 and 2024 travel policies"
    python -m agent --index-dir ./ragbase_index --json "question"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import config
from agent import AgentSettings, run_agent
from llm import get_chat_model
from retriever import get_retriever


def _print_result(result) -> None:
    print(f"\nRoute: {result.route}   Status: {result.status}   "
          f"LLM calls: {result.llm_calls}   Searches: {result.retrieval_calls}   "
          f"Revisions: {result.revision_count}   Time: {result.elapsed_s:.1f}s")
    print(f"Inference: {result.inference_summary()}")
    print("\nTrace:")
    for e in result.trace:
        print(f"  {e['step']:>2}. [{e['node']}] {e['summary']}  ({e['elapsed_s']:.1f}s)")
    print("\nAnswer:\n" + result.answer)
    if result.citations:
        print("\nCited evidence:")
        cited = set(result.citations.valid_ids)
        for ev in result.evidence:
            if ev.eid in cited:
                print(f"  [{ev.eid}] {ev.label()}")
    if result.verification:
        v = result.verification
        print(f"\nVerification: citations {'valid' if v.citation_ok else 'INVALID'}, "
              f"grounding {v.grounding}, verifier {v.llm_status}")
        for c in v.unsupported_claims:
            print(f"  unsupported: {c}")
        for m in v.numeric_mismatches:
            print(f"  number mismatch: {m}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m agent", description=__doc__.strip().splitlines()[0])
    ap.add_argument("question")
    ap.add_argument("--index-dir", default=config.INDEX_DIR)
    ap.add_argument("--backend", default=config.LLM_BACKEND, choices=["ollama", "prism"],
                    help="ollama = direct local; prism = through the Prism gateway (no fallback)")
    ap.add_argument("--model", default=config.OLLAMA_MODEL, help="Ollama model (direct backend)")
    ap.add_argument("--prism-model", default=config.PRISM_MODEL, help="Prism virtual model")
    ap.add_argument("--verifier-model", default=config.AGENT_VERIFIER_MODEL or None)
    ap.add_argument("--max-revisions", type=int, default=config.AGENT_MAX_REVISIONS)
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--json", action="store_true", help="print the full result as JSON")
    args = ap.parse_args(argv)

    logging.disable(logging.WARNING)
    retriever = get_retriever(index_dir=args.index_dir, reranking=not args.no_rerank)
    result = run_agent(
        args.question,
        retriever,
        get_chat_model(backend=args.backend, model=args.model, prism_model=args.prism_model),
        verifier_model=(
            get_chat_model(backend=args.backend, model=args.verifier_model, prism_model=args.verifier_model)
            if args.verifier_model else None
        ),
        settings=AgentSettings(max_revisions=args.max_revisions),
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        _print_result(result)
    return 0 if result.status not in ("failed", "timeout") else 1


if __name__ == "__main__":
    sys.exit(main())
