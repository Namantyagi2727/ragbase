"""
Build the exact prompts RAGBase would send through Prism, for guardrail testing.

    python scripts/prism/build_guardrail_prompts.py --out prompts.jsonl
    python scripts/prism/build_guardrail_prompts.py --out bench.jsonl \
        --index-dir ./benchmark_index --queries benchmark   # real retrieved evidence

Each line: {"case", "role", "messages"}. Roles: standard (Standard RAG and the
agent's simple route), planner, synthesis, verifier, abstention_check, tabular.
Structured roles are emitted exactly as the Prism adapter sends them (with the
field-list instruction appended). Run scan_guardrails.py
(in Prism's environment) or check_live.py (against a running Prism) on the output.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from langchain_core.documents import Document  # noqa: E402

from agent.prompts import (  # noqa: E402
    abstention_check_messages,
    planner_messages,
    synthesis_messages,
    verifier_messages,
)
from agent.schemas import AbstentionCheck, QueryPlan, VerifierOutput  # noqa: E402
from citations import build_evidence  # noqa: E402
from llm.openai_compat import with_json_instruction  # noqa: E402
from retriever import build_standard_messages  # noqa: E402
from tabular import TABULAR_SYSTEM_PROMPT, TabularQuery  # noqa: E402


def on_the_wire(messages, schema):
    """The prompt exactly as the Prism adapter sends a structured request."""
    return with_json_instruction(messages, schema.model_json_schema())

FIXTURES = ROOT / "tests" / "fixtures" / "guardrail_docs"
QUESTIONS = {
    "plain_policy.txt": "What is the daily meal allowance?",
    "named_approvers.txt": "Who approves purchases above 5,000 dollars?",
    "contact_details.txt": "How do I report a lost laptop?",
    "injection_like.txt": "What are the vendor's payment terms?",
    "benign_trigger_words.txt": "When am I eligible for remote access?",
    "network_config.txt": "Which address range do office printers use?",
    "identifiers.txt": "How often are fleet vehicles inspected?",
}


def fixture_cases():
    docs = sorted(FIXTURES.glob("*.txt"))
    names = [d.name for d in docs]
    for path in docs:
        text = path.read_text().strip()
        question = QUESTIONS[path.name]
        evidence = build_evidence([Document(page_content=text, metadata={
            "chunk_id": path.stem, "source": path.name, "loc": "txt"})])
        claim = f"{text.split('.')[1].strip()} [{evidence[0].eid}]."
        yield path.name, "standard", build_standard_messages(question, evidence)
        yield path.name, "planner", on_the_wire(planner_messages(question, names), QueryPlan)
        yield path.name, "synthesis", synthesis_messages(question, evidence)
        yield path.name, "verifier", on_the_wire(verifier_messages(question, evidence, [claim]), VerifierOutput)
        yield path.name, "abstention_check", on_the_wire(abstention_check_messages(question, evidence), AbstentionCheck)
    # Our own prompts with a neutral question and no document text.
    yield "(no documents)", "planner", on_the_wire(planner_messages("What is the refund window?", ["policy.txt"]), QueryPlan)
    tab = [{"role": "system", "content": TABULAR_SYSTEM_PROMPT},
           {"role": "user", "content": "Tables:\n- table 'sales.csv': 4 rows; columns: region (object), units (int64)\n\n"
                                       "Question: How many units were sold in the EU?"}]
    yield "(tabular)", "tabular", on_the_wire(tab, TabularQuery)


def index_cases(index_dir: str, queries: list[str], k: int):
    logging.disable(logging.WARNING)
    import retriever as r

    ret = r.get_retriever(index_dir=index_dir, top_k=k)
    for q in queries:
        evidence = build_evidence(ret.invoke(q))
        yield f"index:{q[:60]}", "standard", build_standard_messages(q, evidence)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--index-dir", help="also build standard prompts from this index")
    ap.add_argument("--queries", help="'benchmark' for benchmark.py's queries, or a text file with one per line")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    cases = list(fixture_cases())
    if args.index_dir:
        if args.queries == "benchmark":
            import benchmark

            queries = benchmark.QUERIES
        else:
            queries = [q.strip() for q in Path(args.queries).read_text().splitlines() if q.strip()]
        cases += list(index_cases(args.index_dir, queries, args.k))

    with open(args.out, "w") as f:
        for case, role, messages in cases:
            f.write(json.dumps({"case": case, "role": role, "messages": messages}) + "\n")
    print(f"Wrote {len(cases)} prompts to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
