"""
Score prompts with Prism's own guardrail function, in-process. Read-only use of Prism.

Run with Prism's Python environment (it needs presidio/spaCy, not RAGBase deps):

    /path/to/prism/.venv/bin/python scripts/prism/scan_guardrails.py \
        --prism-src /path/to/prism/src --prompts prompts.jsonl --out results.jsonl

Text is concatenated exactly as Prism's /v1/chat/completions handler does
(message contents joined with spaces) before calling prism.core.guardrails.scan_prompt.
This measures the guardrail decision only; it does not exercise HTTP, auth or providers.
--simulate-id-format rewrites RAGBase's evidence IDs before scanning, to measure how
an alternative citation format would fare; it does not change what RAGBase sends.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prism-src", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--simulate-id-format", metavar="FMT",
                    help="measurement only: rewrite evidence IDs E<n> to FMT (e.g. 'E-{n}') before scanning")
    args = ap.parse_args()

    sys.path.insert(0, args.prism_src)
    from prism.core.guardrails import scan_prompt  # Prism's real implementation

    rows = [json.loads(line) for line in open(args.prompts)]
    results = []
    for row in rows:
        full_text = " ".join(m.get("content", "") for m in row["messages"])  # as in prism/api/chat.py
        if args.simulate_id_format:
            full_text = re.sub(r"\bE(\d+)\b", lambda m: args.simulate_id_format.format(n=m.group(1)), full_text)
        verdict = scan_prompt(full_text)
        results.append({"case": row["case"], "role": row["role"],
                        "blocked": verdict.flagged, "reason": verdict.reason})

    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    blocked = [r for r in results if r["blocked"]]
    print(f"{len(blocked)}/{len(results)} prompts blocked ({100 * len(blocked) / max(1, len(results)):.0f}%)")
    print("Reasons:", dict(Counter(r["reason"] for r in blocked)))
    by_case = defaultdict(list)
    for r in results:
        by_case[r["case"]].append(r)
    for case, rs in by_case.items():
        marks = ", ".join(f"{r['role']}={'BLOCKED' if r['blocked'] else 'ok'}" for r in rs)
        reasons = sorted({r["reason"] for r in rs if r["reason"]})
        print(f"  {case}: {marks}" + (f"  [{'; '.join(reasons)}]" if reasons else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
