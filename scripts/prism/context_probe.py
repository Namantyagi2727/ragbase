"""
Controlled long-context probe: does a fact at the start / end of long evidence survive?

    python scripts/prism/context_probe.py --target ollama-native   # RAGBase direct (num_ctx set)
    python scripts/prism/context_probe.py --target ollama-openai   # Prism's upstream request shape
    python scripts/prism/context_probe.py --target prism           # through a running Prism (PRISM_API_KEY)

Two facts are planted — one in the first passage, one in the last — inside
neutral filler of increasing length, and the model is asked for both. A fact
that is consistently missed at a size where it is found at smaller sizes
indicates truncation (or loss of attention). Reported prompt-token counts are
recorded but not treated as proof either way.

`ollama-openai` replicates Prism's Ollama provider exactly: POST
/v1/chat/completions with only model, messages and temperature — no context
length option. It does not go through Prism.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

START_FACT = "The archive room on the north side opens at 7:40 each morning."
END_FACT = "The visitor parking permit is printed on green card stock."
QUESTION = "At what time does the north archive room open, and what colour card is the visitor parking permit printed on?"
FILLER = (
    "The facilities team reviews routine maintenance requests every week and records "
    "the outcome in the shared tracker, noting any follow-up work that is required. "
)


def build_messages(n_passages: int) -> list[dict]:
    passages = [FILLER * 3 for _ in range(n_passages)]
    passages[0] = START_FACT + " " + passages[0]
    passages[-1] = passages[-1] + " " + END_FACT
    evidence = "\n\n".join(f"Passage {i + 1}:\n{p}" for i, p in enumerate(passages))
    return [
        {"role": "system", "content": "Answer only from the passages provided. If a detail is absent, say it is not stated."},
        {"role": "user", "content": f"{evidence}\n\nQuestion: {QUESTION}"},
    ]


def call(target: str, model: str, messages: list[dict], num_ctx: int) -> tuple[str, int | None]:
    if target == "ollama-native":
        r = httpx.post("http://localhost:11434/api/chat", timeout=300, json={
            "model": model, "messages": messages, "stream": False,
            "options": {"temperature": 0, "num_ctx": num_ctx}})
        r.raise_for_status()
        d = r.json()
        return d["message"]["content"], d.get("prompt_eval_count")
    if target == "ollama-openai":
        r = httpx.post("http://localhost:11434/v1/chat/completions", timeout=300, json={
            "model": model, "messages": messages, "temperature": 0})
        r.raise_for_status()
        d = r.json()
        return d["choices"][0]["message"]["content"], d.get("usage", {}).get("prompt_tokens")
    if target == "prism":
        from llm.openai_compat import PrismChatModel

        res = PrismChatModel(api_key=os.environ.get("PRISM_API_KEY", "")).complete(messages, temperature=0.0)
        return res.text, res.prompt_tokens
    raise ValueError(target)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=["ollama-native", "ollama-openai", "prism"])
    ap.add_argument("--model", default="llama3.2:3b")
    ap.add_argument("--num-ctx", type=int, default=4096, help="ollama-native only (RAGBase's setting)")
    ap.add_argument("--sizes", default="2,6,12,20,30", help="passage counts (~110 tokens each)")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--out")
    args = ap.parse_args()

    rows = []
    for n in [int(x) for x in args.sizes.split(",")]:
        for rep in range(args.repeats):
            t = time.perf_counter()
            try:
                text, ptoks = call(args.target, args.model, build_messages(n), args.num_ctx)
                err = None
            except Exception as e:  # recorded, not hidden
                text, ptoks, err = "", None, f"{type(e).__name__}: {str(e)[:120]}"
            low = text.lower()
            row = {"target": args.target, "passages": n, "repeat": rep + 1, "prompt_tokens": ptoks,
                   "start_fact_found": "7:40" in low, "end_fact_found": "green" in low,
                   "latency_s": round(time.perf_counter() - t, 2), "error": err}
            rows.append(row)
            print(f"{args.target:13s} passages={n:3d} rep={rep + 1} prompt_tokens={ptoks} "
                  f"start={'yes' if row['start_fact_found'] else 'NO '} end={'yes' if row['end_fact_found'] else 'NO '} "
                  f"{row['latency_s']}s{'  ' + err if err else ''}")
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
