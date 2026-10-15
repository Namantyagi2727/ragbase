"""
evals/original_snapshot.py — Run the pre-upgrade RAGBase code on the evaluation corpus.

Invoked by evals/runner.py in a separate, isolated interpreter (python -I) with
a `git archive` of the original commit, so its modules (config, ingest,
retriever) never mix with the current ones. Imports nothing from the current
repository.

Only the original document Q&A path is exercised: ingest.main ->
retriever.get_retriever -> retriever.build_rag_chain. The original tabular
path, which executed model-generated Python, is never imported or called.
The original code builds its own OllamaLLM without a seed, so runs are unseeded.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--cases", required=True, help="JSON list of {id, category, question}")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--model", required=True)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.snapshot)
    logging.disable(logging.WARNING)
    import ingest  # the original module from the snapshot
    import retriever

    assert ingest.__file__.startswith(args.snapshot) and retriever.__file__.startswith(args.snapshot), \
        "original modules must come from the snapshot"

    ingest.main(ingest_dir=args.corpus, index_dir=args.index_dir)
    ret = retriever.get_retriever(index_dir=args.index_dir, top_k=args.top_k, reranking=True)
    chain = retriever.build_rag_chain(ret, args.model)

    cases = json.loads(args.cases)
    done = set()
    try:
        for line in open(args.out):
            rec = json.loads(line)
            done.add((rec["mode"], rec["repeat"], rec["case_id"]))
    except FileNotFoundError:
        pass

    with open(args.out, "a") as out:
        for repeat in range(1, args.repeats + 1):
            for c in cases:
                if ("original", repeat, c["id"]) in done:
                    continue
                t = time.perf_counter()
                try:
                    result = chain.invoke(c["question"])
                    rec = {
                        "answer": result["answer"].strip(),
                        "evidence": [{"eid": str(i), "source": d.metadata.get("source"), "loc": d.metadata.get("loc"),
                                      "text": d.page_content} for i, d in enumerate(result["context"], start=1)],
                        "cited_ids": [],          # the original prompt asked for free-form citations
                        "citations": None,
                        "llm_calls": 1, "retrieval_calls": 1, "status": "answered",
                    }
                except Exception as e:  # recorded, not hidden
                    rec = {"error": f"{type(e).__name__}: {e}", "llm_calls": 1, "retrieval_calls": 1}
                rec.update({"mode": "original", "repeat": repeat, "case_id": c["id"], "category": c["category"],
                            "latency_s": round(time.perf_counter() - t, 3), "seed": None})
                out.write(json.dumps(rec) + "\n")
                out.flush()
                print(f"[original r{repeat}] {c['id']} {rec['latency_s']}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
