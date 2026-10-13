"""
End-to-end demonstration against a real Prism gateway: RAGBase → Prism → Ollama.

    set -a; source .env; set +a
    python scripts/prism/live_demo.py --out docs/prism_results/live_demo.json

Uses the synthetic, PII-free fixture corpus. Each question is asked once
through Prism and once directly through Ollama (for latency comparison), then
the script exercises cache-hit rejection, mock-provider rejection and a PII
guardrail block. Prism caches responses for an hour without recording the
provider, so re-running within that window will show cache rejections for
repeated prompts; that is expected and is reported, not hidden.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from langchain_core.documents import Document  # noqa: E402

import ingest  # noqa: E402
import retriever as r  # noqa: E402
from agent import run_agent  # noqa: E402
from citations import build_evidence  # noqa: E402
from llm import LLMError, get_chat_model  # noqa: E402
from llm.openai_compat import PrismChatModel  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"
STANDARD_QS = [
    "How long do I have to submit an expense report under the 2024 policy?",
    "Above what nightly rate does a hotel stay need director approval in 2024?",
    "How many days of annual leave can be carried into the next year?",
]
AGENT_QS = [
    "How did the expense report deadline change between the 2023 and 2024 travel policies?",
    "Compare the business class rules in the 2023 and 2024 travel policies.",
]


def call_record(result) -> dict:
    m = result.metadata
    return {k: v for k, v in {
        "backend": result.backend, "provider_used": m.get("provider_used"),
        "inference_mode": m.get("inference_mode"), "cache_hit": m.get("cache_hit"),
        "fallback_triggered": m.get("fallback_triggered"), "cost_usd": m.get("cost_usd"),
        "gateway_latency_ms": m.get("gateway_latency_ms"), "client_latency_s": round(result.latency_s, 3),
        "prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens,
    }.items() if v is not None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    logging.disable(logging.WARNING)

    prism = PrismChatModel(api_key=os.environ.get("PRISM_API_KEY", ""))
    direct = get_chat_model(backend="ollama")
    idx = str(Path(args.out).with_suffix("")) + "_index"
    ingest.ingest_documents(ingest_dir=str(FIXTURES / "agent_docs"), index_dir=idx)
    ret = r.get_retriever(index_dir=idx)
    report = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "machine": f"{platform.machine()} {platform.system()} {platform.release()}",
              "prism_model": prism.model, "ollama_model": direct.model, "standard": [], "agent": [], "checks": {}}

    # Warm both paths so the first measured request does not pay model-load time.
    direct.complete([{"role": "user", "content": "Reply with the word ready."}], temperature=0.0)

    for q in STANDARD_QS:
        row = {"question": q}
        for name, model in (("prism", prism), ("direct", direct)):
            t = time.perf_counter()
            try:
                out = r.build_rag_chain(ret, chat_model=model).invoke(q)
                row[name] = {"ok": True, "end_to_end_s": round(time.perf_counter() - t, 3),
                             "citations_ok": out["citations"].ok, "cited": out["citations"].valid_ids,
                             "answer": out["answer"], **call_record(out["llm"])}
            except LLMError as e:
                row[name] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        report["standard"].append(row)
        print(f"standard | prism={'ok' if row['prism']['ok'] else row['prism']['error']} | {q}")

    for q in AGENT_QS:
        res = run_agent(q, ret, prism)
        report["agent"].append({"question": q, "status": res.status, "route": res.route,
                                "sources": sorted({e.source for e in res.evidence}),
                                "citations_ok": res.citations.ok if res.citations else None,
                                "grounding": res.verification.grounding if res.verification else None,
                                "llm_calls": res.llm_calls, "elapsed_s": round(res.elapsed_s, 2),
                                "inference": res.inference_summary(), "error": res.error,
                                "trace": [f"{e['node']}: {e['summary']}" for e in res.trace],
                                "answer": res.answer})
        print(f"agent    | {res.status} route={res.route} calls={res.llm_calls} | {q}")

    # Cache hit without provenance: repeat the first standard question verbatim.
    try:
        r.build_rag_chain(ret, chat_model=prism).invoke(STANDARD_QS[0])
        report["checks"]["repeat_request"] = "accepted (unexpected)"
    except LLMError as e:
        report["checks"]["repeat_request"] = f"{type(e).__name__}: {e}"
    # Mock provider: Prism's seeded 'mock-fast' route.
    try:
        PrismChatModel(api_key=prism._api_key, model="mock-fast").complete(
            [{"role": "user", "content": "Summarise the travel policy in one sentence."}], temperature=0.0)
        report["checks"]["mock_route"] = "accepted (unexpected)"
    except LLMError as e:
        report["checks"]["mock_route"] = f"{type(e).__name__}: {e}"
    # Guardrail on synthetic PII.
    text = (FIXTURES / "guardrail_docs" / "named_approvers.txt").read_text()
    ev = build_evidence([Document(page_content=text, metadata={"chunk_id": "n", "source": "named_approvers.txt", "loc": "txt"})])
    try:
        prism.complete(r.build_standard_messages("Who approves purchases above 5,000 dollars?", ev), temperature=0.0)
        report["checks"]["pii_document"] = "accepted (unexpected)"
    except LLMError as e:
        report["checks"]["pii_document"] = f"{type(e).__name__}: {e}"
    for k, v in report["checks"].items():
        print(f"check    | {k}: {v}")

    pairs = [(s["prism"], s["direct"]) for s in report["standard"] if s["prism"].get("ok") and s["direct"].get("ok")]
    if pairs:
        report["latency_summary"] = {
            "n": len(pairs),
            "prism_end_to_end_median_s": round(statistics.median(p["end_to_end_s"] for p, _ in pairs), 3),
            "direct_end_to_end_median_s": round(statistics.median(d["end_to_end_s"] for _, d in pairs), 3),
            "prism_client_call_median_s": round(statistics.median(p["client_latency_s"] for p, _ in pairs), 3),
            "direct_client_call_median_s": round(statistics.median(d["client_latency_s"] for _, d in pairs), 3),
            "note": "Answers differ in length between paths, so per-call times are not a pure overhead measure.",
        }
        print("latency  |", report["latency_summary"])
    Path(args.out).write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
