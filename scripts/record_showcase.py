"""
scripts/record_showcase.py — Record genuine RAGBase runs for the public showcase.

    python scripts/record_showcase.py            # writes showcase/data/recorded_runs.json and corpus.json

Runs each curated question through the real local pipeline — Standard RAG and
Agentic RAG, the benchmark's retrieval configuration, Ollama with a fixed
seed — and stores the answer views exactly as the local app renders them,
with provenance (commit, model digest, Ollama version, configuration and
per-run timestamps). Refuses to run with uncommitted changes to tracked
files, so every recording maps to a commit. Nothing is edited after recording;
the curated notes live separately in showcase/curation.json.
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
import ingest  # noqa: E402
import retriever as ret  # noqa: E402
from agent import AgentSettings, run_agent  # noqa: E402
from evals.dataset import CORPUS_DIR, corpus_manifest, load_cases  # noqa: E402
from evals.runner import EvalConfig, environment  # noqa: E402
from llm.ollama import OllamaChatModel  # noqa: E402
from ui import render as rv  # noqa: E402

OUT = ROOT / "showcase" / "data"
CURATION = ROOT / "showcase" / "curation.json"
SEED = 1001          # the benchmark's first repeat
MODEL = "llama3.2:3b"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _clean_tree() -> str | None:
    out = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip()
    return out or None


def corpus_documents() -> list[dict]:
    """Document text as RAGBase's ingestion extracts it (column-aware PDF, DOCX, TXT), with checksums."""
    manifest = corpus_manifest()
    docs = []
    for key in sorted(k for k in manifest["files"] if k.startswith("corpus/")):
        name = key.split("/", 1)[1]
        entries = ingest.load_doc(str(CORPUS_DIR / name))
        docs.append({"source": name, "sha256": manifest["files"][key],
                     "sections": [{"loc": loc, "text": text} for text, _, loc in entries]})
    return docs


def main() -> int:
    logging.disable(logging.WARNING)
    dirty = _clean_tree()
    if dirty:
        print(f"Refusing to record: tracked files have uncommitted changes:\n{dirty}", file=sys.stderr)
        return 2
    curation = json.loads(CURATION.read_text())
    cases = {c.id: c for c in load_cases(labels="1.3")}
    cfg = EvalConfig(out_dir=str(OUT), generator_model=MODEL, base_seed=SEED - 1, label="development")
    env = environment(cfg)
    with tempfile.TemporaryDirectory() as tmp:
        ingest.ingest_documents(ingest_dir=str(CORPUS_DIR), index_dir=tmp, chunk_size=cfg.chunk_size,
                                chunk_overlap=cfg.chunk_overlap, embed_model=cfg.embed_model)
        retriever = ret.get_retriever(index_dir=tmp, embed_model=cfg.embed_model, top_k=cfg.top_k,
                                      reranking=cfg.reranking)
        model = OllamaChatModel(model=MODEL, seed=SEED)
        settings = AgentSettings()
        runs = []
        for item in curation["questions"]:
            case = cases[item["case_id"]]
            entry = {"case_id": case.id, "category": case.category, "question": case.question,
                     "reference_answer": case.reference_answer, "answerable": case.answerable, "runs": {}}
            for mode in ("standard", "agentic"):
                started = _now()
                t = time.perf_counter()
                if mode == "standard":
                    result = ret.build_rag_chain(retriever, chat_model=model).invoke(case.question)
                    view = rv.view_from_standard(result, time.perf_counter() - t)
                else:
                    view = rv.view_from_agent(run_agent(case.question, retriever, model, settings=settings))
                view["recorded"] = {"started": started, "finished": _now()}
                entry["runs"][mode] = view
                print(f"[{mode}] {case.id} {time.perf_counter() - t:.1f}s", flush=True)
            runs.append(entry)
    provenance = {
        "recorded_at": _now(), "git_commit": env["git_commit"], "ollama_version": env.get("ollama_version"),
        "model": MODEL, "model_digest": (env.get("model_digests") or {}).get(MODEL), "seed": SEED,
        "temperature": "application default", "platform": f"{platform.system()} {platform.machine()}",
        "corpus_version": corpus_manifest().get("version"), "corpus_checksum": env.get("corpus_checksum"),
        "retrieval": {"embed_model": cfg.embed_model, "reranking": cfg.reranking, "top_k": cfg.top_k,
                      "chunk_size": cfg.chunk_size, "chunk_overlap": cfg.chunk_overlap,
                      "candidate_pool": config.CANDIDATE_POOL},
        "agent": {"max_revisions": settings.max_revisions, "time_budget_s": settings.time_budget_s,
                  "max_compare_docs": settings.max_compare_docs},
        "labels": "1.3",
        "note": "Generated by scripts/record_showcase.py; answers and traces are stored exactly as produced.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "recorded_runs.json").write_text(json.dumps({"provenance": provenance, "questions": runs}, indent=1))
    (OUT / "corpus.json").write_text(json.dumps({"version": provenance["corpus_version"],
                                                 "documents": corpus_documents()}, indent=1))
    print(f"Wrote {OUT / 'recorded_runs.json'} at commit {env['git_commit']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
