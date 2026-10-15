"""
evals/runner.py — Run benchmark cases through Standard RAG, Agentic RAG and the original code.

All modes share one index built from evals/corpus with the same embedding
model, reranker, retrieval settings and generator model. Each answer is
written as one JSONL record; metrics are computed later from the records
(evals/report.py), so scoring changes never require re-running models.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import httpx

import config
import ingest
import retriever as r
from agent import AgentSettings, run_agent
from agent.verifier import numeric_mismatches
from evals.dataset import CORPUS_DIR, EVALS, Case, corpus_manifest, load_cases
from llm import ChatModel, LLMError
from llm.ollama import OllamaChatModel

REPO = EVALS.parent


@dataclass
class EvalConfig:
    out_dir: str
    modes: list[str] = field(default_factory=lambda: ["standard", "agentic"])
    repeats: int = 3
    case_ids: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    generator_model: str = config.OLLAMA_MODEL
    temperature: Optional[float] = None          # None = the application's own defaults
    base_seed: Optional[int] = 1000              # repeat i uses base_seed + i; None = unseeded
    top_k: int = config.TOP_K
    candidate_pool: int = config.CANDIDATE_POOL
    reranking: bool = config.RERANKING_ENABLED
    embed_model: str = config.EMBED_MODEL
    chunk_size: int = config.CHUNK_SIZE
    chunk_overlap: int = config.CHUNK_OVERLAP
    max_revisions: int = config.AGENT_MAX_REVISIONS
    original_ref: str = "909df31"                # pre-upgrade commit for the baseline mode
    label: str = "development"                   # "official" only after label approval


def select_cases(cfg: EvalConfig, cases: Optional[list[Case]] = None) -> list[Case]:
    cases = cases if cases is not None else load_cases()
    if cfg.case_ids:
        cases = [c for c in cases if c.id in set(cfg.case_ids)]
    if cfg.tags:
        cases = [c for c in cases if set(cfg.tags) & set(c.tags)]
    return cases


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()


def environment(cfg: EvalConfig) -> dict:
    env = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "--short", "HEAD") + ("+uncommitted" if _git("status", "--porcelain") else ""),
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
        "processor": platform.processor(),
        "corpus_version": corpus_manifest()["version"],
        "corpus_checksum": hashlib.sha256((CORPUS_DIR / "MANIFEST.json").read_bytes()).hexdigest()[:16],
        "backend": "ollama (direct)",
    }
    try:
        env["ollama_version"] = httpx.get(f"{config.OLLAMA_BASE_URL}/api/version", timeout=3).json().get("version")
        tags = httpx.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=3).json().get("models", [])
        env["model_digests"] = {m["name"]: m.get("digest", "")[:12] for m in tags
                                if m["name"].split(":")[0] == cfg.generator_model.split(":")[0]}
    except httpx.HTTPError:
        env["ollama_version"] = None
    return env


def preflight(cfg: EvalConfig) -> Optional[str]:
    """Problem that would make every live answer fail (server down, model missing), or None."""
    try:
        tags = httpx.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5).json().get("models", [])
    except httpx.HTTPError:
        return f"Ollama is not reachable at {config.OLLAMA_BASE_URL} (start it with `ollama serve`)"
    names = {m["name"] for m in tags} | {m["name"].split(":")[0] for m in tags if m["name"].endswith(":latest")}
    if cfg.generator_model not in names:
        return f"model '{cfg.generator_model}' is not available in Ollama (run `ollama pull {cfg.generator_model}`)"
    return None


def build_index(cfg: EvalConfig, run_dir: Path, embeddings=None) -> str:
    index_dir = run_dir / "index"
    if not (index_dir / "index.faiss").exists():
        ingest.ingest_documents(
            ingest_dir=str(CORPUS_DIR), index_dir=str(index_dir), chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap, embed_model=cfg.embed_model, embeddings=embeddings,
        )
    return str(index_dir)


def _evidence_dicts(evidence) -> list[dict]:
    return [{"eid": e.eid, "chunk_id": e.chunk_id, "source": e.source, "loc": e.loc, "text": e.text} for e in evidence]


def run_standard(case: Case, retriever, model: ChatModel) -> dict:
    t = time.perf_counter()
    try:
        out = r.build_rag_chain(retriever, chat_model=model).invoke(case.question)
    except LLMError as e:
        return {"error": f"{type(e).__name__}: {e}", "latency_s": round(time.perf_counter() - t, 3),
                "llm_calls": 1, "retrieval_calls": 1}
    rep = out["citations"]
    llm = out["llm"]
    return {
        "answer": out["answer"], "abstained": rep.abstained,
        "evidence": _evidence_dicts(out["evidence"]), "cited_ids": rep.valid_ids,
        "citations": rep.to_dict(), "numeric_mismatches": len(numeric_mismatches(out["answer"], out["evidence"])),
        "llm_calls": 1, "retrieval_calls": 1,
        "prompt_tokens": llm.prompt_tokens, "completion_tokens": llm.completion_tokens,
        "latency_s": round(time.perf_counter() - t, 3), "status": "answered",
    }


def run_agentic(case: Case, retriever, model: ChatModel, cfg: EvalConfig) -> dict:
    res = run_agent(case.question, retriever, model, settings=AgentSettings(max_revisions=cfg.max_revisions))
    rep = res.citations
    return {
        "answer": res.answer, "abstained": bool(rep and rep.abstained),
        "evidence": _evidence_dicts(res.evidence), "cited_ids": rep.valid_ids if rep else [],
        "citations": rep.to_dict() if rep else None,
        "numeric_mismatches": len(numeric_mismatches(res.answer, res.evidence)),
        "verification": res.verification.to_dict() if res.verification else None,
        "status": res.status, "route": res.route, "planner_fallback": res.planner_fallback,
        "revision_count": res.revision_count, "llm_calls": res.llm_calls, "retrieval_calls": res.retrieval_calls,
        "prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens,
        "latency_s": round(res.elapsed_s, 3), "error": res.error,
        "trace": [f"{e['node']}: {e['summary']}" for e in res.trace],
    }


def default_model_factory(cfg: EvalConfig) -> Callable[[int], ChatModel]:
    def make(repeat: int) -> ChatModel:
        kwargs = {"model": cfg.generator_model}
        if cfg.temperature is not None:
            kwargs["temperature"] = cfg.temperature
        if cfg.base_seed is not None:
            kwargs["seed"] = cfg.base_seed + repeat
        return OllamaChatModel(**kwargs)
    return make


def run(
    cfg: EvalConfig,
    model_factory: Optional[Callable[[int], ChatModel]] = None,
    embeddings=None,
    cases: Optional[list[Case]] = None,
    progress: Callable[[str], None] = print,
) -> Path:
    run_dir = Path(cfg.out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    selected = select_cases(cfg, cases)
    make_model = model_factory or default_model_factory(cfg)
    index_dir = build_index(cfg, run_dir, embeddings)
    retriever = r.get_retriever(index_dir=index_dir, embed_model=cfg.embed_model, top_k=cfg.top_k,
                                reranking=cfg.reranking, embeddings=embeddings)

    # Fixed-budget retrieval: the shared retriever's top-k for each question,
    # recorded once and used for every mode's retrieval metrics.
    with open(run_dir / "fixed_retrieval.jsonl", "w") as f:
        for c in selected:
            docs = retriever.search(c.question, k=cfg.top_k)
            f.write(json.dumps({"case_id": c.id, "texts": [d.page_content for d in docs],
                                "sources": [d.metadata["source"] for d in docs]}) + "\n")

    meta = {"config": asdict(cfg), "environment": environment(cfg), "cases": [c.id for c in selected],
            "dataset_review_status": sorted({c.review_status for c in selected})}
    (run_dir / "config.json").write_text(json.dumps(meta, indent=2))

    records_path = run_dir / "records.jsonl"
    done = set()
    if records_path.exists():
        for line in records_path.read_text().splitlines():
            rec = json.loads(line)
            done.add((rec["mode"], rec["repeat"], rec["case_id"]))

    with open(records_path, "a") as out:
        for repeat in range(1, cfg.repeats + 1):
            model = make_model(repeat)
            for mode in cfg.modes:
                if mode == "original":
                    continue
                for c in selected:
                    if (mode, repeat, c.id) in done:
                        continue
                    rec = run_standard(c, retriever, model) if mode == "standard" else run_agentic(c, retriever, model, cfg)
                    rec.update({"mode": mode, "repeat": repeat, "case_id": c.id, "category": c.category,
                                "seed": getattr(model, "seed", None)})
                    out.write(json.dumps(rec) + "\n")
                    out.flush()
                    progress(f"[{mode} r{repeat}] {c.id} {rec.get('status', '')} {rec.get('latency_s', '')}s")

    if "original" in cfg.modes:
        run_original(cfg, run_dir, selected, progress)
    return run_dir


def run_original(cfg: EvalConfig, run_dir: Path, cases: list[Case], progress=print) -> None:
    """Run the pre-upgrade code in an isolated subprocess on the same corpus."""
    snapshot = run_dir / f"original_{cfg.original_ref}"
    if not snapshot.exists():
        snapshot.mkdir(parents=True)
        archive = subprocess.run(["git", "archive", cfg.original_ref], cwd=REPO, capture_output=True, check=True)
        subprocess.run(["tar", "-x", "-C", str(snapshot)], input=archive.stdout, check=True)
    cmd = [sys.executable, "-I", str(EVALS / "original_snapshot.py"), "--snapshot", str(snapshot),
           "--corpus", str(CORPUS_DIR), "--index-dir", str(run_dir / "original_index"),
           "--cases", json.dumps([{"id": c.id, "category": c.category, "question": c.question} for c in cases]),
           "--repeats", str(cfg.repeats), "--model", cfg.generator_model, "--top-k", str(cfg.top_k),
           "--out", str(run_dir / "records.jsonl")]
    progress(f"[original] running snapshot {cfg.original_ref} in a subprocess")
    subprocess.run(cmd, check=True)
