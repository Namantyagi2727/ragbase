"""
benchmark.py — Measures RAGBase's real ingestion and retrieval/generation
latency against an actual corpus, using the production ingest.py /
retriever.py code paths (no mocking).

Usage:
    python make_benchmark_corpus.py          # one-time: builds ./benchmark_corpus
    python benchmark.py                      # ingests it and times queries

Reports, separately:
  - ingestion throughput (files/sec, chunks produced)
  - hybrid retrieval alone (FAISS + TF-IDF, no reranking)
  - hybrid + cross-encoder reranking (RAGBase's default retrieval config)
  - full retrieval + LLM generation (requires Ollama running)

Results are printed and written to benchmark_results.json.
"""
import argparse
import json
import os
import platform
import statistics as stats
import sys
import time

import ingest
import retriever as rmod
import config

QUERIES = [
    "Who is Elizabeth Bennet and how does she feel about Mr. Darcy?",
    "Describe the creature that Victor Frankenstein creates.",
    "What advice does the White Rabbit give Alice?",
    "Describe Captain Ahab's obsession with the white whale.",
    "What deductions does Sherlock Holmes make about his visitor?",
    "How does Dr. Jekyll transform into Mr. Hyde?",
    "What is the significance of Dorian Gray's portrait?",
    "Describe Huckleberry Finn's journey down the river.",
    "What happens to Jonathan Harker at Dracula's castle?",
    "How does Emma Woodhouse react to Mr. Knightley?",
    "What happens to Gregor Samsa at the start of Metamorphosis?",
    "Describe the effect of the yellow wallpaper on the narrator.",
    "What historical events open A Tale of Two Cities?",
    "What does the Modest Proposal suggest about the poor in Ireland?",
    "What does Long John Silver want on the ship?",
    "How is Prince Andrei wounded at Austerlitz?",
    "What is Ulysses' relationship to Leopold Bloom?",
    "Describe the relationship between Mr. Rochester and Jane.",
    "What does Ishmael think of the Pequod's crew?",
    "How does the narrator describe the storm at sea?",
    "What is the moral lesson at the end of the story?",
    "Describe the setting of the opening chapter.",
    "What conflict arises between the two main characters?",
    "How is the villain finally defeated?",
    "What does the letter reveal about the family's secret?",
    "Describe the protagonist's inner thoughts before the confrontation.",
    "What role does fate play in the story's tragic ending?",
    "How does the author describe the countryside in autumn?",
    "What does the doctor conclude about the patient's condition?",
    "Describe the ball scene and who attends it.",
]


def pct(values, p):
    s = sorted(values)
    idx = min(len(s) - 1, max(0, round(p / 100 * (len(s) - 1))))
    return s[idx]


def summarize(label, times):
    return {
        "label": label, "n": len(times),
        "min_s": round(min(times), 3), "mean_s": round(stats.mean(times), 3),
        "median_s": round(stats.median(times), 3), "p95_s": round(pct(times, 95), 3),
        "max_s": round(max(times), 3), "all_s": [round(t, 3) for t in times],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", default="./benchmark_corpus")
    ap.add_argument("--index-dir", default="./benchmark_index")
    ap.add_argument("--skip-generation", action="store_true",
                     help="Skip LLM generation timing (no Ollama required)")
    ap.add_argument("--generation-queries", type=int, default=15)
    ap.add_argument("--out", default="./benchmark_results.json")
    args = ap.parse_args()

    if not os.path.isdir(args.corpus_dir) or not os.listdir(args.corpus_dir):
        sys.exit(f"No corpus found at {args.corpus_dir}. Run make_benchmark_corpus.py first.")

    results = {}
    results["hardware"] = {
        "machine": platform.machine(),
        "processor": os.popen("sysctl -n machdep.cpu.brand_string 2>/dev/null").read().strip() or platform.processor(),
        "cores": os.cpu_count(),
        "os": f"{platform.system()} {platform.release()}",
        "python": sys.version.split()[0],
    }

    file_counts = {}
    for fn in os.listdir(args.corpus_dir):
        ext = fn.rsplit(".", 1)[-1].lower()
        file_counts[ext] = file_counts.get(ext, 0) + 1
    results["corpus"] = {"total_files": sum(file_counts.values()), "by_type": file_counts}

    print("=== Ingesting corpus ===")
    t0 = time.perf_counter()
    n_chunks = ingest.main(ingest_dir=args.corpus_dir, index_dir=args.index_dir)
    t1 = time.perf_counter()
    results["ingestion"] = {
        "total_seconds": round(t1 - t0, 2),
        "total_chunks": n_chunks,
        "files_per_second": round(results["corpus"]["total_files"] / (t1 - t0), 2),
    }
    print(f"Ingested {n_chunks} chunks from {results['corpus']['total_files']} files in {t1-t0:.1f}s")

    print("=== Loading retriever ===")
    t0 = time.perf_counter()
    vectorstore = rmod.load_vectorstore(args.index_dir, config.EMBED_MODEL)
    vectorizer, tfidf_matrix, chunks_meta = rmod.load_tfidf(args.index_dir)
    hybrid = rmod.HybridRetriever(vectorstore=vectorstore, vectorizer=vectorizer,
                                   tfidf_matrix=tfidf_matrix, chunks_meta=chunks_meta,
                                   top_k=config.TOP_K)
    reranked = rmod.CrossEncoderReranker(base_retriever=hybrid, top_k=config.TOP_K)
    reranked._get_relevant_documents("warm up")  # prime model caches, not counted
    results["index_load_seconds"] = round(time.perf_counter() - t0, 2)

    print("=== Timing hybrid-only retrieval (no reranking) ===")
    times = []
    for q in QUERIES:
        t0 = time.perf_counter()
        hybrid._get_relevant_documents(q)
        times.append(time.perf_counter() - t0)
    results["hybrid_only"] = summarize("hybrid_retrieval_only", times)

    print("=== Timing hybrid + reranking (RAGBase's default retrieval config) ===")
    times = []
    for q in QUERIES:
        t0 = time.perf_counter()
        reranked._get_relevant_documents(q)
        times.append(time.perf_counter() - t0)
    results["hybrid_plus_reranking"] = summarize("hybrid_plus_reranking", times)

    if not args.skip_generation:
        print("=== Timing full retrieval + generation (Ollama) ===")
        chain = rmod.build_rag_chain(reranked, config.OLLAMA_MODEL)
        times = []
        for q in QUERIES[: args.generation_queries]:
            t0 = time.perf_counter()
            chain.invoke(q)
            times.append(time.perf_counter() - t0)
        results["retrieval_plus_generation"] = summarize("retrieval_plus_generation", times)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Summary (median / p95 / max, seconds) ===")
    for key in ("hybrid_only", "hybrid_plus_reranking", "retrieval_plus_generation"):
        r = results.get(key)
        if r:
            print(f"  {key}: {r['median_s']} / {r['p95_s']} / {r['max_s']}  (n={r['n']})")
    print(f"\nFull results written to {args.out}")


if __name__ == "__main__":
    main()
