# Deployment and the public showcase

RAGBase has two experiences, built from the same UI components and styling.

| | Full local mode | Public showcase |
|---|---|---|
| Entry point | `streamlit run app.py` | `showcase/streamlit_app.py` (or `RAGBASE_PUBLIC_SHOWCASE=1 streamlit run app.py`) |
| Inference | Live: Ollama by default; Prism when configured | None. Recorded runs only |
| Documents | Uploads indexed in a per-session workspace; an existing local index is loaded read-only | The curated synthetic corpus, read-only |
| Inputs | Chat, uploads, settings | Selection of curated questions and documents only — no text input |
| Dependencies | `requirements.txt` (PyTorch, embedding and reranker models, FAISS, LangChain, LangGraph) | `showcase/requirements.txt` (Streamlit only, pinned) |

**Nothing is deployed.** Deployment needs separate approval.

## Intended Streamlit Community Cloud configuration

| Setting | Value |
|---|---|
| Repository / branch | `Namantyagi2727/ragbase`, `main` after the Phase 4 PR is merged (see *Upload size* below for a deployment-branch alternative) |
| Main file path | `showcase/streamlit_app.py` |
| Python | 3.12 (the version the lock was generated for) |
| Dependencies | `showcase/requirements.txt` — a dependency file next to the entry point takes precedence over the root `requirements.txt`, so the heavy local stack is not installed |
| Config | Root `.streamlit/config.toml` (Community Cloud reads only the root config): theme, usage statistics off |
| Secrets | None. The showcase needs no keys; Prism and Ollama settings are not used |

Community Cloud limits (published by Streamlit, may change without notice):
about 2.7 GB memory at most (figure dated February 2024; older guidance said
1 GB), and apps without traffic for 12 hours go to sleep. A sleeping app shows
a wake-up page; any visitor can wake it, and the cold start then repeats.

## Resource measurements

Measured on 2026-10-16 on an Apple Silicon Mac (Python 3.12, Streamlit
1.37.1). Linux x86-64 figures will differ somewhat; the ratios are the point.

| Measurement | Public showcase (playback) | Full local stack |
|---|---|---|
| Clean dependency install (uv, empty cache) | 2.1 s | 7.3 s |
| Installed environment size | 299 MB | 1.2 GB (Linux default PyTorch wheels with CUDA are several GB larger) |
| Server start to healthy | 1.0 s | — |
| Process memory, idle | 87 MB | — |
| Process memory, 1 session | 107 MB | — |
| Process memory, 5 concurrent sessions, each switching questions | 107 MB, 0.3 % CPU | — |
| Hybrid retrieval in-process (measured, **not enabled**) | +3.9 s import, ~620 MB peak, index build 1.3 s, query p50 31 ms | same stack |

The recorded data (about 125 KB of JSON) is cached once and shared, so extra
sessions add little memory. The GitHub Actions live evaluation (Linux, CPU)
measured about 0.9 GB for the Python process of the full stack, not counting
Ollama.

### Live retrieval in the showcase

Not enabled in the initial version. It would fit the published memory
ceiling, but it adds PyTorch and two model downloads to every cold start and
opens an arbitrary-input surface. If enabled later it must use the same
hybrid retriever (labelled as such), be built in memory from the curated
corpus without loading pickle files, and add input limits and rate limits;
it would show retrieved passages only, never a generated answer.

## Recorded runs

`scripts/record_showcase.py` runs the curated questions (`showcase/curation.json`)
through the real pipeline — Standard and Agentic RAG, the benchmark retrieval
configuration, `llama3.2:3b` via Ollama with seed 1001 — and writes
`showcase/data/recorded_runs.json` with the commit (it refuses to run with
uncommitted changes), model digest, Ollama version, configuration and per-run
timestamps. The answer views are stored exactly as produced. Curator notes
are kept separately in `showcase/curation.json`, written after recording
against the human-reviewed reference answers, and never alter recorded output.
The set includes successes, a partial answer, a correct refusal, flagged
citations and known limitations (false premise accepted, incomplete
multi-document answer).

## Security checklist

| Area | Public showcase | Full local mode |
|---|---|---|
| Uploads | No upload widget; the showcase code never reads uploaded files. Streamlit's upload endpoint still accepts a file for any active session (held in memory until the session ends, capped by `server.maxUploadSize`, which is fixed at server start) | Allowlisted extensions, safe file names, per-file size and file-count limits; files written only to the session workspace |
| Code execution / tabular | Tabular module never imported | Allowlisted dataframe operations only; model output is never executed |
| Ingestion, retrieval, inference | Modules never imported (tested by inspecting loaded modules and the source) | As designed |
| Session isolation | No per-session files; shared data is read-only | Per-session temporary workspaces; the retriever cache is shared across sessions but keyed by index directory |
| Retention | Nothing user-supplied is stored | Workspaces are removed when the session builds a new index or after being idle for 6 hours (best effort, checked when sessions start). Streamlit has no reliable end-of-session hook, so files can outlive a session until then. Session state is held in server memory for the life of the session. Existing shared documents and indexes are never deleted |
| Pickle / index loading | No pickle files loaded; data is JSON | Only indexes RAGBase built itself (session workspace or the existing local index) are loaded; uploaded files are never deserialised |
| HTML / Markdown | All document text, model output and provenance escaped; model answers rendered as plain text, not Markdown (prevents injected image beacons and disguised links) | Same |
| Prompt injection | Curated synthetic documents only; nothing is generated | Documents are untrusted data in every prompt; injected instructions can still influence a small model's answer — citations and verification help detect, not prevent, this |
| Input bounds / rate limits | No free-text input; only selection widgets | Questions capped at 2,000 characters (`RAGBASE_MAX_QUESTION_CHARS`); agent bounded by time budget, revision and search limits. No per-user rate limiting: the local app is single-user by design |
| Logging | No query logging (nothing to log) | Questions are not logged by RAGBase; Ollama and Prism keep their own logs (Prism caches responses for one hour) |
| Configuration and keys | No secrets required or read | `PRISM_API_KEY` read from the environment only, never displayed or logged; Prism controls appear only when it is set |
| Telemetry | Streamlit usage statistics disabled | Same |

### Upload size (decision needed before deployment)

The root `.streamlit/config.toml` sets `maxUploadSize = 50` for local uploads,
and Community Cloud reads only the root config. Options:

1. Deploy from a dedicated branch whose only difference is a stricter root
   config (`maxUploadSize = 1`). Recommended: the showcase cannot use uploads
   anyway, and this caps memory abuse of the upload endpoint.
2. Deploy from `main` and accept the residual risk (bounded by the platform's
   memory limit; an abused app restarts).
