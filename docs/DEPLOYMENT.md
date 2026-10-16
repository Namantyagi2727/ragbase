# Deployment and the public showcase

RAGBase has two experiences, built from the same UI components and styling.

| | Full local mode | Public showcase |
|---|---|---|
| Entry point | `streamlit run app.py` | `showcase/streamlit_app.py` (or `RAGBASE_PUBLIC_SHOWCASE=1 streamlit run app.py`) |
| Inference | Live: Ollama by default; Prism when configured | None. Recorded runs only |
| Documents | Uploads indexed in a per-session workspace; an existing local index is loaded read-only | The curated synthetic corpus, read-only |
| Inputs | Chat, uploads, settings | Selection of curated questions and documents only — no text input |
| Dependencies | `requirements.txt` (PyTorch, embedding and reranker models, FAISS, LangChain, LangGraph) | `showcase/requirements.txt` (Streamlit 1.66.0 only, pinned) |

**Nothing is deployed.** Deployment needs separate approval.

## Intended Streamlit Community Cloud configuration

| Setting | Value |
|---|---|
| Repository / branch | `Namantyagi2727/ragbase`, branch `showcase-deploy` (main plus `maxUploadSize = 1` in the root config; see *Upload size*) |
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

Measured on 2026-10-16 on an Apple Silicon Mac (Python 3.12). Showcase
figures were first taken with Streamlit 1.37.1 and repeated after the update
to the deployed version, 1.66.0. Linux x86-64 figures will differ somewhat;
the ratios are the point.

| Measurement | Public showcase (playback) | Full local stack |
|---|---|---|
| Clean dependency install (uv, empty cache) | 2.1 s (1.37.1); 1.7 s (1.66.0) | 7.3 s |
| Installed environment size | 299 MB (1.37.1); 297 MB (1.66.0) | 1.2 GB (Linux default PyTorch wheels with CUDA are several GB larger) |
| Server start to healthy | 1.0 s (1.37.1); 0.75–1.1 s (1.66.0) | — |
| Process memory, idle | 87 MB (1.37.1); 79 MB (1.66.0) | — |
| Process memory, 1 session | 107 MB | — |
| Process memory, 5 concurrent sessions, each switching questions | 107 MB, 0.3 % CPU (1.37.1) | — |
| Process memory after desktop and mobile sessions and upload probes | 83–103 MB (1.66.0) | — |
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

### Upload size (decided: deployment branch)

The root `.streamlit/config.toml` on `main` sets `maxUploadSize = 50` for
local uploads, and Community Cloud reads only the root config. The showcase
is therefore deployed from `showcase-deploy`, whose only difference is
`maxUploadSize = 1`.

This is a cap, not a prohibition. Verified on Streamlit 1.66.0 with the cap
set to 1 MB, using a real browser session (websocket session id plus XSRF
token):

| Request | Result |
|---|---|
| Upload without the XSRF token | 403 |
| Upload with an invalid session id | 400 |
| 0.1 MB and 0.9 MB under a real session | 204 — accepted into that session's memory |
| 1.2 MB and 5 MB under a real session | 413 — rejected while streaming, before buffering |

Accepted files are held in memory for that session and never read, indexed
or written to disk by the showcase, which has no upload widget and no code
that retrieves uploaded files. Residual risk: a client can hold up to about
1 MB per upload request in server memory for the life of its session, bounded
in total by the platform's memory limit (an exhausted app restarts).

### Preflight fixes made for Streamlit 1.66

- The theme hides Streamlit's header; in 1.66 the control that reopens a
  collapsed sidebar sits inside it, so a collapsed sidebar could not be
  reopened. The control is now kept visible.
- The showcase sidebar starts collapsed on phones (`initial_sidebar_state =
  "auto"`), and the recorded-runs disclosure plus a hint on choosing
  questions now also appear on the main page.
