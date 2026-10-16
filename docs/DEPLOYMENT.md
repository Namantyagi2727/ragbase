# Deployment and the public showcase

RAGBase has two experiences, built from the same UI components and styling.

| | Full local mode | Public showcase |
|---|---|---|
| Entry point | `streamlit run app.py` | `showcase/streamlit_app.py` (or `RAGBASE_PUBLIC_SHOWCASE=1 streamlit run app.py`) |
| Inference | Live: Ollama by default; Prism when configured | None. Recorded runs only |
| Documents | Uploads indexed in a per-session workspace; an existing local index is loaded read-only | The curated synthetic corpus, read-only |
| Inputs | Chat, uploads, settings | Selection of curated questions and documents only — no text input |
| Dependencies | `requirements.txt` (PyTorch, embedding and reranker models, FAISS, LangChain, LangGraph) | `showcase/requirements.txt` (Streamlit 1.66.0 only, pinned) |

**Live showcase:** https://ragbase-showcase.streamlit.app/ — deployed
2026-10-16 from branch `showcase-deploy`. Every answer it shows is a
recorded run; it performs no live inference.

## Streamlit Community Cloud configuration (as deployed)

| Setting | Value |
|---|---|
| Repository / branch | `Namantyagi2727/ragbase`, branch `showcase-deploy` (main plus two root-config overrides: `maxUploadSize = 1` and `headless = true`) |
| Main file path | `showcase/streamlit_app.py` |
| Python | 3.12 (deployed environment reported Python 3.12.15) |
| Dependencies | `showcase/requirements.txt` — a dependency file next to the entry point takes precedence over the root `requirements.txt`, so the heavy local stack is not installed |
| Config | Root `.streamlit/config.toml` (Community Cloud reads only the root config): theme, usage statistics off |
| Secrets | None. The showcase needs no keys; Prism and Ollama settings are not used |

Community Cloud limits (published by Streamlit, may change without notice):
about 2.7 GB memory at most (figure dated February 2024; older guidance said
1 GB), and apps without traffic for 12 hours go to sleep. A sleeping app shows
a wake-up page; any visitor can wake it, and the cold start then repeats.

## Resource measurements (local)

These are **local** measurements, not production measurements. Measured on
2026-10-16 on an Apple Silicon Mac (Python 3.12). Showcase
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
is therefore deployed from `showcase-deploy`, which sets `maxUploadSize = 1`
(and, since the first deployment, `headless = true`; see *Deployment record*).

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

## Deployment record

| | |
|---|---|
| Live URL | https://ragbase-showcase.streamlit.app/ |
| Phase 4 merge into `main` | `f5b6d87` (PR #5, merge commit) |
| Deployment branch | `showcase-deploy` |
| Upload cap override | `4d293a2` (`server.maxUploadSize = 1`) |
| Headless override (deployment-only fix) | `569162e` (`server.headless = true`) |

### First deployment failed; cause verified

The first deployment (from `4d293a2`) installed its dependencies correctly —
from `showcase/requirements.txt` with uv on Python 3.12.15, 36 packages, no
PyTorch, models, FAISS or LangChain — and then never started: the platform
health check reported `connection refused`. The log showed Streamlit's
first-run "enter your email" prompt. Streamlit shows that prompt when
`server.headless` is false, which is `main`'s setting (it opens a browser for
local use); with no interactive input the server waits indefinitely.
Reproduced locally with no Streamlit credentials file and no stdin: with
`headless = false` the server never became healthy; with `headless = true`
it did. `569162e` sets `headless = true` on `showcase-deploy` only; after
the push the platform redeployed and the app was running at the first status
check (05:20 UTC).

Platform behaviour seen in the build log: Community Cloud replaced the
pinned `pyarrow 26.0.0` with `24.0.0`, citing a known segfault
(apache/arrow#50471). The showcase works with it. The platform also noted
that the repository contains several dependency files and used
`showcase/requirements.txt`, as intended.

### Live acceptance checks (2026-10-16, anonymous visitor)

Checked in a browser with no Streamlit or GitHub session.

| # | Check | Observed |
|---|---|---|
| 1 | Anonymous access | Page loads; the platform's context endpoint reported an anonymous user and no viewer authentication |
| 2 | All ten curated questions | Each of the ten selections loaded its own question |
| 3 | Standard and Agentic side by side | Both columns present for all ten, each labelled "— recorded", with the reviewed reference answer |
| 4 | Citations, evidence, documents, traces, methodology | Cited and uncited passage cards, verification boxes and the full agent trace (8 steps for the hotel-approval question, including the two documents searched) rendered; the documents tab showed the two-column driver handbook with its columns read separately, plus corpus version and checksum; all methodology sections present |
| 5 | Recorded-run disclosure and provenance | Disclosure on the main page and in the sidebar; provenance panels (two per question) show commit `1aef28a`, model digest `a80c4f17acd5` and seed 1001 |
| 6 | No question box or uploads | No chat input, text area, file input or uploader. One text field exists: the document dropdown's built-in type-to-filter, which only narrows the eight document names in the browser |
| 7 | No live inference or heavy ML dependencies | Build log lists Streamlit's dependencies only; the platform reports Streamlit 1.66.0; the showcase code imports no inference, retrieval-model, agent or tabular modules |
| 8 | No secrets or private documents | Page source contains no API-key names or key-like strings, no local paths, no local document or index names, and no personal email; all documents are the synthetic Northwind corpus |
| 9 | Resource usage and restart | A real redeploy (after `569162e`) completed and the app served normally afterwards; the public status endpoint reported no CPU throttling |
| 10 | Desktop and mobile layouts | Readable at 1440 px; at 375 px the sidebar starts collapsed with a visible reopen control and the disclosure is on the main page |

### Production measurements not available

Platform memory, CPU usage and an exact cold-start time are visible only in
the app owner's Community Cloud view and were not recorded; they are not
estimated here. The build log shows about 11 seconds from cloning the
repository to finishing dependency processing on the first deployment.

### Upload endpoint on the live app

The 1 MB cap is set on `showcase-deploy` and its behaviour was verified
locally on the same Streamlit version (see *Upload size*). A probe of the
live app was **inconclusive**: through the platform's proxy the test could
not read an XSRF cookie; a 0.5 MB upload under a test session returned HTTP
400 and a 1.5 MB upload received no response within 20 seconds. The probe
was stopped rather than repeated against production. The showcase never
reads or stores uploaded files; the residual risk is up to about 1 MB per
upload request held in a session's memory, bounded by the platform's limit.

### Known polish item (not scheduled)

At narrow desktop widths (around 1,100 px) the agent-trace header in each
half-width column wraps into a narrow column; it remains readable and is
fine at 1,440 px and on phones. Optional future polish.

## Updating the deployment

`showcase-deploy` is `main` plus two configuration overrides in the root
`.streamlit/config.toml`: `maxUploadSize = 1` and `headless = true`.
Community Cloud redeploys automatically when the branch changes. To bring
approved changes from `main` into the showcase:

```bash
git checkout showcase-deploy
git pull --ff-only origin showcase-deploy
git merge --no-ff main            # merge commit; never rebase or force-push this branch
# If .streamlit/config.toml conflicts, keep main's other changes but retain
# both overrides: maxUploadSize = 1 and headless = true.
git diff main showcase-deploy     # must show only the two overrides (and their comments)
git push origin showcase-deploy
```

Then reopen the live URL and repeat the acceptance checks that the change
could affect. Only approved changes that have passed CI on `main` should be
merged into the deployment branch; recordings in `showcase/data/` change
only when re-recording is approved.
