# Using RAGBase with the Prism gateway

RAGBase can send its model calls through [Prism](https://github.com/Namantyagi2727/prism),
an OpenAI-compatible LLM gateway. This is optional: direct local Ollama remains
the default, and RAGBase never falls back from one to the other.

```
RAGBASE_LLM_BACKEND=ollama   # default — RAGBase → Ollama (fully local)
RAGBASE_LLM_BACKEND=prism    # RAGBase → Prism → provider chosen by Prism's route
```

> **Current status:** Prism's guardrails block every RAGBase prompt that
> contains retrieved evidence, so Prism mode cannot answer document questions
> yet. See [Guardrail compatibility](#guardrail-compatibility).

## Inference and privacy modes

| Mode | When | Where document text goes |
|---|---|---|
| **Direct local** | `RAGBASE_LLM_BACKEND=ollama` | Ollama on this machine only |
| **Prism, local model** | `prism`, and the response's `provider_used` is a local provider (`ollama`) | Prism (Docker, this machine) → Ollama on this machine. Also held in Prism's Redis cache (see below) |
| **Prism, remote provider** | `prism`, with a genuine remote provider configured in Prism *and* added to `PRISM_ALLOWED_PROVIDERS` | Leaves the machine. Not available today: Prism's `openai` and `anthropic` providers are mocks |

Every result records the mode of each call (`inference_mode`) from what the
gateway reported, not from configuration alone.

### What Prism stores

From Prism's source (`src/prism/api/chat.py`, `core/cache.py`, `db/models.py`):

- **Response cache:** the full response for every successful request is stored
  in Redis for **1 hour**, keyed by a hash of model, messages and temperature.
  RAGBase responses are derived from document content, so document-derived text
  sits in Redis for that hour.
- **Request log (Postgres):** metadata only — key, virtual model, provider,
  cache/fallback flags, token counts, cost, latency, status, guardrail reason.
  Prompts and responses are not written to the log.
- **Guardrail scan:** Presidio and regex checks run in-process on the full
  prompt (system, evidence and question). Nothing is sent elsewhere.
- **Tracing:** OpenTelemetry instruments HTTP requests (method, route, status,
  timing); request bodies are not added to spans by Prism's code.

## Provider verification (fail closed)

RAGBase accepts a Prism response only when `prism_metadata.provider_used` names
a provider in `PRISM_ALLOWED_PROVIDERS` (default `ollama`). It rejects, with
`ProviderRejectedError`, and discards the text of:

- responses from any other provider — including Prism's `openai` and
  `anthropic` **mocks**, which the `fast` route falls back to when Ollama fails;
- **cache hits** (`provider_used: "cache"`), because Prism does not record
  which provider produced a cached response;
- responses with missing or empty provider metadata.

Mock-looking payloads (`[mock … response]`, `chatcmpl-mock` IDs) are also
rejected, as a second line of defence. A rejected response never reaches
answer synthesis or verification: planning or synthesis ends the run as
`failed`; a rejected verifier response leaves the answer `not_verified`.

**Consequence:** repeating an identical request within an hour is served from
Prism's cache and therefore rejected. Different questions are unaffected;
repeated evaluation runs are.

## Request fields and capabilities

Prism's `/v1/chat/completions` accepts only `model`, `messages`,
`temperature` and `stream` (non-streaming only). Other fields are dropped, so:

| Capability | Direct Ollama | Through Prism |
|---|---|---|
| Temperature | sent | **always sent explicitly** (Prism defaults to 0.7) |
| Structured output | native JSON-schema decoding | **explicit JSON instructions** in the prompt + validation + one repair attempt |
| Context length | `num_ctx` set by RAGBase (4096) | **not controllable**; Ollama server default (4096 on the test machine) |
| Max tokens, streaming, tool calls | not used | not supported |
| Token counts | from Ollama | from Prism's `usage`, when present |
| Cost, fallback, gateway latency | — | recorded only when Prism reports them |

## Guardrail compatibility

Measured with Prism's own `scan_prompt` function run in-process from Prism's
environment (`scripts/prism/scan_guardrails.py`), on the exact prompts RAGBase
builds (`scripts/prism/build_guardrail_prompts.py`). This measures the
guardrail decision only, not HTTP behaviour.

**Finding 1 — every evidence prompt is blocked.** RAGBase's evidence IDs
(`E1`, `E2`, …) match Presidio's `US_DRIVER_LICENSE` recognizer (score 0.3),
and Prism blocks on any match. Prompts without evidence (the planner) pass.

| Corpus (standard RAG prompts) | Blocked |
|---|---|
| Synthetic travel policies (no personal data), 6 questions | 6/6 |
| Driver's manual, 6 questions | 6/6 |
| 1,200-document novel corpus, 30 questions | 30/30 |
| Synthetic guardrail fixtures, all roles with evidence | 21/21 |

**Finding 2 — with a non-matching ID format, content still decides.** Rewriting
the IDs to `E-1` style before scanning (measurement only; RAGBase still sends
`E1`) gives:

| Corpus | Blocked | Reasons |
|---|---|---|
| Synthetic travel policies | 0/6 | — |
| Driver's manual | 3/6 | PERSON |
| Novel corpus | 29/30 | PERSON (28), prompt injection (1) |
| Fixture: plain policy | pass | — |
| Fixture: named approvers | blocked | PERSON |
| Fixture: emails / phone | blocked | EMAIL_ADDRESS, PHONE_NUMBER |
| Fixture: IP ranges | blocked | IP_ADDRESS |
| Fixture: licence number | blocked | US_DRIVER_LICENSE |
| Fixture: "ignore previous instructions" in a document | blocked | prompt injection |
| Fixture: benign "you are now eligible…" | blocked | prompt injection (false positive) |

Blocking is the designed behaviour for a gateway protecting user input; it
means ordinary documents that mention people cannot be used through Prism as
configured. RAGBase reports the block reason and does not redact or bypass.

**Options (each needs separate approval):**

1. *RAGBase:* change the evidence-ID format (e.g. `E-1`) so citations stop
   matching the licence pattern. Unblocks documents without personal data;
   requires re-checking prompt behaviour with the local model.
2. *Prism:* a narrowly scoped, authenticated per-key guardrail policy for
   document-grounded applications (for example a score threshold, or scanning
   only the user's question rather than retrieved evidence). Needed for
   documents that mention people.

## Long-context behaviour

`scripts/prism/context_probe.py` plants one fact in the first passage and one
in the last, inside neutral filler, and asks for both. `ollama-openai`
replicates Prism's upstream request exactly (no context option) without going
through Prism; `ollama-native` is RAGBase's direct call.

| Prompt size | Direct (`num_ctx` 4096) | Prism request shape |
|---|---|---|
| ~270–2,700 tokens | both facts found (one end-fact miss at ~1,800) | identical |
| ~4,000 tokens | start found, end missed | identical |
| ≥ ~5,400 tokens | **both lost; prompt cut to ~2,050 tokens** | identical |

- Up to 4,096 tokens the two paths behave identically. The misses below that
  size are not monotonic in length, so they reflect the 3B model, not truncation.
- Above the context size, Ollama truncates silently and still returns success
  (both paths). Through Prism, RAGBase cannot raise the limit; it depends on the
  Ollama server's default (`OLLAMA_CONTEXT_LENGTH`).
- RAGBase's default prompts stay well below the limit (at most 12 evidence
  chunks of 512 characters, roughly 2,000 tokens with instructions).

## Running against a real Prism (optional)

Prerequisites: Docker, the Prism repository, Ollama on the host with
`llama3.2:3b` pulled.

1. **Prism's own `.env`** (in the Prism repo, never committed): the app container
   must reach Ollama on the host, so set
   `OLLAMA_BASE_URL=http://host.docker.internal:11434`. If it points at
   `localhost`, Prism falls back to its mock providers and RAGBase will reject
   every response (which also demonstrates fail-closed behaviour).
2. **Start Prism** (from the Prism repo):
   ```bash
   docker compose up -d
   ```
3. **Run migrations** from the host (the app image does not contain Alembic files):
   ```bash
   POSTGRES_HOST=localhost uv run alembic upgrade head
   ```
4. **Create a test team and API key** with your admin secret (keep it out of
   shell history, e.g. `read -s PRISM_ADMIN_SECRET`):
   ```bash
   curl -s -X POST localhost:8000/admin/teams \
     -H "Authorization: Bearer $PRISM_ADMIN_SECRET" -H "Content-Type: application/json" \
     -d '{"name": "ragbase-test"}'
   curl -s -X POST localhost:8000/admin/keys \
     -H "Authorization: Bearer $PRISM_ADMIN_SECRET" -H "Content-Type: application/json" \
     -d '{"team_id": "<id from the previous response>", "rate_limit_rpm": 120}'
   ```
5. **Configure RAGBase** in its gitignored `.env` (template: `.env.example`):
   ```bash
   PRISM_API_KEY=<key from step 4>
   PRISM_BASE_URL=http://localhost:8000
   ```
6. **Run the live tests and probes** (from the RAGBase repo):
   ```bash
   set -a; source .env; set +a
   pytest -m prism -v -rs
   python scripts/prism/context_probe.py --target prism --sizes 2,12,30
   python -m agent --backend prism --index-dir <index> "What is the daily meal allowance?"
   ```

`pytest -m prism` reports **skipped** when Prism or the key is missing; a
skipped run is not a verified integration.
