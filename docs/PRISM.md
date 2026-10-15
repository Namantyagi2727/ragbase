# Using RAGBase with the Prism gateway

RAGBase can send its model calls through [Prism](https://github.com/Namantyagi2727/prism),
an OpenAI-compatible LLM gateway. This is optional: direct local Ollama remains
the default, and RAGBase never falls back from one to the other.

```
RAGBASE_LLM_BACKEND=ollama   # default — RAGBase → Ollama (fully local)
RAGBASE_LLM_BACKEND=prism    # RAGBase → Prism → provider chosen by Prism's route
```

**Status:** verified end to end against a real local Prism stack (RAGBase →
Prism → Ollama over HTTP) for Standard RAG and Agentic RAG on documents without
personal data. Documents containing names, emails, IP addresses, licence-style
codes or injection-like text are blocked by Prism's guardrails by design.
Prism itself is unmodified.

## Inference and privacy modes

| Mode | When | Where document text goes |
|---|---|---|
| **Direct local** | `RAGBASE_LLM_BACKEND=ollama` | Ollama on this machine only |
| **Prism, local model** | `prism`, and the response's `provider_used` is a local provider (`ollama`) | Prism (Docker, this machine) → Ollama on this machine; also held in Prism's Redis cache |
| **Prism, remote provider** | `prism`, with a genuine remote provider configured in Prism *and* added to `PRISM_ALLOWED_PROVIDERS` | Leaves the machine. **Not available today:** Prism's `openai` and `anthropic` providers are mocks |

Each call's `inference_mode` is recorded from what the gateway reported, not
from configuration alone. All verified runs were **Prism, local model**.

### What Prism stores

From Prism's source (`src/prism/api/chat.py`, `core/cache.py`, `db/models.py`):

- **Response cache:** the full response to every successful request is stored in
  Redis for **1 hour**, keyed by a hash of model, messages and temperature.
  RAGBase responses are derived from document content, so document-derived
  text sits in Redis for that hour.
- **Request log (Postgres):** metadata only — key, virtual model, provider,
  cache/fallback flags, token counts, cost, latency, status, guardrail reason.
  Prompts and responses are not written to the log.
- **Guardrail scan:** Presidio + regex checks run in-process on the full prompt
  (system, evidence and question). Nothing is sent elsewhere.
- **Tracing:** OpenTelemetry instruments HTTP requests (method, route, status,
  timing); Prism's code does not add request bodies to spans.

## Provider verification (fail closed)

A Prism response is accepted only when `prism_metadata.provider_used` names a
provider in `PRISM_ALLOWED_PROVIDERS` (default `ollama`). Otherwise RAGBase
raises `ProviderRejectedError` and discards the text:

| Case | Verified against real Prism |
|---|---|
| Provider outside the allowlist (Prism's `mock-fast` route → mock `openai`) | yes — rejected |
| Cache hit (`provider_used: "cache"`; Prism does not record the original provider) | yes — rejected |
| Missing / empty provider metadata | mocked tests only |
| Mock-looking payload from an allowed provider (second line of defence) | mocked tests only |

A rejected response never reaches answer synthesis or verification: planning
or synthesis ends the run as `failed`; a rejected verifier response leaves the
answer `not_verified`.

**Cache limitation:** an identical request repeated within Prism's one-hour
cache window is served from cache and therefore rejected. Different questions
are unaffected; repeating the same question (including re-running the same
test or evaluation) fails until the cache entry expires. RAGBase does not alter
requests to avoid the cache.

## Citation format and gateway guardrails

RAGBase's evidence citations were changed for Prism compatibility:

- **Before:** `[E1]`, with evidence rendered as `<doc id="E1" …>`. `E1` matched
  Presidio's `US_DRIVER_LICENSE` recognizer (score 0.3), and Prism blocks on any
  match, so **100% of evidence prompts were blocked**.
- **Candidates rejected:** `E-1`, `doc1`, `src1`, `EV1`, … were still tagged as
  `PERSON` by spaCy in some contexts; `#1` scanned clean but lowered citation
  quality with `llama3.2`.
- **Now:** plain numbered citations — `[1]`, `[2, 3]` — with the id attribute
  placed last: `<doc source="…" location="…" id="1">` (`<doc id="…"` was
  itself tagged as a person). Scanned clean with Prism's guardrail across IDs
  1–60 in every prompt pattern. Model output in older styles (`[E1]`, `[E-1]`,
  `[#1]`) is normalized to `[1]`. Chunk IDs and indexes are unchanged.

**Citation quality** (live `llama3.2`, 10 questions × 2 runs, same session):

| | Before (`E1`) | After (`[n]`) |
|---|---|---|
| Standard RAG — citation check passed | 20/20 | 20/20 |
| Standard RAG — mean claim citation coverage | 0.82–0.83 | 0.86–0.87 |
| Agentic RAG — verified / qualified / citations valid | 2–3 / 4 / 20 of 20 | 2 / 4 / 20 of 20 |

With numeric IDs the verifier initially left `evidence_ids` empty; verdicts now
must list the evidence they checked, which restored parity.

**Structured output** also had to change: the words "JSON" and "Respond" and
raw JSON-schema text were tagged as `PERSON`, blocking every structured call
(planner, verifier, abstention check, tabular). The adapter now sends a
plain-language field list (enums as `a|b|c`). Measured directly against Ollama,
prompted JSON (as Prism requires) was valid on the first attempt in 15/16
planner/verifier calls versus 16/16 with native schema decoding; the one
repair attempt recovered the remaining case. One live agent run still saw a
verifier output fail both attempts and was recorded as `not_verified`.

## Guardrail behaviour

Measured two ways: in-process with Prism's own `scan_prompt` on the exact
prompts the adapter sends (`scripts/prism/scan_guardrails.py`), and over HTTP
against the running gateway (live tests and demo).

| Content | Result |
|---|---|
| Policy text without personal data (all prompt roles) | passes |
| Synthetic travel-policy corpus, standard RAG prompts | 0/6 blocked |
| Names (`named_approvers.txt`) | blocked — PERSON *(verified over HTTP)* |
| Emails and phone numbers | blocked — EMAIL_ADDRESS, PHONE_NUMBER |
| IP addresses | blocked — IP_ADDRESS |
| Licence number in a document | blocked — US_DRIVER_LICENSE |
| "Ignore previous instructions…" inside a document | blocked — prompt injection |
| Benign "you are now eligible…" | blocked — prompt injection (false positive) |
| Driver's manual, 6 questions | 3/6 blocked — PERSON |
| Novel corpus (1,200 documents), 30 questions | 29/30 blocked — PERSON (28), injection (1) |
| Random letter-digit codes in a question (e.g. hex IDs, SKUs) | can be blocked — US_DRIVER_LICENSE *(seen over HTTP)* |

Blocking is Prism's designed behaviour for a gateway protecting user input.
RAGBase reports the reason and does not redact, bypass or alter content.
Documents that mention people cannot currently be used through Prism.

## Request fields and capabilities

Prism's `/v1/chat/completions` accepts only `model`, `messages`, `temperature`
and `stream` (non-streaming only). Other fields are dropped:

| Capability | Direct Ollama | Through Prism |
|---|---|---|
| Temperature | sent | **always sent explicitly** (Prism defaults to 0.7) |
| Structured output | native JSON-schema decoding | plain-language field list + validation + one repair |
| Context length | `num_ctx` set by RAGBase (4096) | **not controllable**; Ollama server default (4096 on the test machine) |
| Max tokens, streaming, tool calls | not used | not supported |
| Token counts | from Ollama | from Prism's `usage` |
| Provider, cost, fallback, gateway latency | — | recorded from `prism_metadata` |

## Measured latency (real gateway)

`scripts/prism/live_demo.py`, three Standard RAG questions, each asked once
through Prism and once directly, `llama3.2:3b`, Apple M4 Pro
([`prism_results/prism_live_demo.json`](prism_results/prism_live_demo.json)):

| Question | Answer tokens (Prism / direct) | LLM call via Prism | Prism-reported gateway latency | Direct call |
|---|---|---|---|---|
| Expense deadline | 30 / 27 | 1.06 s | 978 ms | 0.32 s |
| Hotel approval threshold | 29 / 29 | 0.51 s | 426 ms | 0.34 s |
| Leave carry-over | 26 / 26 | 0.88 s | 802 ms | 0.31 s |

Median end to end (retrieval + call): 0.99 s via Prism vs 0.35 s direct. With
answer lengths matched, Prism added roughly 0.2–0.75 s per call in this setup
(guardrail scan, database and cache writes, Docker networking). Three samples
only; treat as indicative. The two agentic questions through Prism each made
5 model calls (one refinement round) and took 10.2 s and 11.0 s; both ended
`qualified` because the 3B model's answers contained contradictory statements
that verification flagged.

## Long-context behaviour (Prism request shape, not through Prism)

Measured before the real gateway was available, by replicating Prism's upstream
request directly against Ollama (`scripts/prism/context_probe.py --target ollama-openai`);
it does not exercise Prism itself. Facts planted in the first and last passage:

| Prompt size | Direct (`num_ctx` 4096) | Prism request shape |
|---|---|---|
| ~270–2,700 tokens | both facts found (one end-fact miss at ~1,800) | identical |
| ~4,000 tokens | start found, end missed | identical |
| ≥ ~5,400 tokens | **both lost; prompt cut to ~2,050 tokens** | identical |

Below the context size the misses are not monotonic in length (the 3B model,
not truncation). Above it Ollama truncates silently on both paths. RAGBase's
default prompts stay around 2,000 tokens at most.

## Running against a real Prism (optional)

Prerequisites: Docker Desktop, the Prism repository, Ollama on the host with
`llama3.2:3b` pulled. No Prism files need editing.

1. **Host networking.** Prism's container must reach Ollama on the host. If
   Prism's `.env` sets `OLLAMA_BASE_URL=http://localhost:11434`, override it for
   the app container with a compose override file kept outside the Prism repo:
   ```yaml
   # prism-override.yml
   services:
     app:
       environment:
         OLLAMA_BASE_URL: http://host.docker.internal:11434
   ```
   Without this, Prism falls back to its mock providers and RAGBase rejects
   every response.
2. **Start Prism** (from the Prism repo):
   ```bash
   docker compose -f docker-compose.yml -f /path/to/prism-override.yml up -d --build app
   ```
3. **Migrations** from the host (the app image does not contain Alembic files):
   ```bash
   POSTGRES_HOST=localhost .venv/bin/alembic upgrade head
   ```
4. **Create a test team and API key** with your admin secret, without echoing it
   (e.g. `read -s PRISM_ADMIN_SECRET`):
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
6. **Run** (from the RAGBase repo):
   ```bash
   set -a; source .env; set +a
   pytest -m prism -v -rs
   python scripts/prism/live_demo.py --out /tmp/prism_demo.json
   python -m agent --backend prism --index-dir <index> "What is the daily meal allowance?"
   ```

`pytest -m prism` reports **skipped** when Prism or the key is missing, and
skips fixed-prompt tests that Prism answers from cache on a re-run; neither
counts as a verified integration.

## Possible later Prism changes (not made; would need approval)

1. Record the originating provider on cache entries, so cached responses can be
   attributed and accepted.
2. A narrowly scoped per-key guardrail policy for document-grounded clients
   (for example a score threshold, or scanning the user's question rather than
   retrieved evidence), so documents mentioning people can be used.
3. Pass through a context-length option (or document the Ollama default) for
   long-context requests.
4. Native structured output (pass `format`/`response_format` to providers that
   support it).
