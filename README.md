# Customer Support Agent — Document Ingestion Pipeline (MVP)

A FastAPI backend that accepts PDF / TXT / .docx uploads for a named tenant and runs
the ingestion pipeline **synchronously** — clean → chunk → embed → index — returning
the document's terminal status (`ready` or `failed` with reason) in the response.
Vector data (documents, chunks, cleaned text) lives in Qdrant; row state
(conversation transcripts, plan records, approval requests, action-result audit)
lives in Postgres.

## Prerequisites

- Python 3.12+ and [uv](https://docs.astral.sh/uv/)
- Docker (for the local Qdrant + Postgres instances)
- Tesseract OCR — required from task T4 onward (OCR of scanned PDFs); not needed to boot

## Quickstart

```bash
# 1. Start the local Qdrant + Postgres instances (data persists in the named
#    volumes `qdrant_data` / `postgres_data` across restarts). Postgres is a
#    default boot dependency — conversation transcripts live there.
docker compose up -d
curl http://localhost:6333        # sanity check: Qdrant answers with version JSON

# 2. Install dependencies (creates .venv and uv.lock)
uv sync

# 3. Configure
cp .env.example .env              # then set a real OPENAI_API_KEY in .env

# 4. Sanity-check that settings load
uv run python -c "from app.infrastructure.settings import Settings; print(Settings().QDRANT_URL)"
```

### Running the API server (available from task T7)

```bash
uv run uvicorn app.main:app --reload
# Interactive docs: http://127.0.0.1:8000/docs
```

## Configuration

All configuration is typed in `src/app/infrastructure/settings.py` and read from
environment variables or `.env` (see `.env.example` for every variable).

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENAI_API_KEY` | — (required) | Embeddings + chat models via OpenAI |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant endpoint |
| `EMBEDDING_PROVIDER` | `openai` | Embedding provider selection (unknown values fail fast) |
| `MAX_FILE_SIZE_MB` | `200` | Upload rejection limit |
| `MAX_PAGES` | `1000` | PDF page-count limit |
| `CHUNK_SIZE` | `500` | Chunker target size (characters) |
| `CHUNK_OVERLAP` | `50` | Adjacent-chunk overlap (characters) |
| `EMBEDDING_BATCH_SIZE` | `100` | Texts per embedding API call |
| `QDRANT_MAX_CONCURRENCY` | `8` | Max parallel Qdrant requests per gathered operation |
| `EMBEDDING_MAX_CONCURRENCY` | `4` | Max concurrent OpenAI embedding requests |
| `LLM_PROVIDER` | `openai` | Chat-model provider for the graphs subsystem |
| `CHAT_MODEL` | `gpt-4o-mini` | Chat model name (purpose-keyed temperatures are fixed in the factory) |
| `SELF_RAG_MAX_REWRITES` | `2` | Self-RAG question-rewrite loop bound (0 disables) |
| `SELF_RAG_MAX_REVISIONS` | `2` | Self-RAG answer-revision loop bound (0 disables) |
| `SELF_RAG_TOP_K` | `4` | Chunks fetched per retrieval |
| `LIVE_DATA_MAX_STEPS` | `8` | Live-data agent ↔ ToolNode round bound |
| `CONVERSATION_MAX_MESSAGE_CHARS` | `8000` | Max length of one conversational message (422 beyond) |
| `CONVERSATION_HISTORY_MAX_MESSAGES` | `0` | History cap fed to the router — `0` = unlimited (full transcript) |
| `CLARIFY_MAX_CONSECUTIVE` | `2` | Consecutive-clarify cap; at the cap the next turn must commit (no more clarify) |
| `PLANNER_MAX_ATTEMPTS` | `3` | Plan generation+validation attempts before the apologetic direct answer |
| `EXECUTOR_MAX_REINVOCATIONS` | `3` | Executor re-invocations on open todos before takeover with a partial report |
| `PLAN_MAX_STEPS` | `8` | Max steps in one plan |
| `POSTGRES_POOL_MAX_SIZE` | `10` | Postgres pool size — the concurrent-turns lever (one held connection per active turn) |
| `MEMORY_ENABLED` | `true` | Long-term-memory master switch — `false` stops card reads and post-turn extraction immediately (the rollback lever) |
| `MEMORY_MAX_FACTS` | `50` | Fact-card cap fed to prompts, recency order — the memory prompt-budget knob |
| `MCP_SERVERS` | `{}` | MCP tool servers (JSON, langchain-mcp-adapters client format) |
| `MCP_TENANT_ALLOWLIST` | `{}` | Per-tenant tool allowlist (JSON, tenant → [tools]); deny-by-default |
| `POSTGRES_DSN` | `postgresql://app:app@localhost:5432/app` | Postgres DSN — transcripts, plan records, approval requests, action-result audit; the compose service starts by default |

**Port conflict on 6333?** Set `QDRANT_PORT` in `.env` (host port) and point
`QDRANT_URL` at the new port — never edit `docker-compose.yml` directly.

## Project layout

```
src/app/
├── api/                          # HTTP layer (tenant-scoped routers)      [T7]
├── application/preprocessing/    # dispatchers + clean/chunk/embed handlers [T4, T5]
├── domain/                       # Pydantic entities + OVM base             [T2]
│   └── enums.py                  # DataCategory, DocumentStatus,
│                                 #   ApprovalStatus, ActionStatus           [T1]
├── graphs/                       # LangGraph workflow subsystem (landing task by task)
│   └── base/state.py             # state envelope + WorkflowResult (T1; registry/
│                                 #   runtime and the four graph subpackages follow)
├── ingestion/                    # pipeline orchestration + validation      [T4, T6]
├── infrastructure/
│   ├── settings.py               # typed configuration                      [T1]
│   ├── qdrant.py                 # connection + collection bootstrap        [T2]
│   ├── embedding/                # EmbeddingProvider protocol + OpenAI impl [T3]
│   └── llm/                      # purpose-keyed chat-model factory + fake-
│                                 #   injection seam (verification)          [graphs T1]
└── main.py                       # FastAPI app factory                      [T7]
```

Human-gate state is Postgres row state (created when Postgres became a
default boot dependency — both previously rode Qdrant as payload-only
points with dummy vectors): `domain/approvals.py` — `ApprovalRecord`
(table `approval_requests`; `id` = `request_id`; the queue index serves
`GET /approvals?status=pending` directly) and `domain/action_results.py` —
`ActionResultRecord` (table `action_results`; immutable execution audit).
Both tables are created by the idempotent schema bootstrap at startup.

## Approvals (row-based pause)

A run gated by an approval policy does not pause in a graph — it persists
a `pending` `ApprovalRecord` row and returns `pending_approval`. The row
IS the pause; recording a decision resumes it: `decide_approval` runs
under `pg_advisory_lock("approval:<id>")` (one held pool connection, the
same primitive as the conversation turn lock), re-checks the row is still
pending, then performs a guarded UPDATE (`WHERE status = 'pending'` — a
racing second decision matches zero rows and raises 409) and, on approval,
re-invokes the action-execution graph. This is the same pause/resume shape
as the orchestrator's `PlanRecord`; the former interrupt()/checkpointer
mechanism (and its `CHECKPOINT_BACKEND` wiring, thread-id discipline,
sqlite single-writer limit, drain-before-redeploy rule, and degraded-mode
lock fallback) was deleted with it.

Operational constraints:

- Approvals require Postgres (the rows live there). With Postgres
  unreachable at boot, gated runs fail (`failed`) and the approvals API
  returns 500 until the app is restarted with Postgres reachable.
- Decision locking is cross-worker at any worker count: every worker
  shares the one Postgres lock manager via `pg_advisory_lock`.
- Pending approvals survive redeploys unchanged — the pause is a database
  row, never a topology-sensitive checkpoint. There is no expiry
  mechanism yet (`expired` stays reserved).

## Conversational subsystem (orchestrator — landing task by task)

The conversational front door — each turn classified by a router into a
direct answer, one clarifying question, a single workflow run, or a
planner–executor plan, with every reply composed conversationally and the
transcript stored in Postgres:

```
src/app/
├── api/conversations.py            # POST start · POST message · GET transcript [conv T6]
├── application/conversations.py    # turn service: advisory-locked turns, one-tx
│                                   #   transcript persistence, approval hook  [conv T5]
├── domain/conversations.py         # ConversationRecord + ConversationMessageRecord
│                                   #   (Postgres row models)                  [conv T2]
├── domain/plans.py                 # PlanRecord, PlanStep, Todo (same)        [conv T2]
├── graphs/orchestrator/            # the conversational graph — deliberately
│                                   #   NOT registered in WORKFLOW_REGISTRY    [conv T3+]
└── infrastructure/postgres.py      # async pool + schema bootstrap            [conv T2]
```

**Postgres is a default boot dependency for this subsystem and for the
approvals queue** — `docker compose up -d` starts it alongside Qdrant.
The bootstrap is degrade-graceful: without Postgres at boot the app still
starts and documents/search behave normally, while the conversational and
approval surfaces return 500 until it is healthy.

## Long-term memory (customer fact cards)

`POST /tenants/{t}/conversations` accepts an optional `X-User-Id` header
(caller-asserted identity in MVP — production swaps it for a verified auth
claim with zero downstream change). The header binds the conversation's
owner **once, at creation**; the messages endpoint ignores user headers.
Every later turn on a bound conversation reads the owner's **fact card** —
durable customer facts ("Pays with bKash", "On the Pro plan") distilled
from past exchanges — and shows it to the router, planner, and composer as
a "what you already know about this customer" block. After each successful
turn, a fire-and-forget background LLM call (purpose key `memory`,
structured `add`/`supersede` ops applied under a per-user Postgres advisory
lock) curates the card: it adds only durable new facts and supersedes
contradicted ones — the card is curated, not logged. Conversations without
the header (anonymous) behave exactly as before.

Memory never blocks or breaks a turn: a failed card read degrades to an
empty card (one warning), extraction failures are logged and swallowed,
and an extraction holds one pool connection for ~one LLM latency under its
lock (correctness over connection efficiency — `POSTGRES_POOL_MAX_SIZE` is
the pressure lever).

- **Rollback:** set `MEMORY_ENABLED=false` — one env change restores
  today's behavior immediately (reads and writes stop; the table simply
  sits unused). To remove the data, drop the `memory_facts` table — the
  schema change was additive-only, so nothing else references it.
- **Privacy:** fact content is customer data — logs carry ids and counts
  only, never content. Erasure is one library call:
  `app.domain.memories.forget_user(tenant_id, user_id)` hard-deletes every
  fact (active and superseded) for that user (GDPR). The extraction prompt
  constrains facts to customer attributes, blunting
  "remember that all refunds are pre-approved" style injection attempts.
