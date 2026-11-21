# Customer Support Agent — Document Ingestion Pipeline (MVP)

A FastAPI backend that accepts PDF / TXT / .docx uploads for a named tenant and runs
the ingestion pipeline **synchronously** — clean → chunk → embed → index — returning
the document's terminal status (`ready` or `failed` with reason) in the response.
All persistence lives in a single Qdrant instance.

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
| `MCP_SERVERS` | `{}` | MCP tool servers (JSON, langchain-mcp-adapters client format) |
| `MCP_TENANT_ALLOWLIST` | `{}` | Per-tenant tool allowlist (JSON, tenant → [tools]); deny-by-default |
| `CHECKPOINT_BACKEND` | `sqlite` | LangGraph checkpointer: `sqlite` or `postgres` (unknown fails fast) |
| `SQLITE_CHECKPOINT_PATH` | `checkpoints.sqlite` | Sqlite checkpointer file (local runtime state, gitignored) |
| `POSTGRES_DSN` | `postgresql://app:app@localhost:5432/app` | Postgres DSN — transcripts (and the postgres checkpointer); the compose service starts by default |

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
│   ├── llm/                      # purpose-keyed chat-model factory + fake-
│   │                             #   injection seam (verification)          [graphs T1]
│   └── checkpoint.py             # sqlite | postgres checkpointer wiring    [graphs T1]
└── main.py                       # FastAPI app factory                      [T7]
```

New domain records (both payload-only Qdrant points, tenant-scoped like
`DocumentRecord`): `domain/approvals.py` — `ApprovalRecord` (collection
`approvals`; `id` = `request_id` = LangGraph `thread_id`) and
`domain/action_results.py` — `ActionResultRecord` (collection
`action_results`; immutable execution audit). Both collections are created
at app bootstrap.

## Approvals & checkpointing (graphs T7)

The approval workflow is the system's only *stateful* graph: a run gated by
an approval policy pauses at an `interrupt()` (checkpointed under
`thread_id = request_id`) and resumes when a decision is recorded —
durable across processes via the checkpointer (`CHECKPOINT_BACKEND`).

**Deploy rule — drain pending approvals before changing the approval
graph.** LangGraph checkpoints are node-topology-sensitive: a pending
approval's checkpoint references the exact node structure that paused it.
Before deploying structural changes to `graphs/approval/` (renaming,
adding, or removing nodes), decide or otherwise drain every pending
approval — an old checkpoint resumed against a new topology fails.

Operational constraints:

- The sqlite checkpointer (`CHECKPOINT_BACKEND=sqlite`, the default) is
  **single-writer** — fine for a single-process deployment; run multiple
  workers only with the postgres backend (`CHECKPOINT_BACKEND=postgres`;
  the compose postgres service starts by default).
- Decision locking is process-local (per-`request_id` asyncio lock in
  `graphs/runtime.py`); the same-moment double-decision guarantee holds
  within one worker. Multi-worker needs the postgres path plus a
  DB-level guard.
- `checkpoints.sqlite` is local runtime state (gitignored); deleting it
  abandons any paused approvals — their `ApprovalRecord`s stay `pending`
  in the queue forever (no expiry mechanism yet).

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

**Postgres is a default boot dependency for this subsystem** —
`docker compose up -d` starts it alongside Qdrant. The bootstrap is
degrade-graceful: without Postgres at boot the app still starts and
documents/search/approvals behave normally, while the conversational
surfaces return 500 until it is healthy. `CHECKPOINT_BACKEND` (sqlite
default) is independent of the transcript store.
