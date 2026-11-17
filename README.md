# Customer Support Agent — Document Ingestion Pipeline (MVP)

A FastAPI backend that accepts PDF / TXT / .docx uploads for a named tenant and runs
the ingestion pipeline **synchronously** — clean → chunk → embed → index — returning
the document's terminal status (`ready` or `failed` with reason) in the response.
All persistence lives in a single Qdrant instance.

## Prerequisites

- Python 3.12+ and [uv](https://docs.astral.sh/uv/)
- Docker (for the local Qdrant instance)
- Tesseract OCR — required from task T4 onward (OCR of scanned PDFs); not needed to boot

## Quickstart

```bash
# 1. Start the local Qdrant instance (data persists in the named volume
#    `qdrant_data` across restarts)
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
| `LLM_PROVIDER` | `openai` | Chat-model provider for the graphs subsystem |
| `CHAT_MODEL` | `gpt-4o-mini` | Chat model name (purpose-keyed temperatures are fixed in the factory) |
| `SELF_RAG_MAX_REWRITES` | `2` | Self-RAG question-rewrite loop bound (0 disables) |
| `SELF_RAG_MAX_REVISIONS` | `2` | Self-RAG answer-revision loop bound (0 disables) |
| `SELF_RAG_TOP_K` | `4` | Chunks fetched per retrieval |
| `LIVE_DATA_MAX_STEPS` | `8` | Live-data agent ↔ ToolNode round bound |
| `MCP_SERVERS` | `{}` | MCP tool servers (JSON, langchain-mcp-adapters client format) |
| `MCP_TENANT_ALLOWLIST` | `{}` | Per-tenant tool allowlist (JSON, tenant → [tools]); deny-by-default |
| `CHECKPOINT_BACKEND` | `sqlite` | LangGraph checkpointer: `sqlite` or `postgres` (unknown fails fast) |
| `SQLITE_CHECKPOINT_PATH` | `checkpoints.sqlite` | Sqlite checkpointer file (local runtime state, gitignored) |
| `POSTGRES_DSN` | `postgresql://app:app@localhost:5432/app` | Postgres checkpointer DSN (run `docker compose --profile postgres up -d`) |

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
