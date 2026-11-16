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
| `OPENAI_API_KEY` | — (required) | Embeddings via OpenAI |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant endpoint |
| `EMBEDDING_PROVIDER` | `openai` | Embedding provider selection (unknown values fail fast) |
| `MAX_FILE_SIZE_MB` | `200` | Upload rejection limit |
| `MAX_PAGES` | `1000` | PDF page-count limit |
| `CHUNK_SIZE` | `500` | Chunker target size (characters) |
| `CHUNK_OVERLAP` | `50` | Adjacent-chunk overlap (characters) |
| `EMBEDDING_BATCH_SIZE` | `100` | Texts per embedding API call |

**Port conflict on 6333?** Set `QDRANT_PORT` in `.env` (host port) and point
`QDRANT_URL` at the new port — never edit `docker-compose.yml` directly.

## Project layout

```
src/app/
├── api/                          # HTTP layer (tenant-scoped routers)      [T7]
├── application/preprocessing/    # dispatchers + clean/chunk/embed handlers [T4, T5]
├── domain/                       # Pydantic entities + OVM base             [T2]
│   └── enums.py                  # DataCategory, DocumentStatus             [T1]
├── ingestion/                    # pipeline orchestration + validation      [T4, T6]
├── infrastructure/
│   ├── settings.py               # typed configuration                      [T1]
│   ├── qdrant.py                 # connection + collection bootstrap        [T2]
│   └── embedding/                # EmbeddingProvider protocol + OpenAI impl [T3]
└── main.py                       # FastAPI app factory                      [T7]
```
