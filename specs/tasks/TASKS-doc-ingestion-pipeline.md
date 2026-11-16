# Tasks

> **Date:** 2026-11-16
> **Source:** specs/architecture/ARCH-doc-ingestion-pipeline.md (REQ: specs/requirements/REQ-doc-ingestion-pipeline.md)
> **Developer decisions made during task generation:** (1) uv is the package manager; (2) `docker-compose.yml` added to T1 footprint for local Qdrant; (3) **no unit/integration test suite** — the only committed test file is `tests/test_isolation.py` (T7); every other verification below is a run-once checklist command, not a committed test.

## Task T1: Project Scaffold, Settings, Enums, Docker Compose

> **Status:** done
> **Verification:** checklist
> **Effort:** s
> **Priority:** critical
> **Depends on:** None
> **Satisfies REQs:** N/A — enabling task (R10/N3 tunable limits live in settings)
> **Footprint slice:** New: pyproject.toml, .env.example, README.md, src/app/infrastructure/settings.py, src/app/domain/enums.py + developer-approved addition docker-compose.yml
> **High-risk areas touched:** Qdrant instance (M) — compose file defines the local instance; port/volume conventions set here

### Description

Bootstraps the empty repo into a runnable uv-managed Python 3.12+ project with the src-layout package skeleton (`src/app/...`), typed configuration, the two domain enums, a Docker Compose definition for the local Qdrant instance, and a README quickstart. Nothing application-specific yet — this is what every later task installs against and imports from.

### Verification Checklist

- **`uv sync`** — expected: exits 0; `uv.lock` and `.venv` created with all runtime + dev dependencies.
- **`docker compose up -d` then `curl http://localhost:6333`** — expected: Qdrant answers with version JSON (HTTP 200).
- **`cp .env.example .env` (dummy OPENAI_API_KEY) then `uv run python -c "<construct settings object>"`** — expected: exits 0, settings load from env/.env.
- **Same construction with `.env` absent and env vars unset** — expected: clear pydantic "field required" error naming the missing variable, not a stack-trace mystery.
- **`uv run pytest --version`** — expected: prints a version (pytest wired; no test files exist at this stage — the only test file arrives in T7).
- **`uv run python -c "<assert enum members>"`** — expected: `DocumentStatus` has exactly `pending, processing, ready, failed`; `DataCategory` values are `pdfs / texts / docxs`.
- **README quickstart followed verbatim by a human** — expected: every command works as written (compose up → uv sync → uvicorn boot instruction documented for T7).

### Implementation Notes

- **Module(s):** `infrastructure/` (settings), `domain/` (enums) — per ARCH Module Boundaries.
- **Pattern reference:** pydantic-settings typed settings; handbook-style enums.
- **Key decisions:** A5 (EMBEDDING_PROVIDER config-driven selection), A10 (limits explicit and tunable in settings).
- **Libraries:** fastapi, uvicorn, pydantic, pydantic-settings, qdrant-client, pymupdf, pytesseract, python-docx, langchain-text-splitters, openai; dev: pytest, httpx (for T7's in-process API client).
- **Settings must carry:** `OPENAI_API_KEY`, `QDRANT_URL` (default `http://localhost:6333`), `EMBEDDING_PROVIDER` (default `openai`), plus tunables consumed by later tasks: max file size MB (200), max pages (1000), chunk size (500), chunk overlap (50), embedding batch size (100).
- **High-risk callouts:** Qdrant compose sets port 6333 + a named volume so container restarts keep data; document in `.env.example` that port conflicts are resolved by editing `.env`, not the compose file.

### Scope Boundaries

- Do NOT add any Qdrant client/collection code (T2).
- Do NOT add CI config, Makefile, pre-commit, or linting setup — nothing in ARCH asks for them.
- Do NOT create test files (developer decision 2026-11-16; only `tests/test_isolation.py` exists, in T7).
- Do NOT add auth of any kind (MVP has none; flagged future requirement).
- Only implement: packaging, package skeleton with `__init__.py` files, settings, enums, compose, env example, README.

### Files Expected

**New files:** _(from ARCH "New files / modules")_
- `pyproject.toml` + `uv.lock` (project + dependency lock — uv-managed per developer decision)
- `.env.example` (OPENAI_API_KEY, QDRANT_URL, EMBEDDING_PROVIDER)
- `docker-compose.yml` (Qdrant service, port 6333, named volume — added by developer decision during task generation, 2026-11-16)
- `README.md` (quickstart: compose → uv sync → run)
- `src/app/infrastructure/settings.py` (typed settings incl. tunable limits)
- `src/app/domain/enums.py` (`DataCategory`, `DocumentStatus`)
- `src/app/__init__.py` and package `__init__.py` files for the src-layout skeleton

**Modified files:** none (greenfield).

**Must NOT modify:**
- `specs/` (read-only planning context).

---

## Task T2: Qdrant Infrastructure, OVM Base, Domain Entities

> **Status:** done
> **Verification:** checklist
> **Effort:** l
> **Priority:** critical
> **Depends on:** T1
> **Satisfies REQs:** R3 (record persistence), R5/N1 (structural isolation), R6/R7 (delete-by-filter foundation), R8 (content_hash field)
> **Footprint slice:** New: src/app/infrastructure/qdrant.py, src/app/domain/base/vector.py, src/app/domain/documents.py, src/app/domain/chunks.py, src/app/domain/records.py
> **High-risk areas touched:** Qdrant instance (M) — new collections, tenant_id payload indexes, vector-dim binding

### Description

Builds the entire data layer: the Qdrant connection/bootstrap module, the `VectorBaseDocument` OVM base class (all Qdrant access flows through it), and all ten Pydantic entities — `DocumentRecord`, three `Cleaned*Document`, and `Chunk`/`EmbeddedChunk` × three formats. This is where tenant isolation gets its structural guarantee: query methods require `tenant_id` with no default.

### Verification Checklist

_Run-once scratch commands (`uv run python -c` or a throwaway script, documented in the implement session, not committed) against `docker compose up -d` Qdrant:_

- **Round-trip mapping script** — expected: entity → Qdrant point → entity for all 10 classes prints "no field differences" (id, tenant_id, content, metadata, flags, timestamps all survive).
- **Chunk id determinism** — expected: same content + document_id produces the identical printed id twice (MD5-derived — enables idempotent re-uploads).
- **Signature check** — expected: `inspect.signature` shows `tenant_id` has **no default** in `bulk_find` / `search` / `delete_by_filter`; entity construction without `tenant_id` raises a pydantic ValidationError. _(N1/R5 structural guarantee)_
- **Insert into a nonexistent collection** — expected: collection auto-created, all points stored; `curl http://localhost:6333/collections` (or dashboard) shows the collection and its `tenant_id` payload index.
- **Tenant scoping** — expected: after inserting points for tenants A and B, `bulk_find(tenant_id=A)` and `search(vector, tenant_id=A)` print **only A's** points, never B's. _(R5 precursor)_
- **Surgical delete** — expected: `delete_by_filter(tenant_id, document_id)` removes exactly that document's points; a sibling document in the same tenant survives. _(R6/R7 foundation)_
- **Qdrant-down resilience** — expected: with the container stopped (`docker compose stop`), a public method call logs the failure and returns a failure result without raising; after `docker compose start`, the same call succeeds. _(ARCH forward stress-test)_

### Implementation Notes

- **Module(s):** `domain/` + `infrastructure/qdrant.py`; domain imports only the Qdrant connection — per Module Boundaries.
- **Pattern reference:** handbook `vector.py` (OVM base: `to_point()`/`from_record()`, public/private method pairs, inner `Config` class declaring collection name + category), handbook cleaned-doc/chunk entity hierarchy.
- **Key decisions:** A2 (Qdrant-only store), A3 (shared collections + mandatory tenant filter), A4 (tenant_id no default — default-to-nothing), A7 (delete = record-first then chunk purge, no transactions), A8 (only EmbeddedChunk collections are vector-indexed), A11 (content_hash SHA-256 field on DocumentRecord).
- **Libraries:** qdrant-client, pydantic v2.
- **Entity specifics:** `DocumentRecord` payload-only (dummy 1-dim vector, `use_vector_index = false`), status enum from T1, `flags` dict (`duplicate_of`, `has_unreliable_tables`, `ocr_used`); `Cleaned*Document` carry `platform` ("digital-pdf"/"ocr-pdf"/"text"/"docx"); chunks carry `metadata.unreliable_table` and later `embedding_model_id` (written by T5).
- **High-risk callouts:** collections are vector-dim-bound — auto-create must take the dimension from the first batch so T2 stays independent of the embedding provider (T3); bootstrap indexes `tenant_id` on every collection so filters don't degrade.
- Scratch verification must clean up its collections (drop or unique suffix) so reruns stay green.

### Scope Boundaries

- Do NOT write any HTTP, pipeline, or handler code (T4–T7).
- Do NOT implement embedding provider code (T3).
- Do NOT create test files (developer decision 2026-11-16).
- Do NOT persist raw file bytes anywhere (A12 — no raw file retention).
- Only implement: connection/bootstrap, OVM base, the 10 entities, package `__init__.py` files.

### Files Expected

**New files:** _(from ARCH "New files / modules")_
- `src/app/infrastructure/qdrant.py` (connection, collection bootstrap, tenant_id payload indexes — handbook pattern)
- `src/app/domain/base/vector.py` (`VectorBaseDocument` incl. `delete_by_filter` — handbook vector.py)
- `src/app/domain/documents.py` (`CleanedDocument` + PDF/TEXT/DOCX subclasses)
- `src/app/domain/chunks.py` (`Chunk`, `EmbeddedChunk` + 3 subclasses each)
- `src/app/domain/records.py` (`DocumentRecord`, payload-only)

**Modified files:** none.

**Must NOT modify:**
- T1 files except adding imports/`__init__.py` wiring if needed.

---

## Task T3: EmbeddingProvider Protocol, OpenAI Implementation, Factory

> **Status:** not started
> **Verification:** checklist
> **Effort:** s
> **Priority:** high
> **Depends on:** T1
> **Satisfies REQs:** R1, R2 (embedding infrastructure)
> **Footprint slice:** New: src/app/infrastructure/embedding/base.py, openai_provider.py, factory.py
> **High-risk areas touched:** Qdrant instance (M) — provider dim (1536) binds collection vector size; singleton exposes it for collection creation

### Description

The pluggable embedding layer: a `Protocol` (`embed(texts) -> list[list[float]]` plus `dim` / `model_id` / `max_input_length` properties), the OpenAI `text-embedding-3-small` implementation with batching, a config-driven factory (`EMBEDDING_PROVIDER` setting), and a singleton wrapper so chunking (T5) can adapt to the model. Same pattern stubbed for a future `LLMProvider` — do not build that stub.

### Verification Checklist

- **Real embed smoke** (`uv run python -c` with valid `.env`) — expected: `embed(["hello"])` returns 1 vector of 1536 floats; `model_id`, `dim`, `max_input_length` print correct values. (Costs a fraction of a cent.)
- **`EMBEDDING_PROVIDER=bogus`** — expected: factory/import fails fast at startup with a clear "unknown embedding provider" message. _(A5)_
- **Batching** — expected: embedding 250 short texts logs exactly 3 provider calls (100/100/50) per the batch-size setting.
- **Singleton** — expected: two accesses of the factory/singleton print the same instance id.
- **Missing OPENAI_API_KEY** — expected: clear pydantic/settings error at construction, not an opaque OpenAI 401 deep in a call.

### Implementation Notes

- **Module(s):** `infrastructure/` only; external packages only — per Module Boundaries.
- **Pattern reference:** handbook `EmbeddingModelSingleton`.
- **Key decisions:** A5 (OpenAI small behind provider interface; unknown provider fails fast).
- **Libraries:** openai SDK.
- **High-risk callouts:** `dim` (1536) will bind every chunk collection's vector size (T2 note) — expose it accurately; `max_input_length` is consumed by T5's chunking adaptation.

### Scope Boundaries

- Do NOT implement HuggingFace/local providers (out of scope; plug-compatible later via the Protocol).
- Do NOT create an LLMProvider implementation (separate REQ; interface location stubbed in ARCH only).
- Do NOT create test files; verification uses the one-off commands above.
- Only implement: Protocol, OpenAI provider, factory + singleton.

### Files Expected

**New files:** _(from ARCH "New files / modules")_
- `src/app/infrastructure/embedding/base.py` (`EmbeddingProvider` Protocol)
- `src/app/infrastructure/embedding/openai_provider.py` (OpenAI impl, batched)
- `src/app/infrastructure/embedding/factory.py` (config-driven factory + singleton)

**Modified files:** none.

**Must NOT modify:**
- T1/T2 files (read settings/entities only).

---

## Task T4: Upload Validation + Per-Format Cleaning Handlers

> **Status:** not started
> **Verification:** checklist
> **Effort:** m
> **Priority:** high
> **Depends on:** T1, T2
> **Satisfies REQs:** R2 (OCR), R9 (table flag), R10 (rejections), N3 (limits)
> **Footprint slice:** New: src/app/ingestion/validation.py, src/app/application/preprocessing/cleaning_data_handlers.py
> **High-risk areas touched:** None (M/H)

### Description

The VALIDATE and EXTRACT/CLEAN stages: validation checks (password-protected, corrupted, page count over limit) returning human-readable reason strings, and the three format cleaning handlers — PDF with embedded-text detection routing to digital extraction or Tesseract OCR, TEXT passthrough, DOCX paragraphs + best-effort tables. Tables are never rejected: extracted best-effort and flagged unreliable when quality is doubtful.

### Verification Checklist

_Fixtures are generated by documented scratch commands (pymupdf text PDF; text rendered to an image wrapped as a PDF for the scanned case; pymupdf-encrypted PDF; deliberately truncated bytes; 1001 blank-page PDF; python-docx document containing a table; a plain .txt):_

- **Digital PDF** — expected: handler returns the embedded text verbatim; `platform="digital-pdf"`; no OCR invoked.
- **Scanned (image-only) PDF** — expected: no embedded text detected → OCR path; text from the image is recovered; `platform="ocr-pdf"`; `ocr_used` flag true. _(R2)_
- **Password-protected PDF** — expected: rejected with reason string "password-protected". _(R10 / REQ edge case)_
- **Corrupted file (truncated bytes)** — expected: reason "corrupted". _(R10 / REQ edge case)_
- **1001-page PDF** — expected: size-limit reason; limit read from T1 settings, not hardcoded. _(R10/N3 / REQ edge case)_
- **.docx containing a table** — expected: table text present in extracted content **and** the unreliable-table flag set. _(R9)_
- **.txt** — expected: content unchanged; `platform="text"`.

### Implementation Notes

- **Module(s):** `ingestion/` (validation) + `application/preprocessing/` (handlers) — handlers depend on domain entities only.
- **Pattern reference:** handbook cleaning handlers; dispatcher wiring is NOT here (T5 owns `dispatchers.py` — handlers are invoked directly for this task's verification).
- **Key decisions:** A9 (OCR = local Tesseract, routed by embedded-text detection; tables best-effort + flag), A10 (explicit validation limits with reason strings).
- **Libraries:** pymupdf (embedded-text detection, page count, table hints), pytesseract (+ rasterization via pymupdf), python-docx.
- Handlers produce `Cleaned*Document` entities (T2) with `platform`, extraction stats in `metadata`, table flags.

### Scope Boundaries

- Do NOT implement chunking or embedding (T5).
- Do NOT map validation results to HTTP status codes (413/422 mapping is T7's; this layer returns reason strings).
- Do NOT add cloud OCR or OCR quality tuning (out of scope).
- Do NOT create test files; verification uses generated fixtures + scratch commands.
- Only implement: validation module, 3 cleaning handlers.

### Files Expected

**New files:** _(from ARCH "New files / modules")_
- `src/app/ingestion/validation.py` (size/pages/password/corruption checks — R10, N3)
- `src/app/application/preprocessing/cleaning_data_handlers.py` (3 handlers incl. PDF OCR routing + table flag)

**Modified files:** none.

**Must NOT modify:**
- T1–T3 files.

---

## Task T5: Chunking + Embedding Handlers, Dispatchers and Factories

> **Status:** not started
> **Verification:** checklist
> **Effort:** m
> **Priority:** high
> **Depends on:** T1, T2, T3, T4
> **Satisfies REQs:** R1 (chunk + embed stages)
> **Footprint slice:** New: src/app/application/preprocessing/chunking_data_handlers.py, embedding_data_handlers.py, dispatchers.py
> **High-risk areas touched:** Future agent service (M) — chunk payload shape is the durable contract (`embedding_model_id`, `unreliable_table` in metadata); Qdrant instance (M) — vectors written must match provider dim

### Description

The CHUNK and EMBED stages plus the routing layer: per-format chunking handlers (LangChain `RecursiveCharacterTextSplitter`, size/overlap from settings), per-format embedding handlers that batch chunks through the provider into `Embedded*Chunk` entities, and the three dispatchers with their factories that route any stage by `DataCategory`. After this task the whole preprocessing stack is callable; only orchestration (T6) is missing.

### Verification Checklist

_Uses inline stub provider classes defined in the scratch commands (call-counting, tiny max_input_length) — no API cost, no committed files:_

- **Chunk stats script on a long text** — expected: every chunk ≤ 500 chars (chunk_size setting), adjacent chunks share ~50 chars overlap (chunk_overlap setting); stats printed.
- **Provider-limit adaptation** — expected: with a stub provider reporting a tiny `max_input_length`, oversized chunks are split further before reaching the provider; nothing over the limit is handed over.
- **Embedding batching** — expected: 250 chunks through the embedding handler with a call-counting stub → exactly 3 calls of ≤ 100.
- **Routing table** — expected: dispatching pdf/text/docx payloads through each stage prints the correct handler class for all 9 combinations (3 formats × clean/chunk/embed).
- **Unknown category** — expected: explicit error, never a silent wrong route.
- **Metadata contract** — expected: produced `Embedded*Chunk` entities carry `embedding_model_id` in metadata (from the provider) and propagate `unreliable_table`. _(future-agent contract)_

### Implementation Notes

- **Module(s):** `application/preprocessing/` — depends on domain entities + infrastructure provider only.
- **Pattern reference:** handbook dispatchers.py (Factory + Strategy), chunking/embedding handlers.
- **Key decisions:** A6 (dispatcher pattern — adding a format = new handlers + one factory branch, zero pipeline/API changes), A5 (provider singleton consumed here).
- **Libraries:** langchain-text-splitters (`RecursiveCharacterTextSplitter` — REQ Decision 10 preference).
- Chunk ids must use the T2 MD5(content + document_id) derivation so re-uploads upsert idempotently.
- **High-risk callouts:** the EmbeddedChunk payload shape is the v1 contract the future agent reads — `embedding_model_id` and `unreliable_table` must land in metadata here, not later.

### Scope Boundaries

- Do NOT add formats beyond PDF/TEXT/DOCX (out of scope; structure makes them drop-in later).
- Do NOT implement the LOAD stage / persistence loops — pipeline (T6) groups and bulk-inserts.
- Do NOT create test files.
- Only implement: 3 chunking handlers, 3 embedding handlers, 3 dispatchers + 3 factories.

### Files Expected

**New files:** _(from ARCH "New files / modules")_
- `src/app/application/preprocessing/chunking_data_handlers.py` (3 chunking handlers)
- `src/app/application/preprocessing/embedding_data_handlers.py` (3 embedding handlers, batched)
- `src/app/application/preprocessing/dispatchers.py` (3 dispatchers + 3 factories)

**Modified files:** none.

**Must NOT modify:**
- T1–T4 files.

---

## Task T6: Synchronous Ingestion Pipeline + Document Lifecycle

> **Status:** not started
> **Verification:** checklist
> **Effort:** m
> **Priority:** critical
> **Depends on:** T4, T5
> **Satisfies REQs:** R1–R4, R6, R7, R8, N2
> **Footprint slice:** New: src/app/ingestion/pipeline.py
> **High-risk areas touched:** Qdrant instance (M) — record-first/chunk-purge ordering (A7), no transactions; Future agent service (M) — A8 only-ready guarantee

### Description

The orchestrator: runs validate → clean → chunk → embed → load synchronously to a terminal state, driving the `DocumentRecord` lifecycle (`pending → processing → ready|failed`), plus the lifecycle operations the REQs demand — duplicate flagging (SHA-256), replace-by-name (old record + chunk purge), and delete. Also owns temp-file cleanup and the per-document outcome log line. The initial `pending` record creation lives in the API layer (T7); this task's verification creates records via scratch script.

### Verification Checklist

_Scratch script against Docker Qdrant; failure injection via a stub provider that raises; status/points inspected via the script and `curl http://localhost:6333/collections/...`:_

- **Happy path digital PDF** — expected: record transitions `pending → processing → ready` (printed in order); `num_chunks > 0`; chunk points present in Qdrant; the `ready` record is written **after** chunks are inserted. _(R1, R3, A8)_
- **Embedding stage failure** — expected: status `failed` with a reason mentioning the embedding service; **zero** chunk points inserted for that document. _(R4, N2, ARCH forward stress-test "OpenAI API down"; A8)_
- **Duplicate upload** (same bytes, same tenant) — expected: second record stored and flagged `duplicate_of` pointing at the first record's id. _(R8)_
- **Replace-by-name** — expected: after uploading v1 then v2 under the same filename, v1's record and chunk points are gone; only v2's content is retrievable. _(R6, A7 record-first ordering)_
- **Delete of a ready document** — expected: record + chunk points removed; search returns nothing for its content. _(R7)_
- **Validation failure wiring** — expected: a password-protected file flows to `failed` with the T4 reason string, no chunks. _(R10, N2)_
- **Logging** — expected: exactly one outcome log line per document (`ready`/`failed` + reason) with stage metadata (num_chunks, chunk params, embedding model id, extraction path, flags). _(ARCH observability)_

### Implementation Notes

- **Module(s):** `ingestion/` — orchestrates `application/preprocessing`, `domain`, `infrastructure`; never HTTP.
- **Pattern reference:** handbook pipeline steps minus ZenML.
- **Key decisions:** A1 (synchronous, in-request — no worker/queue), A7 (replace/delete = record-first, then chunk purge; orphan points invisible, accepted), A8 (chunks only after full success; `ready` written last), A11 (content_hash checked **before** pipeline runs), A12 (temp file deleted after processing; no raw retention).
- One-way status transitions; `failed` never silently becomes `ready`.
- **High-risk callouts:** no Qdrant transactions exist — A7 ordering is the mitigation; verify replace/delete orderings exactly as the checklist states.

### Scope Boundaries

- Do NOT add background workers, queues, retries, or dead-letter machinery (out of scope — operator re-uploads).
- Do NOT implement orphan-point sweeping (ARCH open question, deferred).
- Do NOT create HTTP endpoints (T7) or test files.
- Only implement: pipeline.py orchestration + lifecycle operations.

### Files Expected

**New files:** _(from ARCH "New files / modules")_
- `src/app/ingestion/pipeline.py` (synchronous orchestration + status transitions — handbook pipeline minus ZenML)

**Modified files:** none.

**Must NOT modify:**
- T1–T5 files.

---

## Task T7: FastAPI App + Tenant-Scoped API + Permanent Isolation Test

> **Status:** not started
> **Verification:** checklist
> **Effort:** m
> **Priority:** critical
> **Depends on:** T6
> **Satisfies REQs:** R1, R3, R4, R5, R6, R7, N1, N3 (R8/R9 surfaced via flags)
> **Footprint slice:** New: src/app/main.py, src/app/api/documents.py, src/app/api/search.py, tests/test_isolation.py
> **High-risk areas touched:** Future agent service (M) — `/search` is the durable v1 external contract; isolation standing guard lives here

### Description

The HTTP surface and the security keystone: the FastAPI app factory, the tenant-scoped document endpoints (upload → synchronous pipeline → terminal response; list; get; delete) and the search endpoint the future agent consumes. Also lands `tests/test_isolation.py` — the single permanent automated test, guarding R5/N1 forever. Tenant identity comes only from the URL path; `tenant_id` is never client-supplied.

### Verification Checklist

- **`uv run uvicorn app.main:app`** — expected: boots without error; `/docs` (FastAPI auto-generated docs page) loads.
- **POST digital PDF upload** — expected: 200 with `{document_id, status: "ready", flags, num_chunks}` — terminal status in the response body. _(R1 via API, N2)_
- **Oversize upload → 413** — expected: rejected *before* the body finishes streaming. Exercise cheaply by temporarily setting the MB limit very low via env — no real 200 MB file needed. _(N3)_
- **Unsupported format (.xlsx)** — expected: 422 with a clear message.
- **GET list + GET single** — expected: statuses, flags, reasons visible; GET with a wrong tenant id → 404 indistinguishable from a wrong document id (no tenant-existence oracle). _(R3)_
- **DELETE then DELETE again** — expected: 204 both times (idempotent); after the first, search no longer returns the doc's content. _(R7)_
- **POST `/tenants/{tid}/search`** with a query whose answer is literally in an uploaded doc — expected: passage + document_id + score + metadata returned, including `unreliable_table` when flagged. _(R1, R9)_
- **`uv run pytest`** — expected: `tests/test_isolation.py` passes: upload the same document for tenants A and B, search as A, assert zero B content in every response; skips with a clear message when Docker Qdrant or `OPENAI_API_KEY` is unavailable. _(R5/N1 — the ARCH's permanent standing guard)_

### Implementation Notes

- **Module(s):** `api/` — HTTP parsing, tenant extraction from path, response shaping; calls ingestion pipeline; reads domain; never touches Qdrant directly (Module Boundaries).
- **Pattern reference:** FastAPI routers; httpx/TestClient for the isolation test (in-process, no server needed).
- **Key decisions:** A3 (tenant identity from URL path only), A4 (server-derived tenant_id, never client-supplied), A8 (response carries terminal status), plus ARCH error rules: 413/422 for upload validation, 200-with-`failed` for document-level problems, 500 only for infrastructure outages.
- Upload flow: hash file (SHA-256) → create `DocumentRecord(pending)` → temp file → pipeline → delete temp file (A12).
- **High-risk callouts:** `/search` is the durable v1 contract — response shape `{content, document_id, score, metadata}` from ready docs of the path tenant only; any drift here breaks the future agent. The isolation test is the only committed test in the repo (developer decision 2026-11-16) — it must remain the permanent CI-style guard.

### Scope Boundaries

- Do NOT add authentication/API-key middleware (MVP has none; ARCH flags it as required before real tenants — future task).
- Do NOT add endpoints beyond the five in ARCH's API contract table.
- Do NOT build any UI (out of scope).
- Do NOT add test files beyond `tests/test_isolation.py` (developer decision 2026-11-16).
- Only implement: app factory, router registration, 5 endpoints, isolation test.

### Files Expected

**New files:** _(from ARCH "New files / modules")_
- `src/app/main.py` (FastAPI app factory, router registration)
- `src/app/api/documents.py` (upload / list / get / delete endpoints)
- `src/app/api/search.py` (tenant-scoped retrieval endpoint)
- `tests/test_isolation.py` (permanent tenant-isolation guard — R5/N1; the only committed test per developer decision 2026-11-16)

**Modified files:** none.

**Must NOT modify:**
- T1–T6 files except router-registration imports wired through `main.py`.

---

_All seven tasks are `checklist` verification mode per developer decision of 2026-11-16: no unit/integration test suite; run-once verification commands documented per task; `tests/test_isolation.py` (T7) is the sole permanent automated check._
