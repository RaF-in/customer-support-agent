# Tasks

> **Source:** `specs/architecture/ARCH-async-data-layer.md` (read it first — this
> file references its decisions as AD1–AD13 and requirements as AR1–AR9)
> **Generated:** 2026-11-19
> **Convention (AR8/AD13):** no committed tests. Every task verifies via a
> **run-once checklist**; nothing lives in `tests/`.
> **Mid-sweep note (AD2 — one breaking sweep):** after each of T1–T6 the package
> stays **import-clean but runtime-broken** by design (still-sync callers of
> now-async functions). T1–T6 therefore verify **mechanically only** — compile,
> import, grep, signature inspection. All runtime evidence is concentrated in
> T7. Do not "fix" intermediate breakage by adding sync shims or dual APIs.

---

## Task T1: Async foundation — settings caps + AsyncQdrantClient infra

> **Status:** done
> **Verification:** checklist
> **Effort:** s
> **Priority:** critical
> **Depends on:** None
> **Satisfies REQs:** AR1, AR2, AR7
> **Footprint slice:** Modified: `infrastructure/settings.py` (+2 caps), `infrastructure/qdrant.py` (async client, race-safe creates, close), `.env.example`, `README.md`
> **High-risk areas touched:** Data layer (H) — infra half only

### Description

Introduces the two concurrency-cap settings and migrates the Qdrant connection
module to `AsyncQdrantClient`: lazy singleton (boots with Qdrant down, AD11),
race-safe collection/index creation where *only* the already-exists outcome
counts as success (AD10), and an explicit async close for lifespan shutdown.
Everything above this layer is still sync after this task — runtime breakage
is expected and covered by T7, not here.

### Verification Checklist

- **Compile + import:** `uv run python -m compileall -q src/app && uv run python -c "import app.main"` — expected: exit 0 both (import-clean; runtime calls are known-broken until T2–T6)
- **Defaults:** `uv run python -c "from app.infrastructure.settings import Settings; s = Settings(); print(s.QDRANT_MAX_CONCURRENCY, s.EMBEDDING_MAX_CONCURRENCY)"` — expected: `8 4`
- **Validation:** `QDRANT_MAX_CONCURRENCY=0 uv run python -c "from app.infrastructure.settings import Settings; Settings()"` (and same for `EMBEDDING_MAX_CONCURRENCY=0`) — expected: pydantic `ValidationError` (`ge=1` enforced)
- **Async client only:** `grep -nE "QdrantClient" src/app/infrastructure/qdrant.py` — expected: every hit is `AsyncQdrantClient` (no sync client remains)
- **Close entrypoint exists:** `grep -n "aclose" src/app/infrastructure/qdrant.py` — expected: an async close on the connection/client, callable from lifespan shutdown (wired in T5)
- **Race-safe create (code inspection):** `create_collection_if_missing` / `create_tenant_id_index` are `async def`; the except path treats *only* the already-exists response as success — a genuine create failure must propagate (no blanket swallow)
- **Lazy singleton preserved (code inspection):** constructor does no I/O; `get_connection()` still cheap and side-effect-free at import — never called at module import time
- **Settings docs:** `grep -nE "QDRANT_MAX_CONCURRENCY|EMBEDDING_MAX_CONCURRENCY" .env.example README.md` — expected: both keys present in each, styled like the existing tunables

### Implementation Notes

- **Module(s):** `app.infrastructure.qdrant`, `app.infrastructure.settings` (ARCH Module Boundaries)
- **Pattern reference:** current `qdrant.py` lazy `QdrantConnection` — keep the shape, swap the client class; settings follow the existing `Field(default=…, ge=1)` tunables block (`SELF_RAG_TOP_K` style)
- **Key decisions:** AD1 (AsyncQdrantClient over REST, not gRPC), AD4 (caps come from per-call `Settings()` reads — no module-level semaphores *anywhere*, including here), AD10 (already-exists-only catch), AD11 (lazy + `aclose()`)
- **Libraries:** `qdrant_client.AsyncQdrantClient` — already installed, zero new deps (AR7)
- **High-risk callouts:** Data layer is H-risk overall; this task only swaps the transport — the over-catch hazard (409 catch swallowing real failures) is the one to watch and is pinned by the race-safe inspection item above plus T7's live race/failure probes

### Scope Boundaries

- Do NOT add gRPC / `prefer_grpc` (ARCH Out of Scope)
- Do NOT introduce module-level or global semaphores — caps are *read* here, *used* in T2/T3
- Do NOT touch `vector.py` or any caller — they break at runtime until their own tasks; that is the plan
- Do NOT eagerly connect at startup or add health-check pings

### Files Expected

**New files:** none.

**Modified files:**
- `src/app/infrastructure/settings.py` (+`QDRANT_MAX_CONCURRENCY` int default 8 ge=1, +`EMBEDDING_MAX_CONCURRENCY` int default 4 ge=1)
- `src/app/infrastructure/qdrant.py` (AsyncQdrantClient lazy singleton; async race-safe `create_collection_if_missing` / `create_tenant_id_index`; async close)
- `.env.example` (document both caps)
- `README.md` (tunables section, if/where it lists them)

**Must NOT modify:**
- `src/app/domain/base/vector.py` (T2), any caller file (T3–T6)
- `docker-compose.yml` (REST transport unchanged — silent-regression hotspot)
- `tests/test_isolation.py`

---

## Task T2: OVM goes async — VectorBaseDocument + bulk_find_all + retrieve_by_ids

> **Status:** done
> **Verification:** checklist
> **Effort:** m
> **Priority:** critical
> **Depends on:** T1
> **Satisfies REQs:** AR1, AR2, AR5, AR9
> **Footprint slice:** Modified: `domain/base/vector.py` (all methods async; +`bulk_find_all`; +`retrieve_by_ids`; gathered upsert batches)
> **High-risk areas touched:** Data layer (H) — the churn center of the whole sweep

### Description

The one **H**-risk file: every `VectorBaseDocument` method becomes an `async`
classmethod with unchanged semantics (swallow-and-log public / raising private
twins, `wait=True` upserts), plus two new methods — `bulk_find_all` (full
offset-paged scroll, the shared replacement for four copy-pasted loops, AD6)
and `retrieve_by_ids` (tenant-filtered point-id fetch, routing `runtime.py`'s
bypass through the OVM, AD7). Upsert batches gather under a per-call semaphore
sized from fresh `Settings()` (AD4). All callers are still sync after this
task — expected, fixed by T4–T6.

### Verification Checklist

- **Compile + import:** `uv run python -m compileall -q src/app && uv run python -c "import app.main"` — expected: exit 0
- **All async:** `grep -nE "def (bulk_insert|bulk_find|bulk_find_all|search|delete_by_filter|delete_by_ids|retrieve_by_ids)" src/app/domain/base/vector.py` — expected: every hit is `async def`, names unchanged (AR: same method names, new signatures)
- **A4 discipline (mechanical):** `uv run python -c "import inspect; from app.domain.base.vector import VectorBaseDocument as V; [print(n) for n in ('bulk_find','bulk_find_all','search','retrieve_by_ids','delete_by_filter') if inspect.signature(getattr(V,n)).parameters['tenant_id'].default is not inspect.Parameter.empty]"` — expected: **no output** (no query method has a defaultable `tenant_id`; `delete_by_ids` is ids-only by design)
- **New method shapes (code inspection):** `bulk_find_all(tenant_id)` pages via point-id offset until a short page (mirrors the loops it replaces); `retrieve_by_ids(tenant_id, ids)` filters by tenant — wrong tenant reads as empty, never as an error (no oracle)
- **Twin structure (code inspection):** both new methods follow the swallow-and-log public / `_`-prefixed raising private convention; failure returns `None` for finds, `False` for writes
- **Gathered upserts:** `grep -n "Semaphore\|gather\|wait=True" src/app/domain/base/vector.py` — expected: per-call `asyncio.Semaphore` built from a fresh `Settings()` read (`QDRANT_MAX_CONCURRENCY`), batches gathered, `wait=True` retained on every upsert (AR9)
- **No settings capture:** `grep -nE "^_[A-Z]|^[A-Z_]+ = Settings" src/app/domain/base/vector.py` — expected: no module-level `Settings()` capture (house rule)

### Implementation Notes

- **Module(s):** `app.domain.base.vector` — the only Qdrant surface (OVM chokepoint charter)
- **Pattern reference:** the file's own public/private twin layout; the scroll-paging shape to generalize is `retrieval.py:_tenant_records` (pre-migration) — same `limit`/`offset`/short-page contract
- **Key decisions:** AD2 (one sweep, no dual API), AD3 (gather at upserts), AD4 (per-call semaphore), AD6 (`bulk_find_all`), AD7 (`retrieve_by_ids`), AD12 (`wait=True` kept)
- **Libraries:** `asyncio` only
- **High-risk callouts:** H-risk because every Qdrant call site changes signature here. Mitigation: names stay identical, semantics documented in this spec, and T7's final un-awaited-coroutine grep plus the live OVM roundtrip probe cover what mechanical checks cannot

### Scope Boundaries

- Do NOT keep a sync variant of any method (AD2 explicitly rejects dual API)
- Do NOT rename methods or alter return types/semantics
- Do NOT touch `retrieval.py` / `pipeline.py` / `api/*` / `graphs/*` — their loops get replaced in their own tasks
- Do NOT batch `bulk_find_all` differently from the loops it replaces (same page size class of behavior — full scroll, `Optional[list[T]]`)

### Files Expected

**New files:** none.

**Modified files:**
- `src/app/domain/base/vector.py` (async sweep + `bulk_find_all` + `retrieve_by_ids` + gathered batched upserts under semaphore)

**Must NOT modify:**
- Any caller of the OVM (T4–T6 own them)
- `src/app/infrastructure/qdrant.py` (T1 delivered it)

---

## Task T3: Embedding goes async — AsyncOpenAI provider + async dispatch

> **Status:** done
> **Verification:** checklist
> **Effort:** s
> **Priority:** high
> **Depends on:** T1
> **Satisfies REQs:** AR2, AR4, AR7
> **Footprint slice:** Modified: `infrastructure/embedding/base.py`, `infrastructure/embedding/openai_provider.py`, `application/preprocessing/embedding_data_handlers.py`, `application/preprocessing/dispatchers.py`
> **High-risk areas touched:** Embedding layer (M) — order-preservation invariant

### Description

Flips the embedding path to async end to end: the `EmbeddingProvider`
Protocol's `embed` becomes `async def`, `OpenAIEmbeddingProvider` moves to
`AsyncOpenAI` with batches gathered under a semaphore (`EMBEDDING_MAX_CONCURRENCY`),
and the embedding data handler / dispatcher await it. Vectors must come back
**positionally ordered** — gather preserves order, and the per-batch length
check at the handler stays (AD5). Cleaning and chunking dispatch remain sync.

### Verification Checklist

- **Compile + import:** `uv run python -m compileall -q src/app && uv run python -c "import app.main"` — expected: exit 0
- **Protocol:** `grep -n "async def embed" src/app/infrastructure/embedding/base.py` — expected: present; `grep -nE "def (dim|model_id|max_input_length)"` on the same file — expected: still plain (sync) properties
- **Async client only:** `grep -nE "\bOpenAI\(" src/app/infrastructure/embedding/openai_provider.py` — expected: **no hits** (no sync `OpenAI(...)` construction; `AsyncOpenAI` does not match `\bOpenAI\(`)
- **Eager-client pattern kept:** provider still builds the client in `__init__` from typed settings (missing key fails at construction, not mid-embed)
- **Gathered, capped, ordered:** `grep -n "Semaphore\|gather" src/app/infrastructure/embedding/openai_provider.py` — expected: per-call `asyncio.Semaphore` from fresh `Settings()` (`EMBEDDING_MAX_CONCURRENCY`), batches gathered, results reassembled positionally (index/ordered extend — never completion order)
- **Handler awaits:** `grep -n "async def embed\|await" src/app/application/preprocessing/embedding_data_handlers.py` — expected: `embed` is async, awaits the provider; the `len(vectors) != len(batch)` check remains
- **Dispatcher split:** `grep -n "def dispatch" src/app/application/preprocessing/dispatchers.py` — expected: exactly one `async def dispatch` (Embedding); Cleaning/Chunking still plain `def`
- **Batch log line unchanged in shape:** one info line per batch, same fields as today (interleaving order may differ under gather — not a contract)

### Implementation Notes

- **Module(s):** `app.infrastructure.embedding`, `app.application.preprocessing` (embedding handler only)
- **Pattern reference:** current `openai_provider.py` batching loop — keep batching math and logging; T1's settings style for the cap read
- **Key decisions:** AD4 (per-call semaphore from fresh settings), AD5 (`AsyncOpenAI`, order preserved positionally); AR4 scope decision
- **Libraries:** `openai.AsyncOpenAI` — installed, zero new deps
- **High-risk callouts:** M-risk: order preservation under gather. Mechanical check pins the reassembly shape; T7's live order probe (embed 5 texts at batch size 2, compare against single-call vectors) proves it at runtime

### Scope Boundaries

- Do NOT make `dim` / `model_id` / `max_input_length` async
- Do NOT touch cleaning/chunking handlers or dispatchers (sync, zero-diff per ARCH)
- Do NOT parallelize *within* a batch or retry/rate-limit-handle 429s (first error propagates — fail-fast is the contract)
- Do NOT change chunk metadata written by the handler

### Files Expected

**New files:** none.

**Modified files:**
- `src/app/infrastructure/embedding/base.py` (Protocol `embed` → `async def`)
- `src/app/infrastructure/embedding/openai_provider.py` (`AsyncOpenAI`, gathered batches under semaphore)
- `src/app/application/preprocessing/embedding_data_handlers.py` (`embed` → async, awaits provider)
- `src/app/application/preprocessing/dispatchers.py` (`EmbeddingDispatcher.dispatch` → async only)

**Must NOT modify:**
- `src/app/application/preprocessing/` cleaning/chunking handlers (touched-not-changed)
- `src/app/ingestion/pipeline.py` (T4 wires the awaits)

---

## Task T4: Services — async retrieval with gathers + async pipeline with off-loop CPU stages

> **Status:** done
> **Verification:** checklist
> **Effort:** m
> **Priority:** high
> **Depends on:** T2, T3
> **Satisfies REQs:** AR2, AR5, AR6, AR9
> **Footprint slice:** Modified: `application/retrieval.py`, `ingestion/pipeline.py`
> **High-risk areas touched:** Ingestion pipeline (M) — A7/A8/N2 ordering under parallelism

### Description

The orchestration layer goes async. `retrieve_passages` runs its two gathers
(records scroll ∥ query embedding, then the three chunk-class searches) and
replaces its local scroll loop with `bulk_find_all`. The pipeline's
`process`/`delete_document` become async with **one** shared tenant scroll
feeding both replace-purge and duplicate-flag; clean/chunk CPU stages move off
the event loop via `asyncio.to_thread` (AD8 — pymupdf/tesseract must not block
the loop); grouped inserts and purges gather, while A7 record-first deletes,
A8 ready-record-last, `wait=True`, and N2 one-call-terminal survive verbatim.

### Verification Checklist

- **Compile + import:** `uv run python -m compileall -q src/app && uv run python -c "import app.main"` — expected: exit 0
- **Retrieval async:** `grep -n "async def retrieve_passages" src/app/application/retrieval.py` and `grep -c "gather" src/app/application/retrieval.py` — expected: present, and **2** gather sites (scroll ∥ embed; three searches)
- **Loop replaced:** `grep -nE "_tenant_records|_PAGE_SIZE|offset" src/app/application/retrieval.py` — expected: no hits; one `await DocumentRecord.bulk_find_all(...)` call instead
- **Frozen error surface:** `grep -nE "class (RetrievalServiceError|RecordListingError|QueryEmbeddingError|VectorSearchError)|vector db unavailable|embedding service unavailable" src/app/application/retrieval.py` — expected: all four classes and the exact v1 strings still defined/raised (AR3)
- **Pipeline async:** `grep -nE "async def (process|delete_document)" src/app/ingestion/pipeline.py` — expected: both present
- **One shared scroll:** `grep -c "bulk_find_all" src/app/ingestion/pipeline.py` — expected: exactly **1** — `_purge_replaced` and `_flag_duplicate` consume one scan, not two
- **CPU stages off-loop:** `grep -n "to_thread" src/app/ingestion/pipeline.py` — expected: wrapping the cleaning and chunking dispatch sites **only** (embedding dispatch and load are awaited directly)
- **Ordering survives (code inspection):** `_delete_document_points` still deletes the record point *before* chunk/cleaned purges (A7); `_load` still writes the `ready` transition **last** (A8); `_ALLOWED_TRANSITIONS` and the N2 safety-net in `process` are untouched (`git diff` on those blocks shows no change)
- **Gathered writes:** `grep -n "gather" src/app/ingestion/pipeline.py` — expected: grouped chunk/cleaned inserts gathered, and the post-record purges gathered after the record delete

### Implementation Notes

- **Module(s):** `app.application.retrieval` (shared by HTTP + Self-RAG node), `app.ingestion.pipeline`
- **Pattern reference:** the gather/semaphore shape introduced in T2/T3; `_abort_load` purge-on-partial-write behavior stays exactly as is
- **Key decisions:** AD3 (gather sites), AD6 (shared scroll — this is where the four loops die), AD8 (`to_thread` for CPU stages), AD12 (A7/A8/wait=True under parallelism)
- **Libraries:** `asyncio` only
- **High-risk callouts:** M-risk: parallelism must not reorder lifecycle writes. The inspection items above pin the code shape; T7's replace/delete/duplicate flows prove it live. Watch especially that gather never wraps the record-point delete together with purges *behind* it (A7 is order, not just completion)

### Scope Boundaries

- Do NOT parallelize the pipeline stages themselves (clean → chunk → embed stays sequential; only intra-stage batches gather)
- Do NOT change the outcome log line format (one scannable line per document)
- Do NOT add per-(tenant, filename) upload serialization (pre-existing race, explicitly out of scope)
- Do NOT alter validation logic or `_fail` reason strings

### Files Expected

**New files:** none.

**Modified files:**
- `src/app/application/retrieval.py` (async; 2 gathers; `bulk_find_all`; exceptions frozen)
- `src/app/ingestion/pipeline.py` (async process/delete; shared scroll; `to_thread` clean/chunk; gathered inserts/purges; ordering preserved)

**Must NOT modify:**
- `src/app/ingestion/validation.py`, preprocessing cleaning/chunking handlers (touched-not-changed)
- `src/app/api/*`, `src/app/graphs/*` (T5/T6)

---

## Task T5: HTTP surface + app — async handlers, frozen contracts, lifespan close

> **Status:** not started
> **Verification:** checklist
> **Effort:** m
> **Priority:** high
> **Depends on:** T4
> **Satisfies REQs:** AR3, AR5, AR6
> **Footprint slice:** Modified: `api/documents.py`, `api/search.py`, `api/approvals.py`, `main.py`
> **High-risk areas touched:** HTTP surface (L) — but the contract freeze is the point

### Description

Every endpoint flips to `async def` awaiting the now-async services — upload
stops occupying a threadpool and, with T4's off-loop CPU stages, stops
blocking the event loop entirely (AD9, AR6). The local scroll loops in
documents/approvals are replaced by `bulk_find_all` (the last two of the four
loops, AD6). `main.py`'s bootstrap becomes async with gathered collection
creates, and lifespan shutdown awaits the T1 close entrypoint. Routes,
models, status codes, and error strings are **byte-frozen** (AR3).

### Verification Checklist

- **Compile + import:** `uv run python -m compileall -q src/app && uv run python -c "import app.main"` — expected: exit 0
- **All handlers async:** `grep -nA1 "@router" src/app/api/documents.py src/app/api/search.py src/app/api/approvals.py | grep -E "def "` — expected: every shown handler is `async def`
- **Loops replaced:** `grep -nE "offset|_PAGE_SIZE" src/app/api/documents.py src/app/api/approvals.py` — expected: no paging loops; `await ... bulk_find_all(...)` instead (documents' helper may keep its name; its body is the one-liner)
- **Contract frozen:** `git diff src/app/api/` — expected: hunks touch only handler bodies (async/await + scroll replacement); **zero** changes to response models, route paths/methods, status codes, or `detail=` strings
- **Bootstrap async + gathered:** `grep -n "async def _bootstrap_collections\|gather" src/app/main.py` — expected: async bootstrap, collection creates gathered under the T2 semaphore pattern (AD3's sixth gather site); failure still warns and falls back to first-insert creation
- **Shutdown closes:** `grep -n "aclose" src/app/main.py` — expected: lifespan shutdown awaits the qdrant close (AD11); uvicorn drains in-flight requests first, so no request loses its client
- **Upload awaits pipeline:** `grep -n "await IngestionPipeline" src/app/api/documents.py` — expected: present (no `to_thread` — the pipeline is async end to end now)

### Implementation Notes

- **Module(s):** `app.api.*`, `app.main` (composition root)
- **Pattern reference:** `api/search.py`'s try/except → HTTPException mapping stays verbatim; only the call gains `await`
- **Key decisions:** AD9 (all endpoints async, contract frozen), AD6 (last two scroll loops die), AD3 (bootstrap gather), AD11 (shutdown close)
- **Libraries:** none new
- **High-risk callouts:** L-risk mechanically, but this is where AR3 lives: the T7 Qdrant-down run compares error strings byte-for-byte against v1. The `git diff` inspection here is the early gate

### Scope Boundaries

- Do NOT change any route, model, status code, or error string (AR3)
- Do NOT add streaming upload handling or background tasks — upload stays synchronous-to-terminal (N2)
- Do NOT gate bootstrap on Qdrant being up (app must boot with Qdrant down — warn-and-continue stays)
- Do NOT touch `tests/test_isolation.py` (must pass unchanged in T7)

### Files Expected

**New files:** none.

**Modified files:**
- `src/app/api/documents.py` (async handlers; `bulk_find_all`; upload awaits pipeline)
- `src/app/api/search.py` (handler → async, awaits service)
- `src/app/api/approvals.py` (`list_approvals` → async; loop → `bulk_find_all`)
- `src/app/main.py` (async gathered bootstrap; lifespan shutdown close)

**Must NOT modify:**
- `tests/test_isolation.py`, `docker-compose.yml` (touched-not-changed)
- `src/app/graphs/*` (T6 — the sweep's last code task)

---

## Task T6: Graphs — drop to_thread bridges, route runtime through the OVM

> **Status:** not started
> **Verification:** checklist
> **Effort:** s
> **Priority:** high
> **Depends on:** T4
> **Satisfies REQs:** AR1, AR5
> **Footprint slice:** Modified: `graphs/knowledge/nodes.py`, `graphs/approval/graph.py`, `graphs/actions/nodes.py`, `graphs/runtime.py`
> **High-risk areas touched:** Graphs / runtime (L) — mechanical, plus one routing fix

### Description

The four `asyncio.to_thread` bridges dissolve: the Self-RAG retrieve node
awaits `retrieve_passages` directly, and the three `bulk_insert` call sites
(Self-RAG-adjacent approval graph ×2, action results ×1) await the OVM
directly. `runtime.py`'s `_fetch_approval` drops its direct
`get_connection().client.retrieve` bypass — the one existing OVM-chokepoint
violation — and awaits the new `ApprovalRecord.retrieve_by_ids` (AD7), keeping
the no-tenant-oracle behavior (unknown id and wrong tenant are
indistinguishable `None`).

### Verification Checklist

- **Compile + import:** `uv run python -m compileall -q src/app && uv run python -c "import app.main"` — expected: exit 0
- **Bridges gone:** `grep -rn "to_thread" src/app/graphs/` — expected: **zero hits** (all five current sites are in the four footprint files; `live_data/` has none today and must stay zero-diff)
- **Bypass gone:** `grep -n "get_connection\|infrastructure.qdrant" src/app/graphs/runtime.py` — expected: zero hits — graphs reach Qdrant only via the OVM (Module Boundaries)
- **OVM-routed fetch:** `grep -n "retrieve_by_ids" src/app/graphs/runtime.py` — expected: `_fetch_approval` awaits it; wrong-tenant/unknown still collapse to `None` before any resume attempt
- **Direct awaits:** `grep -n "bulk_insert" src/app/graphs/approval/graph.py src/app/graphs/actions/nodes.py` — expected: plain `await ...bulk_insert(...)` (no thread hop); `grep -n "retrieve_passages" src/app/graphs/knowledge/nodes.py` — expected: awaited directly
- **Logic untouched:** `git diff src/app/graphs/` — expected: only await/to_thread/bypass mechanics; no state keys, edges, node bodies, or interrupt/checkpoint logic changes

### Implementation Notes

- **Module(s):** `app.graphs.*` — allowed dependency is the domain OVM, not `infrastructure.qdrant` (the boundary this task restores)
- **Pattern reference:** the call sites' current shapes — mechanical await insertion at the same lines
- **Key decisions:** AD7 (retrieve_by_ids + violation removed), AD2 (callers migrate, no shims)
- **Libraries:** none
- **High-risk callouts:** L-risk. The one semantic trap: `retrieve_by_ids` must preserve the retrieve-then-compare *behavior* (tenant check reads as empty), not reintroduce an oracle

### Scope Boundaries

- Do NOT alter graph topology, state schemas, interrupt flow, or checkpoint usage
- Do NOT touch `graphs/live_data/` or `graphs/knowledge/graders.py` (touched-not-changed, zero-diff)
- Do NOT add new nodes, retries, or error handling around the awaited calls

### Files Expected

**New files:** none.

**Modified files:**
- `src/app/graphs/knowledge/nodes.py` (retrieve node awaits service)
- `src/app/graphs/approval/graph.py` (2× direct `await bulk_insert`)
- `src/app/graphs/actions/nodes.py` (1× direct `await bulk_insert`)
- `src/app/graphs/runtime.py` (`_fetch_approval` → `await retrieve_by_ids`; bypass + `to_thread` removed)

**Must NOT modify:**
- `src/app/graphs/live_data/`, `src/app/graphs/knowledge/graders.py`, `src/app/infrastructure/checkpoint.py` (touched-not-changed — already async, no vector calls)
- Anything outside `src/app/graphs/` — this task closes the sweep

---

## Task T7: End-to-end run-once verification — the full matrix

> **Status:** not started
> **Verification:** checklist
> **Effort:** m
> **Priority:** critical
> **Depends on:** T5, T6
> **Satisfies REQs:** AR3, AR6, AR8, AR9 (all verified live)
> **Footprint slice:** None — no production files. Owns every runtime evidence item displaced from T1–T6 plus the ARCH Risk & Stress-Test matrix
> **High-risk areas touched:** all — this is the gate for the H/M areas above

### Description

The sweep's single verification gate: boots the real stack (docker Qdrant +
uvicorn + real `OPENAI_API_KEY`) and drives every functional, failure, and
ordering scenario the ARCH promises, including the mechanical
un-awaited-coroutine grep over the completed sweep and the zero-diff guards
for touched-but-not-changed files. Run-once: heredocs and curl only, nothing
committed (AR8). Use throwaway probe tenants (e.g. `probe7a`, `probe7b`) so
dev data never mixes with evidence.

### Verification Checklist

**7A — service-level probes (heredoc `asyncio.run`, before HTTP):**

- **OVM roundtrip + batching:** insert 150 `DocumentRecord`s (> `UPSERT_BATCH_SIZE` 100, forcing gathered batches) → `bulk_insert` returns True; `bulk_find_all` returns all 150; `retrieve_by_ids` same tenant → 150, wrong tenant → `[]` _(verifies AR1/AR5 displaced from T2)_
- **Race-safe create:** `gather` two concurrent `create_collection_if_missing` on one fresh name → both succeed; repeat after → no-op; with Qdrant stopped → the call **raises** (already-exists is the only swallowed outcome) _(AD10, displaced from T1)_
- **Embedding order under gather:** run with `EMBEDDING_BATCH_SIZE=2`, embed 5 texts → 5 vectors × 1536 dims; vectors[0] equals a separate single-text embed of the same text (positional order) _(AD5, displaced from T3)_
- **Clean shutdown:** heredoc opens the connection, calls the close entrypoint, exits → no "unclosed client"/httpx warning in output _(AD11, displaced from T1)_

**7B — HTTP matrix (uvicorn running, fresh tenant per group):**

- **Boot:** startup log shows every collection created (+ `tenant_id` indexes); `GET /tenants/{t}/documents` → `200 []` _(displaced from T5)_
- **Happy paths:** upload small `.txt`, `.pdf`, `.docx` → each `200` `status=ready`, `num_chunks>0`; `GET` document → terminal record fields; `POST /search` → hits with score-sorted-desc, `len ≤ top_k` _(AR3/AR9)_
- **Replace-by-name (A7/A8 under parallelism):** re-upload same filename → old id absent from list, new record ready; search reflects only new content _(AR9)_
- **Delete:** `DELETE` → `204`; then `GET` → `404`, and search no longer returns its chunks (orphan-join works) _(AR9)_
- **Duplicate flag:** upload identical bytes under a new filename → `flags.duplicate_of` equals the original id
- **Failed upload:** corrupt `.pdf` content → `200` `status=failed` with reason (never a bare 500)
- **Approval surface:** `GET /approvals` → `200`; `POST /approvals/{unknown}/decision` → `404` `"approval request not found"` — this drives `_fetch_approval` through the OVM-routed path; additionally heredoc-probe `_fetch_approval` right tenant → record, wrong tenant → `None` _(AR5/AD7, displaced from T6)_
- **Loop not blocked (the latent-bug fix):** start an upload of an OCR-heavy/scanned PDF, then `GET /tenants/{t}/documents` mid-flight → responds while the upload is still processing _(AR6/AD8)_

**7C — failure, ops, and guards:**

- **Un-awaited-coroutine gate:** `grep -rnE "\.(bulk_insert|bulk_find|bulk_find_all|search|delete_by_filter|delete_by_ids|retrieve_by_ids)\(|retrieve_passages\(|EmbeddingDispatcher\.dispatch\(" src/app --include="*.py" | grep -v await | grep -v "async def"` → every remaining hit is a line-continuation whose `await` ends the previous line; and the whole 7B run's server logs contain **zero** `RuntimeWarning: coroutine ... was never awaited` _(AD13's mechanical gate over the finished sweep)_
- **Qdrant down, strings frozen:** stop the container → `POST /search` → `500` with the exact v1 strings (`"vector db unavailable: search failed"` / `"vector db unavailable: could not list documents"` by path); upload → `500` `"vector db unavailable: could not create the document record"`; restart Qdrant → next request succeeds, no app restart needed _(AR3, ARCH forward-stress)_
- **Embedding failure staged:** run a second uvicorn with an invalid `OPENAI_API_KEY` and `EMBEDDING_BATCH_SIZE=1` → upload → `200` `status=failed`, reason starts `embedding service unavailable:` _(ARCH forward-stress: batch failure → failed document)_
- **Cold boot race:** `docker compose down -v` → start app → all collections + keyword indexes exist (gathered, race-safe bootstrap worked from empty) _(AD3/AD10)_
- **Server shutdown:** Ctrl-C the 7B uvicorn → logs show lifespan shutdown, no unclosed-client warning _(AD11)_
- **Optional sanity:** `uv run pytest tests/test_isolation.py` → passes, file untouched (ARCH Out of Scope allows this optional run)
- **Zero-diff guards:** `git diff --stat` shows **no** changes under `src/app/graphs/live_data/`, `src/app/graphs/knowledge/graders.py`, `src/app/application/preprocessing/` cleaning/chunking handlers, `src/app/infrastructure/checkpoint.py`, `docker-compose.yml`, `tests/`; `checkpoints.sqlite` untracked/unmodified _(guards every touched-but-not-changed row)_

### Implementation Notes

- **Module(s):** none — no code. This task is executed, not implemented
- **Pattern reference:** ARCH "Risk & Stress-Test Scenarios" tables — every row appears above as an item
- **Key decisions:** AD13 (this checklist *is* the verification), AD12/AR9 (ordering items), AR3 (frozen-string comparisons)
- **Libraries:** docker compose + curl only
- **High-risk callouts:** this gate carries all runtime evidence for the H (data layer) and M (embedding, pipeline) areas that T1–T6 could only check mechanically — if any item fails, fix in the owning task and re-run the affected group

### Scope Boundaries

- Do NOT write any committed test, fixture, or script — heredocs and shell history only (AR8)
- Do NOT "fix" observed races with new locking beyond what T1/T2 shipped (per-(tenant,filename) serialization stays out of scope)
- Do NOT tune `EMBEDDING_MAX_CONCURRENCY` away from 4 to make items pass — it is operator-tunable by design; evidence runs use defaults (except the staged `EMBEDDING_BATCH_SIZE=1` failure item)

### Files Expected

**New files:** none — verification only; any file that appears in `git status` after T7 is a defect.

**Modified files:** none.

**Must NOT modify:**
- Everything — T7 produces evidence, not diffs
