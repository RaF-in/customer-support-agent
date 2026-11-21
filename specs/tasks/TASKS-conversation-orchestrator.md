# Tasks

> **Date:** 2026-11-21
> **Source:** `specs/architecture/ARCH-conversation-orchestrator.md` (read it first — this file references its decisions as A1–A18). Requirements: `specs/requirements/REQ-conversation-router.md` (slice 1, cited CR-Rn / CR-Nn) + `specs/requirements/REQ-planner-executor.md` (slice 2, cited PE-Rn / PE-Nn). Todo-tool reference semantics: `specs/context.txt`.
> **Developer decisions made during task generation (2026-11-21):**
> (1) **All ten tasks are `checklist` verification mode** (A16 / project convention): run-once scratch commands and scripts, deleted after their runs, never committed; throwaway tenants/tables purged; `tests/test_isolation.py` stays the lone committed test.
> (2) **Fakes-only LLM verification** — every LLM-dependent item runs against langchain-core fake chat models injected at the `infrastructure/llm` factory seam (`set_chat_model`); there are **no real-API-key items**. T1 adding `"router"` / `"planner"` to `CHAT_MODEL_PURPOSES` is what makes the seam accept them.
> (3) **Scratch toy workflows** registered in verification scripts (never committed) give deterministic `WorkflowResult`s for the workflow path and plan steps; the catalog assertions still run against the real registry contents.
> (4) **Postgres = the promoted default compose service** for every runtime checklist; T2 and T6 each include a degrade-gracefully run with Postgres stopped.
> (5) **Declared shared-file slices** (runtime T3/T7 precedent): `orchestrator/__init__.py` (T3 minimal wiring → T4 export), `graph.py` (T4 slice-1 topology → T9 slice-2 wiring), `executor.py` (T8 tools half → T9 node half), `application/conversations.py` (T5 slice-1 service + slice-1 approval branch → T10 resume slice + slice-2 branch).
> (6) **T-numbers are scoped to this document** (the ingestion, workflows, and async-data-layer docs' T1–T8 are separate history).
> (7) Multi-process items (T5 cross-process serialization, T10 fresh-process resume) use two `uv run` processes against the same compose Postgres.

---

## Task T1: Conversational substrate — enums, settings, LLM purposes, compose promotion

> **Status:** done
> **Verification:** checklist
> **Effort:** s
> **Priority:** critical
> **Depends on:** None
> **Satisfies REQs:** enables CR-R5/PE-R1 (router+planner purposes), CR open question (message cap), PE-R13 (caps as settings, A13)
> **Footprint slice:** Modified: `src/app/domain/enums.py` (5 StrEnums), `src/app/infrastructure/settings.py` (+7 caps), `src/app/infrastructure/llm/factory.py` (+2 purposes), `docker-compose.yml` (postgres profile promotion), `.env.example`, `README.md`
> **High-risk areas touched:** Dev/deploy infra (M) — Postgres becomes a default boot dependency

### Description

Everything conversational that is neither Postgres-shaped nor graph-shaped: the five lifecycle enums, the seven tunable caps, the two deterministic LLM purposes, and the one-line compose change promoting the postgres service from its opt-in profile to the default (transcripts are a boot dependency now). After T1 the fake-injection seam accepts `"router"` and `"planner"` — load-bearing for every later task's verification.

### Verification Checklist

- **Enums exact** — `uv run python -c` importing and printing each: `TurnOutcome = direct|clarify|workflow|multi_workflow`, `MessageRole = user|assistant`, `MessageKind = text|workflow_outcome|plan`, `PlanStatus = validated|executing|paused|done|failed`, `TodoStatus = pending|in_progress|completed|abandoned` — expected: exactly these value sets, StrEnum style matching `ApprovalStatus`
- **Settings defaults** — expected: `CONVERSATION_MAX_MESSAGE_CHARS=8000`, `CONVERSATION_HISTORY_MAX_MESSAGES=0` (0 = unlimited), `CLARIFY_MAX_CONSECUTIVE=2`, `PLANNER_MAX_ATTEMPTS=3`, `EXECUTOR_MAX_REINVOCATIONS=3`, `PLAN_MAX_STEPS=8`, `POSTGRES_POOL_MAX_SIZE=10`; env overrides change the printed values (A13 — nothing hardcoded)
- **Factory purposes** — `get_chat_model("router")` / `get_chat_model("planner")` — expected: both map to temperature `0.0` rows in `CHAT_MODEL_PURPOSES`; an unknown purpose still raises ValueError listing the known ones
- **Seam accepts the new purposes** — `set_chat_model("router", fake)` then `get_chat_model("router")` returns the fake (same for planner) — expected: no ValueError (the seam validates against the purpose vocabulary)
- **Zero new dependencies** — `uv run python -c "from psycopg_pool import AsyncConnectionPool"` — expected: exit 0 (already resolved via `psycopg[binary]` / the checkpointer-postgres path; A15); `uv lock` unchanged
- **Compose promotion** — `docker compose config --services` — expected: `qdrant` **and** `postgres` listed with no `--profile` flag; the service comment no longer describes postgres as opt-in
- **Docs** — grep the seven new keys in `.env.example` + `README.md` — expected: all present in the existing tunables style; README notes Postgres is now a default boot dependency for the conversational subsystem
- **No regressions** — `uv run pytest` green; `docker compose up -d` boots both services

### Implementation Notes

- **Module(s):** `domain/enums`, `infrastructure/settings`, `infrastructure/llm/factory`, compose — per ARCH Module Boundaries.
- **Pattern reference:** enums follow `ApprovalStatus` (docstring + values); settings join the existing `Field(default=…, ge=…)` tunables block (`SELF_RAG_TOP_K` style); the factory change is exactly two rows in `CHAT_MODEL_PURPOSES`.
- **Key decisions:** A13 (caps as settings), A5/A10 (temp-0 `router`/`planner` purposes), A15 (zero new deps — verified, not assumed), A18 (compose promotion).
- **High-risk callouts:** Dev/deploy infra (M) — the promotion makes Postgres required for conversations but changes nothing for the existing surfaces; `CHECKPOINT_BACKEND` (sqlite default) stays independent (ARCH migrations note). T2's degrade item and T6's boot-without-Postgres item are the guards.

### Scope Boundaries

- Do NOT touch `domain/conversations.py`, `domain/plans.py`, `infrastructure/postgres.py`, or `main.py` (T2).
- Do NOT change `CHECKPOINT_BACKEND` semantics or `checkpoint.py`.
- Do NOT add any dependency row to `pyproject.toml` (already present).
- Do NOT create test files.

### Files Expected

**New files:** none.

**Modified files:**
- `src/app/domain/enums.py` (+`TurnOutcome`, `MessageRole`, `MessageKind`, `PlanStatus`, `TodoStatus`)
- `src/app/infrastructure/settings.py` (+7 settings, default values exactly as the ARCH footprint row)
- `src/app/infrastructure/llm/factory.py` (`CHAT_MODEL_PURPOSES` += `"router": 0.0`, `"planner": 0.0`)
- `docker-compose.yml` (remove the `profiles: ["postgres"]` line from the postgres service; update its comment)
- `.env.example`, `README.md` (document new settings + the boot-dependency note)

**Must NOT modify:**
- `src/app/domain/base/vector.py`, `src/app/infrastructure/qdrant.py` (Qdrant untouched by this architecture — silent-regression hotspot)
- `src/app/main.py` (T2's slice), `src/app/graphs/**`, `src/app/api/*`, `tests/test_isolation.py`

---

## Task T2: Postgres wiring + conversational domain records

> **Status:** done
> **Verification:** checklist
> **Effort:** m
> **Priority:** critical
> **Depends on:** T1
> **Satisfies REQs:** CR-R2/R3/R4 (storage), CR-R12/N3 (WHERE-clause isolation), CR-N2 substrate (pool + advisory-lock target), PE-R12 (plan persistence), PE-N2
> **Footprint slice:** New: `src/app/infrastructure/postgres.py`, `src/app/domain/conversations.py`, `src/app/domain/plans.py`. Modified: `src/app/main.py` (lifespan: pool open/close + `bootstrap_schema()`)
> **High-risk areas touched:** App bootstrap (L — new lifespan work); Postgres wiring (L — new subsystem)

### Description

The conversational data layer: a process-wide `psycopg_pool.AsyncConnectionPool` at the existing `POSTGRES_DSN`, opened/entered and closed in the app lifespan beside the Qdrant bootstrap (degrade-gracefully — unreachable at boot logs a warning, the rest of the app runs), `bootstrap_schema()` creating the three tables idempotently (`CREATE TABLE IF NOT EXISTS`), and the two domain modules as plain pydantic row models plus small async SQL accessors whose every statement carries `tenant_id` in its WHERE clause by construction. Qdrant and the OVM are untouched — transcripts, linkage, and plans live in Postgres.

### Verification Checklist

- **Pool lifecycle** — boot the app with Postgres up → pool opened at lifespan startup; shutdown → `close_postgres()` closes it (`infrastructure/checkpoint.py` enter/exit pattern); a second close is harmless — expected: clean open/log/close lines, no leaked connections (`pg_stat_activity` empty of app rows after shutdown)
- **Schema bootstrap** — after boot the three tables exist (`conversations`, `conversation_messages`, `conversation_plans`) with `UNIQUE (conversation_id, seq)` and the partial index `(tenant_id, pending_request_id) WHERE pending_request_id IS NOT NULL`; a second boot — expected: idempotent, no error, no duplicated objects
- **Degrade-gracefully** — boot with Postgres stopped — expected: app starts, one warning logged, documents/search/approvals endpoints behave normally _(ARCH backward-regression row: lifespan pool + schema bootstrap)_
- **Header no-oracle** — insert a header for tenant A; fetch `WHERE id AND tenant_id` as tenant B, and as tenant A with a random uuid — expected: both return no row, indistinguishable _(CR-R12/N3)_
- **Turn-write atomicity + seq uniqueness** — drive the append-batch accessor with an induced failure between the `UPDATE conversations SET last_seq = last_seq + 1 … RETURNING` and the message inserts — expected: full rollback, `last_seq` unchanged, zero messages visible; a good batch commits bump + inserts together; a direct insert duplicating an existing `(conversation_id, seq)` → constraint error _(ARCH: one ACID transaction per turn; no gaps, no collisions)_
- **Transcript read** — seed messages across two conversations × two tenants — expected: the read accessor returns all-and-only the tenant+conversation's rows, `ORDER BY seq` chronological _(CR-R4)_
- **Pending-request lookup** — seed an assistant message with `pending_request_id` — expected: the partial-index lookup finds it; the same lookup under another tenant returns empty _(CR-R10 linkage)_
- **Plan accessors** — insert a `PlanRecord` (`validated`), UPDATE → `paused` + `pending_request_id`, re-fetch — expected: mutation persisted (the mutable working state vs. the immutable transcript snapshot); cross-tenant plan get → no row _(PE pause/resume substrate)_
- **Coexistence + untouched Qdrant** — with `CHECKPOINT_BACKEND=postgres`, checkpointer tables coexist untouched (sqlite default independent); `curl http://localhost:6333/collections` — expected: the collection list is unchanged (zero new collections)
- **No regressions** — `uv run pytest` green

### Implementation Notes

- **Module(s):** `infrastructure/postgres.py` (pool + DDL only) and the two domain modules (row models + accessors) — the structural-tenant-isolation point lives in the accessors' WHERE clauses (ARCH Module Boundaries).
- **Pattern reference:** pool lifecycle mirrors `infrastructure/checkpoint.py` (process-wide entered instance); the degrade-gracefully guard mirrors `main.py`'s Qdrant bootstrap try/except; record shape mirrors `domain/approvals.py` minus the OVM base; accessor style mirrors the OVM's public-swallow / private-raise twins, expressed in SQL.
- **Key decisions:** A3 (Postgres tables, one-tx turn writes, partial index), A17 (advisory-lock substrate — the pool must be able to hold one connection per active turn), A18 (lifespan bootstrap, no migration framework).
- **Libraries:** `psycopg_pool.AsyncConnectionPool` + psycopg async connections; JSONB columns for `data` / `steps` / `todos`; `timestamptz` defaults.
- **High-risk callouts:** turn writes need an explicit transaction block (bump + inserts commit together or not at all); accessors must not assume pool exclusivity — T5's advisory lock holds a connection for the turn's duration, and `POSTGRES_POOL_MAX_SIZE` is the concurrent-turns lever (document in README).

### Scope Boundaries

- Do NOT implement the advisory lock, turn orchestration, or anything graph-shaped (T5).
- Do NOT introduce a migration framework (A18 — `CREATE TABLE IF NOT EXISTS` only).
- Do NOT touch `base/vector.py`, `qdrant.py`, `checkpoint.py`, or any Qdrant collection.
- Do NOT create test files.

### Files Expected

**New files:**
- `src/app/infrastructure/postgres.py` (pool lifecycle + `bootstrap_schema()` + `close_postgres()`)
- `src/app/domain/conversations.py` (`ConversationRecord`, `ConversationMessageRecord` — row models + SQL accessors)
- `src/app/domain/plans.py` (`PlanRecord`, `PlanStep`, `Todo` — row models + SQL accessors)

**Modified files:**
- `src/app/main.py` (lifespan: open pool + `bootstrap_schema()` beside the Qdrant bootstrap; close on shutdown)

**Must NOT modify:**
- `src/app/domain/base/vector.py`, `src/app/infrastructure/qdrant.py` (Qdrant untouched — silent-regression hotspot)
- `src/app/infrastructure/checkpoint.py` (checkpointer independent), `src/app/graphs/**`, `src/app/api/*`, `tests/test_isolation.py`

---

## Task T3: Orchestrator foundations + router

> **Status:** done
> **Verification:** checklist
> **Effort:** m
> **Priority:** high
> **Depends on:** T1
> **Satisfies REQs:** CR-R5 (classification), CR-R7 (routing rules embedded in the catalog), CR-R13 (robustness ladder), PE-R1 (fourth outcome vocabulary)
> **Footprint slice:** New: `src/app/graphs/orchestrator/__init__.py` (minimal package wiring — deliberately no `register()`), `…/orchestrator/state.py`, `…/orchestrator/catalog.py`, `…/orchestrator/router.py`
> **High-risk areas touched:** None (M/H)

### Description

The classification machinery: the orchestrator package (deliberately NOT registered — the router menu is registry contents by construction, so it can never route to itself), the state contracts (`OrchestratorState` extending `WorkflowState`, the `RouterDecision` structured-output target, `TurnResult`), the deterministic catalog built from `WORKFLOW_REGISTRY` + `ACTION_REGISTRY` with the routing rules embedded, and the route node — structured-output classification with semantic validation, retry ×1, and the direct fallback.

### Verification Checklist

_Fake chat models injected via the T1 seam; direct node/function invocation from a scratch script (deleted after the run):_

- **Catalog contents** — build the menu after `app.graphs` imports — expected: exactly `self_rag`, `live_data`, `action_execution` with their descriptions and input JSON schemas (ACTION_REGISTRY feeding `action_execution` inputs); `approval` absent (facade-only, never registered); the embedded routing rules present in the text (knowledge questions → `self_rag`; "direct" reserved for small talk/meta; compound requests must not be force-fit into a single workflow) _(CR-R7, PE router rule — inspection)_
- **No self-routing** — `"orchestrator" not in WORKFLOW_REGISTRY` after all imports — expected: True, by construction (A1)
- **Classification contract** — fake router model returning each outcome — expected: parsed `RouterDecision` with `outcome ∈ direct|clarify|workflow|multi_workflow`; `workflow` + `workflow_input` round-trip; `clarifying_question` carried _(CR-R5)_
- **Semantic validation** — out-of-contract classifications (unknown workflow name; `workflow_input` violating the target `input_schema`; `outcome=workflow` with no workflow named) — expected: treated as invalid → the single retry fires with the error fed back _(CR-R13)_
- **Twice-garbage fallback** — a fake that raises/emits garbage on both attempts — expected: classification resolves to the direct fallback; the attempt counter never exceeds 1 retry; no exception escapes the node _(CR-R13, CR-N1)_
- **Second-attempt recovery** — garbage then a valid decision — expected: the valid decision wins; exactly 2 model calls
- **Purpose + seam discipline** — code inspection — expected: the router resolves `get_chat_model("router")` lazily per call (graders.py style) and wraps it in `with_structured_output`; no direct provider imports anywhere in the package

### Implementation Notes

- **Module(s):** `graphs/orchestrator/` — orchestration only; no Qdrant, no Postgres, no `api/` (ARCH Module Boundaries).
- **Pattern reference:** `knowledge/graders.py` (structured-output style + lazy per-call model resolution, which is what makes the seam control every decision); `knowledge/state.py` (TypedDict `total=False` envelope extension + pydantic input contracts).
- **Key decisions:** A1 (StateGraph, not registered), A5 (router purpose, structured output, semantic validation = registry membership + input schema, retry ×1 → direct fallback).
- `must_commit` is a state key the route node reads but does not act on — the clarify→direct override is the conditional edge's deterministic guard (T4). Do not implement it here.
- `OrchestratorState` declares the slice-2 resume keys (`plan`) now; T9/T10 use them.

### Scope Boundaries

- Do NOT build reply paths, composer, or graph topology (T4).
- Do NOT register the orchestrator or touch `graphs/__init__.py`.
- Do NOT call `run_workflow` from the router (the paths node's job).
- Do NOT create test files.

### Files Expected

**New files:**
- `src/app/graphs/orchestrator/__init__.py` (package wiring only — the `build_orchestrator_graph` export is T4's slice)
- `src/app/graphs/orchestrator/state.py` (`OrchestratorState`, `RouterDecision`, `TurnResult`)
- `src/app/graphs/orchestrator/catalog.py` (menu builder over both registries)
- `src/app/graphs/orchestrator/router.py` (classification node)

**Modified files:** none.

**Must NOT modify:**
- `src/app/graphs/__init__.py` (orchestrator must not be imported there — hotspot), `src/app/graphs/runtime.py`, `src/app/graphs/registry.py`, T1/T2 files, `tests/test_isolation.py`

---

## Task T4: Slice-1 reply paths + composition + topology

> **Status:** done
> **Verification:** checklist
> **Effort:** m
> **Priority:** critical
> **Depends on:** T3
> **Satisfies REQs:** CR-R6 (direct), CR-R8 (clarify), CR-R9 (composed reply), CR-R11 (failure apology), CR-R10 reply half (pending template), CR edge case (must_commit), CR-N1 (termination)
> **Footprint slice:** New: `src/app/graphs/orchestrator/paths.py`, `…/composer.py`, `…/graph.py`. Modified: `…/orchestrator/__init__.py` (export `build_orchestrator_graph`)
> **High-risk areas touched:** None (M/H)

### Description

The three slice-1 reply paths and the graph that wires them: direct (a `generator` reply from history), clarify (the classification's question becomes the reply verbatim — no extra LLM call), and single-workflow (a facade `run_workflow` + conversational composition, never the raw payload). `compose` handles the success / failure-apology / pending-approval-template / static-fallback branches; the topology adds the deterministic `must_commit` override on the conditional edge, and a `multi_workflow` classification degrades safely to the direct fallback until slice 2 (ARCH rollout note).

### Verification Checklist

_The compiled graph driven end-to-end with fakes via the seam; a scratch toy workflow registered in the script gives deterministic `WorkflowResult`s (script deleted after the run):_

- **Topology** — the compiled graph's nodes/edges match the slice-1 diagram (`route → {direct_reply | clarify_reply | run_workflow} → compose → END`) — expected: the exact node set; `multi_workflow` routes into the direct-fallback path _(ARCH rollout note)_
- **Direct** — a greeting classified direct — expected: a `generator` reply; exactly **2 LLM calls** (router + generator); toy-workflow run-counter 0 _(CR-R6)_
- **Clarify** — an ambiguous request — expected: the classification's `clarifying_question` is the reply verbatim; exactly **1 LLM call**; no workflow run _(CR-R8, A6)_
- **must_commit override** — `must_commit=True` in state + a fake router that classifies clarify — expected: the conditional edge deterministically overrides to the direct path; no LLM call beyond router+generator _(CR edge case, A7)_
- **Workflow** — a valid classification — expected: `run_workflow` called through the facade with (name, envelope tenant_id, schema-valid input); the composed reply reflects the toy output; the raw payload is not shown verbatim _(CR-R9)_
- **Failure apology** — the toy workflow returns `failed` — expected: an apologetic reply naming the workflow; conversational shape (the conversation can continue) _(CR-R11)_
- **Pending template** — the toy workflow returns `pending_approval` + `request_id` — expected: a deterministic template naming the request id; **zero generator calls**; the request_id carried in state for persistence _(CR-R10 reply half, A6)_
- **Degradation** — the generator fake raises — expected: a static fallback reply; the turn terminates; no exception escapes the graph _(CR-N1)_
- **Router-fallback integration** — router fake garbage twice — expected: the graph still ends at compose with a direct-fallback reply

### Implementation Notes

- **Module(s):** `graphs/orchestrator/` paths/composer/graph — orchestration only.
- **Pattern reference:** `live_data/graph.py` (StateGraph wiring discipline); `knowledge/nodes.py` (node-function style); the composer's ultimate fallbacks are static strings.
- **Key decisions:** A6 (clarify rides the classification; the pending reply is a deterministic template), A7 (`must_commit` override lives in the conditional edge — a deterministic guard, not prompt discipline).
- Workflow runs go through the facade only — never sibling graph internals (Module Boundaries).
- The toy workflow exists only in the verification script — never committed (house pattern).

### Scope Boundaries

- Do NOT implement planner/executor/verify/compose_final (T7/T9).
- Do NOT persist anything — the graph never touches Postgres/Qdrant (the service is T5).
- Do NOT create test files.

### Files Expected

**New files:**
- `src/app/graphs/orchestrator/paths.py` (direct / clarify / single-workflow nodes)
- `src/app/graphs/orchestrator/composer.py` (result / failure / pause-template / fallback composition)
- `src/app/graphs/orchestrator/graph.py` (`build_orchestrator_graph()` — slice-1 topology)

**Modified files:**
- `src/app/graphs/orchestrator/__init__.py` (export `build_orchestrator_graph` — the T4 slice of it)

**Must NOT modify:**
- `src/app/graphs/registry.py`, `src/app/graphs/__init__.py`, `src/app/graphs/runtime.py`, T1–T3 files, `tests/test_isolation.py`

---

## Task T5: Conversation service (slice 1)

> **Status:** done
> **Verification:** checklist
> **Effort:** l
> **Priority:** critical
> **Depends on:** T2, T4
> **Satisfies REQs:** CR-R1/R2 (turn + identity), CR-R3 (history), CR-R4 (transcript read), CR-R10 service half (outcome append), CR-N1 (termination), CR-N2 (serialization at any worker count), CR-N4 (logs), CR-R12/N3 (service-level no-oracle)
> **Footprint slice:** New: `src/app/application/conversations.py`
> **High-risk areas touched:** New conversational surfaces (M) — customer content in Postgres behind a new service

### Description

The heart of slice 1: `start_conversation` / `send_message` / `get_transcript` / `on_approval_decided` over the T2 accessors and the T4 graph. Turns are serialized by a turn-spanning `pg_advisory_lock(hashtext(conversation_id))` on a held pool connection — correct at any worker count, released at turn end or session death; the turn write is one ACID transaction (seq bump + message inserts commit together); history assembly feeds the graph (full transcript by default, lever-capped); the consecutive-clarify streak computes `must_commit`; `on_approval_decided` (slice-1 branch) appends the completed outcome via the `pending_request_id` linkage, its failures logged and swallowed.

### Verification Checklist

_Scratch scripts against the promoted compose Postgres; fakes via the seam; the scratch toy workflow where a run is needed; two `uv run` processes for the concurrency items:_

- **First turn + transcript** — `start_conversation` — expected: header row created; `[user msg, assistant reply, outcome msg]` committed in one transaction; `TurnResult` carries `conversation_id`/`message_id`/`reply`/`outcome`/`runs`; `get_transcript` returns the rows chronological by seq _(CR-R1/R2/R4)_
- **History-aware follow-up** — a fake router that echoes what it saw — expected: turn 2's history contains turn 1's reply and outcome (the "now compare it to Y" acceptance) _(CR-R3)_
- **Unknown vs cross-tenant** — `send_message`/`get_transcript` with an unknown cid, and with tenant B on tenant A's cid — expected: identical `ConversationNotFoundError` both times, no oracle _(CR-R12/N3)_
- **Cross-process serialization** — two worker processes sending to one conversation concurrently — expected: the advisory lock serializes the turns across the process boundary; the second turn's router sees the first's persisted outcome; seqs strictly increasing, no interleaved transcript _(CR-N2, A17)_
- **Worker death** — `kill -9` one process mid-turn — expected: session death releases the advisory lock; the next turn on that conversation proceeds — never stranded _(ARCH forward scenario)_
- **Clarify streak** — two consecutive clarify turns — expected: the third turn is seeded `must_commit=True` (cap read from `CLARIFY_MAX_CONSECUTIVE`) _(CR edge case)_
- **Approval hand-off (slice 1)** — a pending-approval turn — expected: the assistant reply row carries `pending_request_id`; `on_approval_decided` appends the outcome message via the partial-index lookup, positioned after any newer user messages (append-order); an append failure is logged + swallowed, never raising into the decision path _(CR-R10 service half, A8)_
- **Atomic persist** — Postgres stopped at persist (after a side-effectful toy run) — expected: a single user-meaningful error from the service; nothing partial visible; `last_seq` unchanged; the next successful turn has no seq gap _(CR-N1; ARCH one-tx note — the gap caveat is deleted by design)_
- **History lever + log discipline** — `CONVERSATION_HISTORY_MAX_MESSAGES` env override truncates the assembled history; the turn log line (`turn tenant=… conversation=… turn_id=… outcome=… workflow=… status=… duration_ms=…`) carries ids/statuses only — expected: scanning the logs for a phrase from the user's message finds nothing _(CR-N4)_

### Implementation Notes

- **Module(s):** `application/conversations.py` — deps: domain records, the orchestrator graph builder, `WorkflowResult`, the `infrastructure.postgres` pool handle; never `api/` (Module Boundaries).
- **Pattern reference:** `graphs/runtime.py` lock + logging idioms (`_request_locks` precedent, upgraded to turn-spanning advisory locks); the compile-once graph cache mirrors `_compiled_graphs`.
- **Key decisions:** A17 (advisory lock on a held connection, acquired before the header check, released in `finally`), A3 (one-tx turn writes), A14 (full transcript + settings lever), A8 (`on_approval_decided` log-and-swallow).
- **High-risk callouts:** New conversational surfaces (M) — all SQL goes through T2's accessors (the service writes no SQL of its own; if an accessor is genuinely missing, add it to the T2 files as a declared additive change and note it); the service is the single catch point for graph crashes — any exception maps to the one user-meaningful error (CR-N1).
- `POSTGRES_POOL_MAX_SIZE` is the concurrent-turns lever — the lock holds one connection for the turn's (LLL-duration) turn; document beside the settings.

### Scope Boundaries

- Do NOT implement HTTP (T6) or anything slice-2 (T7–T10).
- Do NOT write SQL in the service — accessors only.
- Do NOT let the graph touch Postgres — persistence is service-side only.
- Do NOT create test files.

### Files Expected

**New files:**
- `src/app/application/conversations.py` (turn service: advisory locks, one-tx persistence, history assembly, `on_approval_decided` slice-1 branch)

**Modified files:** none (a genuinely-missing T2 accessor may be added as a declared exception, noted in the task).

**Must NOT modify:**
- `src/app/api/*`, `src/app/graphs/**` (consumed as-is), `src/app/infrastructure/postgres.py`, `tests/test_isolation.py`

---

## Task T6: Conversations HTTP API + approvals decision hook

> **Status:** done
> **Verification:** checklist
> **Effort:** m
> **Priority:** critical
> **Depends on:** T5
> **Satisfies REQs:** CR-R1/R2/R4 over HTTP, CR-R5 (outcome observability), CR-R10 end-to-end, CR edge cases (422 battery, 404 no-oracle), CR-N3
> **Footprint slice:** New: `src/app/api/conversations.py`. Modified: `src/app/api/approvals.py` (one failure-isolated post-decision call), `src/app/main.py` (router include)
> **High-risk areas touched:** Approvals decision path (M) — frozen HTTP contract gains a side effect; New conversational surfaces (M); Dev/deploy infra (M)

### Description

The three endpoints (POST start → 201, POST message → 200, GET transcript) in the existing tenant-scoped router style — parsing + error mapping only, everything behind the conversation service — plus the approvals decision handler modification: after a successful `decide_approval`, await `conversations.on_approval_decided(...)` inside a failure-isolating guard. `DecisionResponse` bodies, status codes, and detail strings stay byte-identical.

### Verification Checklist

_uvicorn + an httpx scratch script; fakes via the seam; the decision endpoint's contract captured pre-change for the byte-diff:_

- **OpenAPI** — expected: exactly the 3 new paths beside the existing ones; the app boots clean with Postgres up
- **POST start / message** — expected: 201 / 200 with `{conversation_id, message_id, reply, outcome, workflow?}`; `outcome ∈ direct|clarify|workflow|multi_workflow` _(CR-R5 observability)_
- **GET transcript** — expected: chronological by seq; the declared message shape `{seq, role, kind, content, created_at, pending_request_id?, data?}` _(CR-R4)_
- **422 battery** — empty string, whitespace-only, >`CONVERSATION_MAX_MESSAGE_CHARS` message — expected: clear 422 each; the transcript is unchanged (no turn recorded) _(CR edge cases)_
- **404 no-oracle** — an unknown cid, and tenant B on tenant A's cid — expected: one byte-identical 404 detail for both _(CR-R12)_
- **Approvals byte-identical** — before/after byte-diff over the decision endpoint's 200 approve / 200 reject / 404 / 409 / 500 paths — expected: identical statuses, bodies, and detail strings _(guards the M-risk approvals decision path)_
- **Conversational approval end-to-end** — a conversational pending-approval turn → POST decision → 200 — expected: the transcript now shows the completed outcome appended after the waiting reply _(CR-R10 acceptance over HTTP)_; with Postgres stopped post-decision — expected: the decision still returns 200, the append failure is logged, approval + action records remain the durable truth _(A8)_
- **Boot without Postgres** — start the app with Postgres down — expected: the app starts; the three conversation endpoints return 500; documents/search/approvals behave normally _(ARCH backward-regression row, A18)_
- **Runtime infra 500s** — Postgres stopped mid-run — expected: a single user-meaningful 500 per conversation endpoint; no partial turns
- **No regressions** — `uv run pytest` green; the existing endpoints behave unchanged

### Implementation Notes

- **Module(s):** `api/conversations.py` (conversation service only), `api/approvals.py` (+1 guarded call), `main.py` (`include_router`).
- **Pattern reference:** `api/approvals.py` router + error-mapping style; the message body is a strip-aware pydantic model (whitespace-only rejected, `max_length` from settings).
- **Key decisions:** A4 (REST pair, id in path, byte-identical 404 — never silent new conversation), A8 (hook after `decide_approval` succeeds, inside try/except + error log).
- **High-risk callouts:** the approvals byte-diff item is the guard for the M-risk decision path — if it diffs, the hook changed observable behavior and must be re-isolated. Only the 404 detail must be byte-identical between unknown and cross-tenant; 422 prose needs to be clear, not frozen.

### Scope Boundaries

- Do NOT change `DecisionResponse` fields, status codes, or detail strings.
- Do NOT add auth, streaming, or transcript pagination (full-conversation fetch; paginatable later per the ARCH performance note).
- Do NOT create test files.

### Files Expected

**New files:**
- `src/app/api/conversations.py` (POST start · POST message · GET transcript)

**Modified files:**
- `src/app/api/approvals.py` (one post-decision, failure-isolated `on_approval_decided` call)
- `src/app/main.py` (`include_router(conversations.router)`)

**Must NOT modify:**
- `src/app/application/conversations.py` (the service), `src/app/graphs/**`, `src/app/api/search.py` (frozen v1 contract — hotspot), `tests/test_isolation.py`

---

## Task T7: Planner

> **Status:** done
> **Verification:** checklist
> **Effort:** m
> **Priority:** high
> **Depends on:** T3
> **Satisfies REQs:** PE-R2 (plan contract), PE-R3 (validation), PE-R4 (retry/exhaustion), PE-R13 (bounded)
> **Footprint slice:** New: `src/app/graphs/orchestrator/planner.py`
> **High-risk areas touched:** None (M/H)

### Description

Plan generation and validation: the `planner`-purpose LLM emits the JSON plan contract; the validator rejects with five distinguishable error classes; invalid plans are fed back for correction within `PLANNER_MAX_ATTEMPTS`; exhaustion yields an apologetic direct answer with **nothing executed**. Verified at function/node level — the graph cycle wiring lands with T9.

### Verification Checklist

_Fakes via the seam; direct invocation of the generation/validation functions:_

- **Valid plan** — a two-workflow goal — expected: validates; steps numbered sequentially; every workflow known; placeholders reference only earlier steps; the same workflow may appear in two steps _(PE-R2 + edge case)_
- **Five distinguishable errors** — malformed JSON / unknown workflow / placeholder referencing a nonexistent-or-later step / zero steps / >`PLAN_MAX_STEPS` — expected: each invalidity class produces its own named error; a well-formed plan passes all checks _(PE-R3)_
- **Retry with feedback** — attempt 1 invalid, errors fed back, attempt 2 valid — expected: the corrected plan is accepted; the attempt counter reads 2 _(PE-R4)_
- **Exhaustion** — an always-invalid fake — expected: exactly `PLANNER_MAX_ATTEMPTS` attempts; the outcome is the apologetic direct answer; the run-counter is 0 (nothing runs) _(PE-R4)_
- **Bounds from settings** — `PLAN_MAX_STEPS` / `PLANNER_MAX_ATTEMPTS` env overrides — expected: the observed thresholds move with the settings, nothing hardcoded _(PE-R13)_
- **Purpose + prompt inspection** — expected: generation resolves `get_chat_model("planner")` lazily (seam control); the prompt embeds the catalog menu and the `$$STEP_X_OUTPUT$$` contract in the `specs/context.txt` planner style (capabilities block + JSON-list instructions + a worked example)

### Implementation Notes

- **Module(s):** `graphs/orchestrator/planner.py` — orchestration only, no persistence.
- **Pattern reference:** graders.py lazy per-call model resolution; the planner prompt style follows `specs/context.txt` (the reference implementation's capabilities/examples structure).
- **Key decisions:** A10 (planner purpose temp 0; JSON contract; ≤3 attempts → apologetic direct answer, nothing runs).
- Malformed JSON is a **handled validation class** (a distinguishable error), not an exception that escapes — the plan text is parsed by the validator.
- Persistence (the PlanRecord row + the transcript `plan`-kind message) happens at validation success in the service layer — T9/T10 wire it; T7's functions return the validated plan or the exhaustion outcome.

### Scope Boundaries

- Do NOT wire the planner into `graph.py` (T9's slice).
- Do NOT execute plan steps or resolve placeholders (T9).
- Do NOT persist plans (service-side only).
- Do NOT create test files.

### Files Expected

**New files:**
- `src/app/graphs/orchestrator/planner.py` (plan generation + validation with distinguishable errors)

**Modified files:** none.

**Must NOT modify:**
- `src/app/graphs/orchestrator/graph.py` (T9's slice), T1–T6 files, `tests/test_isolation.py`

---

## Task T8: Todo tools

> **Status:** done
> **Verification:** checklist
> **Effort:** m
> **Priority:** high
> **Depends on:** T1
> **Satisfies REQs:** PE-R5 (todo discipline + the vacuous-truth guard), PE-R6 (reference semantics)
> **Footprint slice:** New: `src/app/graphs/orchestrator/executor.py` (tools half: the `TodoList` model + `AddTodo` / `CompleteTodo` / `AbandonTodo` / `ListTodos`)
> **High-risk areas touched:** None (M/H)

### Description

The four executor-bound todo tools and their `TodoList` state, implementing exactly the `specs/context.txt` semantics: statuses pending/in_progress/completed/abandoned; completing a todo requires having listed the plan since creating it (observe-before-claim); abandonment requires a stated reason; a list is complete only when it has no open items **and** is non-empty. Verified by direct tool invocation — no LLM involved.

### Verification Checklist

_Direct tool invocation against a fresh TodoList, no LLM:_

- **Add** — expected: a pending todo with an incrementing id; empty text refused with a clear error
- **Observe-before-claim** — add then complete without `ListTodos` — expected: a corrective refusal naming the rule (create-then-complete busywork guard); list then complete — expected: succeeds _(PE-R6)_
- **Abandon** — expected: requires a reason; the reason is recorded on the todo; abandoning a created-unlisted todo also clears the unlisted flag (reference behavior)
- **Completion rule** — an empty list — expected: never complete (the `all([]) == True` trap is closed); a list with open items — expected: not complete; all closed + non-empty — expected: complete _(PE-R5)_
- **ListTodos** — expected: marks the list observed; reports the rendered list, the open count, and the complete flag
- **Unknown id** — expected: a clear "no todo with id N" error
- **Render marks** — expected: `[ ]` / `[~]` / `[x]` / `[-]` + `id. text`, exactly per the reference

### Implementation Notes

- **Module(s):** `graphs/orchestrator/executor.py` — the tools are langchain tools (bound to the executor model in T9); the `TodoList` rides the graph state.
- **Pattern reference:** `specs/context.txt` is normative — port the **semantics** (including the private created-unlisted guardrail set) into langchain tool form, not the foreign `AgentTool` framework.
- **Key decisions:** A11 (todo tools per the reference semantics — the foundation of system-verified completion).
- The executor node itself is T9's declared slice of this file.

### Scope Boundaries

- Do NOT implement the executor node, step execution, or placeholder resolution (T9).
- Do NOT bind tools to a model here (T9).
- Do NOT create test files.

### Files Expected

**New files:**
- `src/app/graphs/orchestrator/executor.py` (TodoList + the four todo tools)

**Modified files:** none.

**Must NOT modify:**
- T1–T7 files, `tests/test_isolation.py`

---

## Task T9: Executor + verify + slice-2 topology

> **Status:** not started
> **Verification:** checklist
> **Effort:** l
> **Priority:** critical
> **Depends on:** T4, T7, T8
> **Satisfies REQs:** PE-R1 (fourth outcome live), PE-R5 (in-graph discipline), PE-R7 (chaining), PE-R8 (verified completion + takeover), PE-R9 (fail-stop), PE-R10 pause half, PE-R11 (final-answer synthesis), PE-R13 (bounded)
> **Footprint slice:** Modified: `src/app/graphs/orchestrator/executor.py` (node half), `…/composer.py` (+`compose_final`), `…/graph.py` (planner→validate→executor⇄verify→compose_final wiring)
> **High-risk areas touched:** None (M/H)

### Description

The executor working the plan: one todo per step in plan order, sequential step runs through the facade, `$$STEP_X_OUTPUT$$` resolution against earlier outputs before each run, strict fail-stop with transitive dependency skipping (dependents' todos abandoned with reasons, independents still run), and the approval pause — the graph emits the pause state; the service persists the PlanRecord (the graph never touches Postgres). The verify node re-invokes the executor while todos remain open (≤ `EXECUTOR_MAX_REINVOCATIONS`), then takes over with an honest partial report; `compose_final` makes one generator call over all step outputs.

### Verification Checklist

_The graph driven with fakes via the seam; scratch toy workflows returning scripted per-step outcomes (completed / failed / pending_approval):_

- **Topology** — expected: the planner → validate cycle (≤ `PLANNER_MAX_ATTEMPTS`) with the exhaustion branch → apologetic direct; executor ⇄ verify (≤ `EXECUTOR_MAX_REINVOCATIONS`); all-closed/takeover → `compose_final` → END; the resume entry seeds the executor _(ARCH diagram)_
- **Chaining** — a two-step plan, step 2's input containing `$$STEP_1_OUTPUT$$` — expected: step 2's resolved input carries step 1's real output, observable in the recorded run _(PE-R7)_
- **Todo discipline in-graph** — expected: N todos created in plan order before any step runs; an empty-list completion claim is rejected and corrected _(PE-R5)_
- **Fail-stop** — a 3-step plan where step 1 fails, step 2 depends on step 1, step 3 is independent — expected: step 2 skipped with its todo abandoned (reason: dependency failed), step 3 runs and completes, and the final report covers all three outcomes; zero open todos at the end _(PE-R9)_
- **Verified completion** — a stalling executor fake leaves todos open — expected: re-invocation exactly up to the bound, then takeover; the user-visible report lists the open items honestly instead of claiming success _(PE-R8)_
- **Approval pause** — a step returns `pending_approval` + `request_id` — expected: the turn ends with the templated waiting reply naming the request; no later step runs; the pause state (plan + todos + pending_request_id) is emitted for the service to persist as a paused PlanRecord; the resume entry continues from the paused step _(PE-R10 pause half)_
- **compose_final** — two completed steps — expected: exactly one generator call receiving all step outputs; the answer references content from both steps; the last step's output is never returned verbatim alone _(PE-R11)_
- **Termination** — adversarial fakes (always-invalid planner AND stalling executor in one run) — expected: every orchestrated path terminates within the caps; `recursion_limit` is never reached on legitimate paths _(PE-R13)_
- **Module-boundary inspection** — grep the orchestrator package's imports — expected: no Qdrant, no Postgres, no domain-record, no `api/` imports (persistence flows through state + the service)

### Implementation Notes

- **Module(s):** the executor node half of `executor.py`, `compose_final` in `composer.py`, the slice-2 wiring of `graph.py`.
- **Pattern reference:** `live_data/graph.py` bounded-loop discipline (counters checked on conditional edges); the runtime's two-failure-classes discipline — step outcomes are `WorkflowResult` values, never exceptions.
- **Key decisions:** A9 (pause = mutable PlanRecord; resume re-seeds — no checkpointer: its `Command(resume)` no-ops after an interleaved turn, verified on langgraph 1.2.22, and checkpointer-as-store measured ~35× storage at 20 turns), A11 (system-verified completion), A12 (strict fail-stop).
- **Dependency rule:** step B depends on A iff any input of B contains `$$STEP_A_OUTPUT$$` (transitive closure); dependents are cut **before** resolution — a placeholder is never resolved against a failed/skipped step.
- Rejection of a step's approval counts as that step failing (PE decision 8) — exercised end-to-end in T10.
- **Persistence discipline:** the graph never touches Postgres — pause/completion state rides `OrchestratorState`; the service writes the PlanRecord and transcript rows.

### Scope Boundaries

- Do NOT implement plan resume or decision handling in the service (T10).
- Do NOT add parallel branches, in-run replanning, or executor adaptation (ARCH Out of Scope).
- Do NOT create test files.

### Files Expected

**New files:** none.

**Modified files:**
- `src/app/graphs/orchestrator/executor.py` (the executor node — T9's declared slice)
- `src/app/graphs/orchestrator/composer.py` (+ `compose_final`)
- `src/app/graphs/orchestrator/graph.py` (the planner/validate/executor/verify/compose_final topology)

**Must NOT modify:**
- `src/app/graphs/orchestrator/planner.py` (consumed as-is — raise back to planning if its contract falls short), `src/app/graphs/runtime.py`, `src/app/api/*`, `src/app/application/*`, `tests/test_isolation.py`

---

## Task T10: Plan pause/resume + acceptance walkthroughs

> **Status:** not started
> **Verification:** checklist
> **Effort:** m
> **Priority:** critical
> **Depends on:** T6, T9
> **Satisfies REQs:** PE-R10 (full pause/resume incl. rejection), PE-R12 (artifacts + zero-exec follow-ups), PE-N2 (artifact isolation), PE-N4 (logs); the full CR acceptance sweep
> **Footprint slice:** Modified: `src/app/application/conversations.py` (resume slice: `on_approval_decided` slice-2 branch + the resume entry)
> **High-risk areas touched:** New conversational surfaces (M)

### Description

The completion of the orchestrator: `on_approval_decided` finds the paused PlanRecord by `pending_request_id`, re-seeds the graph's resume entry, runs the remainder, and appends the final answer to the transcript (append-order). Then the acceptance gate: scripted walkthroughs of both REQs' acceptance criteria through the HTTP surface with fakes at the seam — the run-once evidence that the whole architecture delivers what the requirements asked for.

### Verification Checklist

_Two `uv run` processes where noted; the promoted compose Postgres; fakes via the seam:_

- **Approve → resume** — compound request → pause reply naming the request → approve via the existing surface — expected: the remainder runs from the paused step; the final answer is appended after any newer messages; the PlanRecord reaches `done` _(PE-R10 approve path)_
- **Reject → step failure** — the same setup, rejected — expected: the paused step counts as failed; dependents skipped and their todos abandoned; the final answer reports the failure and its downstream consequences honestly; the PlanRecord records the truth (no deletion) _(PE-R10 rejection path, PE decision 8)_
- **Liveness while paused** — a new user message while the plan is paused — expected: it routes normally (not queued behind the plan); the resumed final answer is appended after it _(PE edge case)_
- **Deploy-safety** — pause in one process; resume from a **fresh process** reading the PlanRecord from Postgres — expected: resume works; no checkpointer involvement; `checkpoints.sqlite` untouched _(A9 + the hotspot row)_
- **Artifacts + zero-exec follow-up** — after a finished plan — expected: the transcript holds the `plan`-kind message (the validated plan JSON) and the per-step outcomes; a follow-up "summarize what you just did for me" is answered from the stored transcript with the run-counter at 0 _(PE-R12)_
- **Acceptance matrix** — a scripted sweep of CR R1–R13 and PE R1–R13 (each acceptance criterion exercised once, over HTTP where applicable) — expected: every row passes; this matrix run is the phase's acceptance gate
- **Cross-tenant probes** — tenant B against tenant A's conversation id, transcript, and plan artifacts — expected: byte-identical 404s / empty reads; nothing inferable _(CR-R12, PE-N2)_
- **Log discipline** — the turn/plan/step lines during one full compound run — expected: ids and statuses only; scanning the logs for message text, replies, plan JSON, or todo text finds nothing _(CR-N4, PE-N4)_
- **Isolation guard** — `uv run pytest` green

### Implementation Notes

- **Module(s):** the resume slice of `application/conversations.py` — the service owns re-seeding the graph from the persisted PlanRecord (the `entry(resume)` topology T9 delivered).
- **Pattern reference:** `runtime.py` `decide_approval`'s lock → re-check → act discipline is the approvals-side twin; the resume path reuses the T5 turn machinery (advisory lock, one-tx append).
- **Key decisions:** A9 (mutable PlanRecord + immutable transcript snapshot; deploy-safe pauses), A8 (`on_approval_decided` failures logged and swallowed — never change the decision response).
- The acceptance-matrix script is the largest scratch artifact: organize it per REQ row, delete it after the run, purge throwaway tenants/tables.

### Scope Boundaries

- Do NOT add new approval mechanics — expiry, escalation, reminders (ARCH Out of Scope; the open question is inherited deliberately).
- Do NOT add push delivery, retention limits, or summarization.
- Do NOT commit the acceptance-matrix script or any test file.

### Files Expected

**New files:** none.

**Modified files:**
- `src/app/application/conversations.py` (the resume slice: `on_approval_decided` slice-2 branch + resume entry)

**Must NOT modify:**
- `src/app/graphs/orchestrator/*` (the resume entry exists from T9 — raise back to planning if the seeding contract needs a graph change), `src/app/api/*`, `src/app/graphs/runtime.py`, `tests/test_isolation.py`

---

_All ten tasks are `checklist` verification mode (A16 / project convention): run-once scratch commands, deleted after their runs; no test files beyond the pre-existing `tests/test_isolation.py`. LLM behavior is verified exclusively through fake chat models injected at the `set_chat_model` seam (developer decision, carried from 2026-11-17) — real-model runs are deliberately absent from these checklists. Postgres-dependent items run against the promoted default compose service; each of T2/T5/T6/T10 includes its stated degrade/crash probe._
