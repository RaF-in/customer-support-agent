# Tasks

> **Date:** 2026-11-23
> **Source:** specs/architecture/ARCH-long-term-memory.md (standalone brief — no REQ doc; inferred requirements LM-R1..R7, LM-N1..N5 in the ARCH)
> **Developer decisions made during task generation (2026-11-23):**
> (1) **Pure recency in v1** — no importance weight anywhere in the contract; the card is `ORDER BY updated_at DESC` capped by `MEMORY_MAX_FACTS`. Importance arrives with the v2 semantic-retrieval work if the ~100-fact tripwire ever fires (resolves the ARCH Open Question).
> (2) **Extraction never retries** — exactly one `memory`-purpose call per turn; garbage/unparseable output is caught, logged, and swallowed (A8). A missed turn's facts typically recur in a later exchange.
> (3) **Invalid ops are skipped, valid ops still apply** — per-op validation (`add` needs non-blank content; `supersede` needs a `fact_id` naming an active fact of the current card); skipped ops are counted in the extraction log line, never void the batch.
> (4) **Fakes-only LLM verification** — every LLM-dependent checklist item runs against langchain-core fake chat models injected at the `set_chat_model` seam under the new `"memory"` purpose; no real-API-key items (2026-11-17 convention carried forward).
> (5) **Three tasks by subsystem** (substrate → read path → write path); T-numbers are scoped to this document (prior docs' T1–T10 are separate history).
> (6) No test suite (carried convention): all verification is run-once checklist commands via scratch scripts — deleted after their runs, never committed; throwaway tenants/users purged. `tests/test_isolation.py` remains the only committed test and must stay green after every task.

---

## Task T1: Facts Substrate — Schema, Settings, Purpose Key, Domain Accessors

> **Status:** done
> **Verification:** checklist
> **Effort:** m
> **Priority:** critical
> **Depends on:** None
> **Satisfies REQs:** LM-R4 (structural tenant isolation), LM-R6 (`forget_user`), LM-R7 (nullable `user_id` — anonymous = today), LM-N1 (zero new dependencies), LM-N5 (additive schema, one-lever rollback)
> **Footprint slice:** New: src/app/domain/memories.py. Modified: src/app/infrastructure/postgres.py (3 additive DDL statements), src/app/infrastructure/settings.py (2 fields), src/app/infrastructure/llm/factory.py (1 purpose entry), src/app/domain/conversations.py (record field + create param)
> **High-risk areas touched:** Schema bootstrap (L) — additive `IF NOT EXISTS` statements against a populated database; nothing M/H — the pattern is proven three times over (conversations, plans, approvals)

### Description

Everything memory needs to store and nothing that uses it: the `memory_facts` table + card index + nullable `conversations.user_id` column (additive DDL inside `bootstrap_schema()`), the two `MEMORY_*` levers, the `"memory"` purpose key in the chat-model factory, the `MemoryFact` row model, and the four accessor twins (`add_fact`, `list_active_facts`, `supersede_fact`, `forget_user`) in a new `domain/memories.py` mirroring `domain/plans.py`. After T1 the card exists as a data structure; T2 reads it and T3 fills it.

### Verification Checklist

_Direct accessor calls from a scratch async script against `docker compose --profile postgres up`; schema inspected via psql; scratch script deleted after the run:_

- **Bootstrap idempotency** — expected: boot the app twice; `memory_facts` exists with the `(tenant_id, user_id, status, updated_at DESC)` index (psql `\d`); `conversations.user_id` is `TEXT NULL`; conversations created before T1 keep `NULL` (no backfill, by design).
- **Settings levers** — expected: `MEMORY_ENABLED` defaults `True`, `MEMORY_MAX_FACTS` defaults `50`; `MEMORY_MAX_FACTS=0` fails construction with the clear pydantic `ge=1` message; both read from env/`.env` like every lever.
- **Purpose key** — expected: `get_chat_model("memory")` returns a cached model at temperature 0.0 riding `CHAT_MODEL` (no new setting); unknown purposes still fail fast listing known ones; `set_chat_model("memory", fake)` makes subsequent `get_chat_model("memory")` calls return the fake (the T3 verification seam).
- **`add_fact` round-trip** — expected: one call → one row `status='active'` with the exact content, `source_conversation_id` recorded, `created_at`/`updated_at` set; returned row equals a fresh `list_active_facts` read.
- **`list_active_facts`** — expected: rows in `updated_at DESC` order; superseded rows never listed; `limit` respected (seed 3 facts, `limit=2` → the 2 most recent); unknown user → `[]`.
- **Supersede semantics** — expected: `supersede_fact` on an active fact flips it to `superseded` with `superseded_by` set and bumps `updated_at`; the replacing fact (from a prior `add_fact`) stays `active`; superseding an already-superseded fact → `None`, row unchanged (belt-and-braces no-op).
- **Tenant isolation** — expected: tenant A's card never lists tenant B's facts for the same `user_id` string; cross-tenant `supersede_fact`/`forget_user` → `None`/`False` no-ops, byte-identical to unknown ids (no oracle). _(LM-R4)_
- **`forget_user`** — expected: deletes every fact (active and superseded) for `(tenant, user)` → `True`; an immediate second call → `False`; a different user's facts under the same tenant untouched. _(LM-R6)_
- **Twin discipline** — expected: with the pool closed (never opened in the scratch process), every public accessor returns its failure value (`None` / `[]` / `False`) and logs an exception; every private `_`-twin raises (the application service stays the single catch point).
- **No regressions** — expected: `uv run pytest` behaves exactly as before (isolation test); the app boots clean with the grown `_SCHEMA_STATEMENTS`.

### Implementation Notes

- **Pattern reference:** `src/app/domain/plans.py` — pydantic row model over `RETURNING *`, public-swallow/private-raise twins, `get_postgres_pool().connection()` per statement, structural `tenant_id` in every WHERE clause. `MemoryFact` mirrors `PlanRecord` style: `id` (default `uuid4`), `tenant_id`, `user_id`, `content`, `status: Literal["active", "superseded"] = "active"`, `source_conversation_id: Optional[UUID]`, `superseded_by: Optional[UUID]`, `created_at`/`updated_at` (default `_utc_now`).
- **DDL — three additive statements in `_SCHEMA_STATEMENTS` (A18):** (1) `ALTER TABLE conversations ADD COLUMN IF NOT EXISTS user_id TEXT`; (2) `CREATE TABLE IF NOT EXISTS memory_facts` with the fields above (`id UUID PRIMARY KEY DEFAULT gen_random_uuid()`); (3) the card index `ON memory_facts (tenant_id, user_id, status, updated_at DESC)`. No FK on `user_id` (identity is caller-supplied strings in MVP); `superseded_by` is a loose self-reference (no constraint — the audit trail is convention, same looseness as `source_conversation_id`).
- **`supersede_fact` must be atomic:** the UPDATE carries `AND status = 'active'` in its WHERE and `RETURNING *` — a lost race reads as the no-op `None` at the statement itself, before any belt-and-braces.
- **`updated_at` bumps on supersede** — it is the recency ordering key; the bump keeps the audit trail honest about when truth changed.
- **Factory:** one entry — `"memory": 0.0` in `CHAT_MODEL_PURPOSES` (deterministic distillation). Nothing else changes; `set_chat_model` already accepts any known purpose.
- **`domain/conversations.py`:** `ConversationRecord` gains `user_id: Optional[str] = None`; `_create_conversation(tenant_id, user_id=None)` inserts it (positional callers unaffected — the only caller is the service). `RETURNING *` already covers the new column for reads.
- **High-risk callouts:** the ALTER and CREATE both run inside the existing bootstrap transaction — `IF NOT EXISTS` everywhere is what makes re-deploys safe on populated databases; do NOT add a second bootstrap path (the `main.py` hotspot note in the ARCH).

### Scope Boundaries

- Do NOT touch `application/`, `graphs/`, or `api/` — nothing reads or writes facts yet (T2/T3).
- Do NOT add extraction, seeding, or prompt logic of any kind.
- Do NOT add an importance/weight column, TTL columns, or a Qdrant collection (out of scope + decision 1).
- Do NOT expose memories over HTTP (the "what do you know about me" endpoint is ARCH out-of-scope).
- Do NOT create test files; scratch verification scripts are deleted after runs.
- Only implement: the footprint files listed above.

### Files Expected

**New files:**
- `src/app/domain/memories.py` (`MemoryFact` + `add_fact`/`list_active_facts`/`supersede_fact`/`forget_user` public twins + private raisers)

**Modified files:**
- `src/app/infrastructure/postgres.py` (`_SCHEMA_STATEMENTS` += ALTER + table + index)
- `src/app/infrastructure/settings.py` (`MEMORY_ENABLED`, `MEMORY_MAX_FACTS`)
- `src/app/infrastructure/llm/factory.py` (`"memory"` purpose entry)
- `src/app/domain/conversations.py` (`ConversationRecord.user_id` + `_create_conversation` param)

**Must NOT modify:**
- `src/app/application/*`, `src/app/graphs/*`, `src/app/api/*`
- `src/app/main.py` (bootstrap_schema already covers the DDL — nobody adds a second bootstrap path)
- `tests/test_isolation.py`

---

## Task T2: Identity + Read Path — Header Binding, Card Seeding, Memory Prompt Blocks

> **Status:** not started
> **Verification:** checklist
> **Effort:** m
> **Priority:** high
> **Depends on:** T1
> **Satisfies REQs:** LM-R1 (cross-conversation recall — read half), LM-R2 (router + planner + composer consumption), LM-R5 (degrade-to-empty keeps turns alive), LM-R7 (anonymous behaves exactly as today)
> **Footprint slice:** Modified: src/app/api/conversations.py (X-User-Id parsing), src/app/application/conversations.py (read half: user_id threading, `_run_turn` card fetch + `memories` seeding), src/app/graphs/orchestrator/{state,router,planner,graph,composer}.py
> **High-risk areas touched:** Turn-path prompts (M) — a new context block on every card-carrying turn shifts routing/planning behavior (the feature working as intended; the guard is the neutral empty-card render + the anonymous before/after diff); resume path (M) — `_resume_paused_plan` seeds no `memories`, every consumer must render absence neutrally or resumed plans change behavior

### Description

The read half: an optional `X-User-Id` header binds a conversation's owner at creation; `_run_turn` fetches the owner's card once (try/except → `[]`), seeds it as the new optional `memories` state key, and the router, planner, and composer prompts render it as a "what you already know about this customer" block whose absence reads "(nothing known yet)". The graph imports nothing new — memory reaches it the way history does, as a service-seeded key. Cards stay empty until T3, so T2 is verified by seeding facts directly through T1's `add_fact`.

### Verification Checklist

_Fake chat models injected at the seam for every LLM touch (`router`/`planner`/`generator`); cards seeded directly via `add_fact`; prompts captured by fake models that record what they were invoked with; HTTP items via uvicorn/httpx scratch script; every script deleted after its run:_

- **Header binding** — expected: `POST /tenants/{t}/conversations` with `X-User-Id: "  u1  "` → the conversation row carries exactly `u1` (surrounding whitespace stripped); header absent or blank → `NULL`; the 201 response shape is unchanged. _(LM-R7)_
- **Owner bound once** — expected: `POST …/{cid}/messages` carrying any `X-User-Id` on an anonymous conversation changes nothing — the conversation's bound owner wins (the messages endpoint ignores user headers).
- **Card reaches the router** — expected: a user with 3 seeded facts sends a turn (fake router classifies direct) → the captured router prompt contains all 3 facts in recency order inside the memory block; with `MEMORY_MAX_FACTS=2` env-overridden, exactly the 2 most recent render.
- **Card reaches the planner** — expected: direct `planner_node` call on a state carrying `memories` (fake planner returns a valid plan text) → the captured planner prompt contains the block; same for `exhaustion_reply(message, history, memories=…)` (fake generator capture).
- **Card reaches the composer** — expected: `compose_reply` on a state with a completed run + `memories` → the captured `generator` prompt contains the block; `compose_final` likewise (the one `_generate` touch covers both).
- **Neutral renders** — expected: `memories=[]` (known user, empty card), missing key (resume seed), and anonymous conversation all render the byte-identical "(nothing known yet)" line in every one of the three consumers. _(LM-R5/R7)_
- **Anonymous before/after** — expected: a scripted anonymous turn (all purposes faked, fixed transcript) captured **before** T2 work begins produces a byte-identical reply/outcome re-run after — anonymous behavior is provably unchanged.
- **Degrade-to-empty** — expected: `memories.list_active_facts` patched in the scratch script to raise → the turn completes normally, one warning logged, prompts render the neutral line (memory is enhancement, never critical path). _(LM-R5)_
- **Lever off stops reads** — expected: `MEMORY_ENABLED=false` → no card fetch occurs (a patched call counter shows zero) and prompts render neutral — the rollback lever. _(LM-N5)_
- **Resume neutrality** — expected: the T10-style pause→decide walkthrough (mock path) completes; the resumed run's captured prompts render the neutral line (the resume seed carries no `memories`); the final answer appends append-order as before.
- **Envelope safety** — expected: one workflow run through the facade (mock action, as in prior checklists) behaves identically beside the extended orchestrator state — the optional key breaks nothing downstream.
- **No regressions** — expected: `uv run pytest` green; the other four HTTP endpoints behave unchanged.

### Implementation Notes

- **Import style for patchability:** the service imports `from app.domain import memories` (module reference, not `from … import list_active_facts`) so checklist scripts can patch `memories.list_active_facts` — same seam discipline the codebase already uses for singleton overrides.
- **Seeding rule (keeps anonymous state byte-identical):** `_run_turn` fetches only when `header.user_id` is set AND `MEMORY_ENABLED` — and seeds the `memories` key only when the fetched list is non-empty. Anonymous turns therefore invoke the graph with exactly today's state, which is what makes the before/after diff meaningful.
- **`state.py`:** `memories: list[str]` joins `OrchestratorState` (`total=False` makes it optional by construction); contents are plain fact strings in recency order — never `MemoryFact` objects (the graph sees text, like `history`).
- **Prompt block shape:** one helper per module (`_memory_text(memories)` in router.py; planner/composer inline or shared shape) rendering `What you already know about this customer:` + one `- fact` line per entry, or the literal "(nothing known yet)" — the exact neutral string is pinned by the checklist. Facts render inside the labeled data block, same trust treatment as transcript text (ARCH security note).
- **Threading:** `planner_node` and `validate_node`'s exhaustion call pass `state.get("memories") or []`; `generate_plan_text` and `exhaustion_reply` gain `memories: Optional[list[str]] = None` (default keeps every existing caller valid — the standalone `build_plan` path unchanged); `_generate` gains the memory text param with `compose_reply`/`compose_final` feeding it from state.
- **API:** `start()` reads `user_id: Optional[str] = Header(default=None, alias="X-User-Id")`, strips it, treats blank as absent, threads to `start_conversation(tenant_id, message, user_id)`; `send_message` signature unchanged (owner read from the header row inside `_run_turn`).
- **High-risk callouts:** the executor deliberately gets NO memory block (A7 — it works a validated plan); do not "helpfully" thread `memories` into executor prompts. `must_commit`, history budgeting, and all slice-1/2 behavior are untouched — the anonymous before/after diff is the proof.

### Scope Boundaries

- Do NOT add extraction, background tasks, or any write path (T3).
- Do NOT render memories in executor prompts or any workflow graph (A7).
- Do NOT touch `domain/memories.py`, schema, settings, or the factory (T1 files).
- Do NOT seed `memories` on the resume path (ARCH hotspot: absence must render neutrally, not be back-filled).
- Do NOT create test files.
- Only implement: the footprint files listed above.

### Files Expected

**New files:** none.

**Modified files:**
- `src/app/api/conversations.py` (`start()` header parsing + threading)
- `src/app/application/conversations.py` (`start_conversation` gains `user_id`; `_create_conversation` call; `_run_turn` card fetch + seeding)
- `src/app/graphs/orchestrator/state.py` (`memories` key)
- `src/app/graphs/orchestrator/router.py` (`_memory_text` + block in `classify()`)
- `src/app/graphs/orchestrator/planner.py` (`generate_plan_text` / `exhaustion_reply` optional param + block)
- `src/app/graphs/orchestrator/graph.py` (`planner_node` + `validate_node` threading)
- `src/app/graphs/orchestrator/composer.py` (`_generate` memory block — covers `compose_reply` + `compose_final`)

**Must NOT modify:**
- `src/app/graphs/orchestrator/executor.py`, `src/app/graphs/registry.py`, the five workflow graphs (the envelope rides untouched)
- `src/app/domain/*`, `src/app/infrastructure/*` (T1 files)
- `src/app/main.py`, `tests/test_isolation.py`

---

## Task T3: Write Path — Post-Turn Extraction Under the Per-User Lock

> **Status:** not started
> **Verification:** checklist
> **Effort:** l
> **Priority:** critical
> **Depends on:** T1 (accessors + purpose key); sequenced after T2 — the end-to-end item consumes T2's seeding
> **Satisfies REQs:** LM-R1 (write half — the card fills itself), LM-R3 (curated, not logged — extraction adds only durable facts and supersedes contradictions), LM-R5 (never breaks or delays a turn), LM-R6 (erasure end-to-end), LM-N2 (ids and counts in logs only), LM-N4 (correct at any worker count), LM-N5 (lever rollback)
> **Footprint slice:** Modified: src/app/application/conversations.py (write half: `MemoryOp`/`MemoryExtraction`, `_extract_memories`, background-task reference set, post-turn scheduling), README.md (header + levers + rollback)
> **High-risk areas touched:** Turn-path failure isolation (M) — a background LLM call must never leak latency or exceptions into a turn (A8); advisory-lock pool pressure (M) — `advisory_lock` holds one pool connection for the whole block, and the block includes the LLM call by design (A4); privacy surface (M) — fact content is customer data and must never reach a log (CR-N4 extension)

### Description

The write half: after a successful non-anonymous turn (reply persisted, turn lock released), the service schedules a fire-and-forget task that — under `pg_advisory_lock("memory:<tenant>:<user>")` — reads the owner's current card, shows it plus the exchange to the `"memory"`-purpose model as structured output (`add`/`supersede` ops, empty is valid and common), validates each op independently (decision 3), applies the valid ones, and logs one ids-and-counts line. One call, no retry (decision 2); every failure caught, logged, swallowed. Plus `forget_user` verified end-to-end and the README surface.

### Verification Checklist

_Fake `"memory"` chat model (and router/generator fakes) via the seam; concurrent items via `asyncio.gather` in the scratch script; marker strings prove log discipline; scripts deleted after runs:_

- **Extraction contract — add** — expected: a fake memory model returning `ops=[add ×2]` after a turn → two `active` rows with `source_conversation_id` = that conversation; content matches the ops verbatim.
- **Extraction contract — supersede** — expected: card carries fact F; fake returns `[add(new), supersede(F)]` → F flips to `superseded` with `superseded_by` = the new fact's id; the new fact is `active` — the card shows current truth, the rows keep the audit trail. _(LM-R3)_
- **Curation input** — expected: the captured memory-model prompt contains the current card (with fact ids, so supersede can name them) AND the turn's exchange (user message + assistant reply) — the model that writes also curates (A5).
- **Empty ops** — expected: the common case writes nothing and logs exactly one line with `ops_add=0 ops_supersede=0`.
- **Garbage / raising model** — expected: a fake that raises or returns unparseable structured output → caught, one warning logged, zero rows; the turn's response was already delivered and is untouched. _(LM-R5)_
- **No retry** — expected: the garbage path shows a call count of exactly **1** on the fake memory model (decision 2).
- **Invalid-op skip** — expected: a batch of `[supersede(unknown fact_id), add(valid)]` → the add lands, the supersede is skipped, and the log line's counts reflect it (`ops_add=1`, supersede not applied — decision 3).
- **Fire-and-forget timing** — expected: with the fake memory model resolving a deliberately slow future, the HTTP turn response returns before extraction completes; the module-level task reference set keeps the task alive to completion (not GC'd mid-flight). _(LM-R5)_
- **Crash isolation** — expected: the apply phase induced to raise (patched accessor) → logged and swallowed; the turn response unchanged; the turn's committed truth untouched (A8).
- **Lock serialization** — expected: two same-user extractions (two conversations) driven concurrently → they serialize on `memory:<tenant>:<user>`; both fakes supersede "the first active fact" yet exactly one supersede lands per target, no lost adds (the read-card + apply-ops section is atomic, N4).
- **Skip conditions** — expected: `MEMORY_ENABLED=false` → no fetch and no extraction task at all; anonymous conversation → no task; both paths leave zero rows and no memory log line.
- **Log discipline** — expected: grep over the run's entire captured log output for a distinctive marker fact string → zero hits; exactly one `memory tenant=… user=… conversation=… ops_add=N ops_supersede=M` line per extraction; `user_id` never more prominent than `tenant_id`. _(LM-N2)_
- **Erasure end-to-end** — expected: a user with facts → `forget_user` → the user's next turn renders "(nothing known yet)" and extraction starts from an empty card. _(LM-R6)_
- **The feature, end-to-end** — expected: conversation 1 (fake memory model emits an add-op from the exchange) → a brand-new conversation for the same `X-User-Id` shows the fact in its captured router prompt — recall across conversations with fakes only. _(LM-R1)_
- **README** — expected: documents `X-User-Id`, both levers, and the rollback story (`MEMORY_ENABLED=false` / drop the table); no regressions in `uv run pytest`.

### Implementation Notes

- **Models:** `MemoryOp` (`op: Literal["add", "supersede"]`, `fact_id: Optional[UUID]`, `content: Optional[str]`) and `MemoryExtraction` (`ops: list[MemoryOp]`, default empty) — the structured-output target, planner/`RouterDecision` style; the extraction model resolves lazily per call (the seam controls it) wrapped in `with_structured_output`.
- **Per-op validation (decision 3), pure functions:** `add` requires non-blank content (≤ the ~50–100-char convention enforced by prompt, not code); `supersede` requires a `fact_id` naming an active fact **of the snapshot card just read** — validation against the in-lock snapshot, then `supersede_fact`'s atomic WHERE is the second guard. Invalid ops are skipped and counted, never void the batch; op application order is emission order (a supersede followed by an add of the same content is the model's business).
- **Lock scope (A4, accepted cost):** the whole hook body — read card → build prompt → memory-model call → apply ops — runs inside `advisory_lock(f"memory:{tenant}:{user}")`. The lock helper holds one pool connection for the block, so extraction holds a connection for ~one LLM latency; it is off the turn path and serialized per user, and `POSTGRES_POOL_MAX_SIZE` is the documented pressure lever. This is the ARCH's declared trade-off (correctness over connection efficiency) — note it in the module docstring.
- **Use the raising twins inside the hook** (`_list_active_facts` & co. — the `records._get_conversation` precedent): `_extract_memories` is its own swallow point (A8), so infrastructure failures are one caught class, not pre-swallowed `[]` misread as "empty card".
- **Scheduling:** inside `_run_turn`, after the turn's persist succeeds and the turn advisory lock block exits — one scheduling point covers both entry points (`start_conversation` and `send_message` both funnel through `_run_turn`). Conditions: turn succeeded AND `header.user_id` AND `MEMORY_ENABLED`. `asyncio.create_task` with the task added to a module-level reference set (discard on done) so the loop never GCs an in-flight extraction.
- **What the prompt sees:** the current card (fact id + content per line) and the turn's exchange (the user message and assistant reply strings, passed explicitly — no transcript re-read). The system prompt constrains facts to durable **customer attributes** (never policies or instructions — the prompt-injection blunting), one self-contained fact each, and instructs: add only what is durable and new, supersede what is contradicted, emit empty ops otherwise.
- **Logging:** the one line `memory tenant=%s user=%s conversation=%s ops_add=%d ops_supersede=%d` (+ skipped counts where non-zero) — ids and counts only, never content (CR-N4 extension); one warning on the degrade/swallow paths.
- **README section:** the `X-User-Id` header contract (POST /conversations only, caller-asserted, maps to verified claims later), the two levers with defaults, and the two-step rollback story.

### Scope Boundaries

- Do NOT add importance weights, TTL/retention sweeps, semantic retrieval, or episodic memory (out of scope + decision 1 — v2 tripwire work).
- Do NOT run extraction in-turn, add a queue, or batch extractions (A4's fire-and-forget is the design).
- Do NOT retry extraction on invalid output (decision 2) — one call, caught, logged, swallowed.
- Do NOT expose any memory endpoint over HTTP (erasure is `forget_user` at the library level; a surface can come later).
- Do NOT log fact content anywhere, for any debugging convenience (LM-N2).
- Do NOT create test files.
- Only implement: the footprint files listed above.

### Files Expected

**New files:** none.

**Modified files:**
- `src/app/application/conversations.py` (write half: `MemoryOp`/`MemoryExtraction` models, `_extract_memories`, task reference set, post-turn scheduling in `_run_turn`)
- `README.md` (header + levers + rollback documentation)

**Must NOT modify:**
- All T1/T2 files beyond the `_run_turn` scheduling point already in this footprint
- `src/app/graphs/*` (the graph never learns about extraction), `src/app/api/*`, `src/app/domain/memories.py`
- `tests/test_isolation.py`

---

_All three tasks are `checklist` verification mode (carried convention): run-once scratch commands, deleted after their runs; no test files beyond the pre-existing `tests/test_isolation.py`. All LLM behavior — router, planner, generator, and the new memory extraction — is verified exclusively through fake chat models injected at the `set_chat_model` seam (developer decision, carried from 2026-11-17); real-model runs are deliberately absent from these checklists._
