# Tasks

> **Date:** 2026-11-17
> **Source:** specs/architecture/ARCH-predefined-langgraph-workflows.md (standalone brief — no REQ doc; inferred requirements R1–R11 in the ARCH)
> **Developer decisions made during task generation (2026-11-17):**
> (1) **Fakes-only LLM verification** — every LLM-dependent checklist item runs against langchain-core fake chat models injected at the `infrastructure/llm` factory seam; there are **no real-API-key checklist items** in any task. Accepted consequence: real-model behavior (prompt quality, structured-output robustness) is never exercised by these checklists.
> (2) The **fake-injection seam** (a `set_chat_model`-style override beside the `get_chat_model` singleton) is a **T1 deliverable** — it is load-bearing for T4/T5/T6/T7 verification.
> (3) MCP verification uses a **throwaway scratch stdio MCP server** (the `mcp` SDK arrives transitively via `langchain-mcp-adapters`); the server script is deleted after its run, never committed.
> (4) **MCP tenant allowlist is deny-by-default**: a tenant absent from `MCP_TENANT_ALLOWLIST` gets no tools, and running the live-data graph for it fails fast with a clear reason (T5).
> (5) **An action with no policy entry requires no approval** — mock-e-commerce stance for now; revisit before real tenants (T6).
> (6) **Ineligible `validate()` is a business failure**: terminal state with the handler's reason, `ActionResultRecord(failed)` written, `execute()` never called (T6).
> (7) `graphs/runtime.py` is deliberately sliced: T3 delivers the run surface; T7 adds resume/decide/chaining. Both slices declared here at generation time.
> (8) Eight tasks, split by subsystem; T-numbers are scoped to this document (the ingestion doc's T1–T7 are separate history).
> (9) No unit/integration test suite (ARCH R10/A10): all verification below is run-once checklist commands via scratch scripts — deleted after their runs, never committed; throwaway tenants/collections purged. `tests/test_isolation.py` remains the only committed test.

---

## Task T1: Graph Foundations — Dependencies, Settings, Enums, Records, LLM Factory, Checkpointer

> **Status:** done
> **Verification:** checklist
> **Effort:** l
> **Priority:** critical
> **Depends on:** None
> **Satisfies REQs:** R3 (tenant-scoped records), R9 (checkpointer + bounded-config foundations); enables R4/R6 (LLM factory)
> **Footprint slice:** New: src/app/graphs/base/state.py, src/app/domain/approvals.py, src/app/domain/action_results.py, src/app/infrastructure/llm/factory.py, src/app/infrastructure/checkpoint.py + package `__init__` files. Modified: pyproject.toml, src/app/infrastructure/settings.py, src/app/domain/enums.py, src/app/main.py, docker-compose.yml, .env.example, README.md, .gitignore
> **High-risk areas touched:** Dependency surface (M) — langgraph ↔ langchain-core ↔ langchain-openai ↔ mcp-adapters ↔ both checkpointers must co-resolve with the existing `langchain-text-splitters` pin; Qdrant instance (L) — two new payload-only collections; App startup (L)

### Description

Everything the four graphs install against, and nothing graph-shaped: the dependency matrix, the new settings groups, the two lifecycle enums, the two payload-only Qdrant records, the shared graph state envelope + `WorkflowResult`, the purpose-keyed chat-model factory (with the fake-injection seam all later verification drives through), and the pluggable checkpointer. After T1 the graphs subsystem has its entire non-graph substrate; T3–T7 only add graphs and wiring.

### Verification Checklist

- **`uv lock` + `uv sync`** — expected: resolves cleanly; `langgraph`, `langchain-core`, `langchain-openai`, `langchain-mcp-adapters`, `langgraph-checkpoint-sqlite`, `langgraph-checkpoint-postgres` co-exist with the existing `langchain-text-splitters`; an `uv run python -c` import smoke of all six succeeds.
- **Settings** — expected: constructs with the new groups and defaults; a deliberately missing required variable still fails with the clear pydantic field-required message.
- **Enums exact** — expected: `ApprovalStatus` has exactly `pending / approved / rejected / expired`; `ActionStatus` exactly `succeeded / failed`.
- **Record round-trip + isolation** — expected: insert `ApprovalRecord` + `ActionResultRecord` for tenants A and B; `bulk_find(tenant_id=A)` lists only A's; construction without `tenant_id` raises ValidationError; `curl http://localhost:6333/collections` shows `approvals` + `action_results` with `tenant_id` indexes (bootstrap list grew).
- **Chat factory** — expected: `get_chat_model("grader" / "generator" / "agent")` returns purpose-keyed models (temperature 0 / 0.3 / 0.2, A6); unknown purpose → immediate ValueError listing known purposes; **fake seam**: `set_chat_model("grader", fake)` makes subsequent `get_chat_model("grader")` calls return the fake.
- **Checkpointer** — expected: default `CHECKPOINT_BACKEND=sqlite` yields a usable (entered) sqlite checkpointer at `SQLITE_CHECKPOINT_PATH`; `CHECKPOINT_BACKEND=postgres` selects the postgres saver (class/DSN check only — live connection optional behind `docker compose --profile postgres up`); unknown backend fails fast.
- **Compose** — expected: `docker compose config` unchanged (qdrant only, default profile); `docker compose --profile postgres config` adds the postgres service.
- **No regressions** — expected: `uv run pytest` behaves exactly as before (isolation test); `uv run uvicorn app.main:app` boots with the two new bootstrap entries.

### Implementation Notes

- **Module(s):** `domain/`, `infrastructure/`, plus empty package inits under `graphs/` — per ARCH Module Boundaries (domain imports the Qdrant connection only).
- **Pattern references:** records mirror `domain/records.py` `DocumentRecord` (payload-only, dummy 1-dim vector, `ClassVar collection_name`, `tenant_id` required no default); the chat factory mirrors `infrastructure/embedding/factory.py` (provider dict, fail-fast unknown name, process-wide singleton); checkpointer follows the same factory style.
- **Settings additions (ARCH footprint):** `LLM_PROVIDER` (default `openai`), `CHAT_MODEL` (default `gpt-4o-mini`), `SELF_RAG_MAX_REWRITES`, `SELF_RAG_MAX_REVISIONS`, `SELF_RAG_TOP_K`, `LIVE_DATA_MAX_STEPS`, `MCP_SERVERS` (JSON), `MCP_TENANT_ALLOWLIST` (JSON), `CHECKPOINT_BACKEND` (default `sqlite`), `SQLITE_CHECKPOINT_PATH`, `POSTGRES_DSN`.
- **Default bounds must reproduce the ARCH latency budget (≤ ~25 LLM calls = 1 decide + 3 attempts × 8):** `SELF_RAG_MAX_REWRITES=2`, `SELF_RAG_MAX_REVISIONS=2`; propose `SELF_RAG_TOP_K=4`, `LIVE_DATA_MAX_STEPS=8`. All tunable — checklists in T4/T5 assert behavior *follows the settings*, never the specific values.
- **ApprovalRecord fields (ARCH data model):** `id` (= `request_id` = LangGraph `thread_id`), `tenant_id` (required, no default), `action_name`, `action_input: dict`, `reason`, `risk_level`, `approver_role`, `status: ApprovalStatus = pending`, `decision_note`/`decided_by` (optional audit), `thread_id`, `created_at`/`decided_at`. **ActionResultRecord:** `id`, `tenant_id` (required), `action_name`, `approval_id?`, `status: ActionStatus`, `result: dict`, `error?`, `created_at`. Lifecycle transitions are enforced by the runtime (T7), not the record models.
- **`graphs/base/state.py`:** `WorkflowState` envelope (TypedDict: `tenant_id: str`, `run_id: str`, `errors: list[str]`) extended by every graph; `WorkflowResult` (pydantic: `status: completed | pending_approval | rejected | failed`, `output`, `request_id?`, `error?`) — the facade's universal return.
- **High-risk callouts:** `AsyncSqliteSaver` must be *entered* before use — wire one process-wide entered instance (setup/async-context), and note the single-writer dev constraint; `>=` floors per project convention, verified to co-resolve at lock time; add the sqlite checkpoint path to `.gitignore`.

### Scope Boundaries

- Do NOT write any graph, node, grader, policy, registry, or runtime code (T3–T7).
- Do NOT touch `api/`, `ingestion/`, `application/preprocessing/`, `infrastructure/embedding/` beyond the settings/bootstrap lines listed.
- Do NOT create test files (developer decision, carried from the ingestion convention); scratch verification scripts are deleted after runs, never committed; throwaway tenants purged.
- Do NOT wire authentication anywhere (MVP has none; ARCH Open Question).
- Only implement: the substrate modules + the modified-file lines listed above.

### Files Expected

**New files:**
- `src/app/graphs/__init__.py`, `src/app/graphs/base/__init__.py` (package wiring)
- `src/app/graphs/base/state.py` (envelope + `WorkflowResult`)
- `src/app/domain/approvals.py` (`ApprovalRecord`, collection `approvals`)
- `src/app/domain/action_results.py` (`ActionResultRecord`, collection `action_results`)
- `src/app/infrastructure/llm/__init__.py`, `src/app/infrastructure/llm/factory.py` (`get_chat_model(purpose)` + fake-injection seam)
- `src/app/infrastructure/checkpoint.py` (`get_checkpointer()` sqlite | postgres)

**Modified files:**
- `pyproject.toml` + `uv.lock` (six new dependencies)
- `src/app/infrastructure/settings.py` (new settings groups)
- `src/app/domain/enums.py` (`ApprovalStatus`, `ActionStatus`)
- `src/app/main.py` (`payload_only` bootstrap list gains both records — else a never-inserted collection "reads as an outage")
- `docker-compose.yml` (postgres service under `profiles: ["postgres"]`)
- `.env.example`, `README.md` (document new settings + subsystem tree)
- `.gitignore` (sqlite checkpoint path)

**Must NOT modify:**
- `src/app/api/*`, `src/app/ingestion/*`, `src/app/application/preprocessing/*`, `src/app/infrastructure/{qdrant,embedding}/*`
- `src/app/domain/{records,documents,chunks}.py`, `src/app/domain/base/vector.py` (new records ride the OVM untouched)
- `tests/test_isolation.py`

---

## Task T2: Retrieval Service Extraction (contract-preserving)

> **Status:** done
> **Verification:** checklist
> **Effort:** s
> **Priority:** high
> **Depends on:** None (independent of T1 — may land in either order)
> **Satisfies REQs:** A9; prerequisite for R4 (the retrieve node consumes this service)
> **Footprint slice:** New: src/app/application/retrieval.py. Modified: src/app/api/search.py (delegation only)
> **High-risk areas touched:** Existing search API contract (M) — zero automated coverage; the only live-code refactor in the ARCH

### Description

Moves the body of `search()` (`api/search.py:50-99` — ready-record join, query embedding, three-format search, orphan/mid-pipeline filtering, top-k) into `application/retrieval.py` as a plain tenant-scoped service, leaving the endpoint as parsing + error mapping. The durable v1 external contract must survive **byte-identical**: same response shape, same error strings, same filtering semantics. This task exists in isolation on purpose — its before/after diff is the mitigation for having no automated net (ARCH backward-risk table).

### Verification Checklist

_A deterministic 1536-dim stub embedding provider injected at the existing embedding singleton (ingestion-doc pattern) keeps scores reproducible; scratch capture script deleted after the run:_

- **Before/after byte-diff** — expected: a fixed query battery (empty tenant; tenant with ready docs; a `processing` doc's content; an orphan chunk with no live record; `top_k` clamp; one query per case) captured against the **current** endpoint pre-refactor, then re-run post-refactor → response bodies identical.
- **Error strings identical** — expected: embedding failure → 500 `"embedding service unavailable"`; Qdrant failure → 500 `"vector db unavailable: search failed"` (both exercised with stubs that raise).
- **Orphan + mid-pipeline invisibility** — expected: neither ever appears in any response (A7/A8 ready-join preserved).
- **Structural check** — expected: `api/search.py` contains no embed/search/join logic (models + endpoint + error mapping only); `SearchRequest`/`SearchHit` untouched; the `tenant_records` import is dropped from `api/search.py` while `api/documents.py` and its own callers (`documents.py:105`, `documents.py:219`) are unchanged.

### Implementation Notes

- **Move the logic verbatim, not a rewrite.** Keep the service synchronous (matches the sync Qdrant client) and HTTP-agnostic (plain function, no FastAPI imports) so T4's retrieve node can call it via `asyncio.to_thread`.
- Keep the endpoint `def` (not `async def`) exactly as today, preserving thread-pool behavior.
- **Key decisions:** A9 (one implementation shared by HTTP route and the future graph node).
- **High-risk callouts:** the response shape `{content, document_id, score, metadata}` and both 500 strings are the v1 agent contract — any drift here breaks the future orchestrator; the byte-diff checklist item is the guard.

### Scope Boundaries

- Do NOT change any response field, status code, error string, or default value.
- Do NOT touch `api/documents.py`, `main.py`, `domain/*`, or anything else.
- Do NOT add the graph retrieve node yet (T4).
- Do NOT create test files.

### Files Expected

**New files:**
- `src/app/application/retrieval.py` (tenant-scoped retrieval service)

**Modified files:**
- `src/app/api/search.py` (delegates to the service; models + error mapping stay)

**Must NOT modify:**
- `src/app/api/documents.py` (its `tenant_records` behavior is a silent-regression hotspot per ARCH)
- Everything else outside the two files above.

---

## Task T3: Workflow Registry + Runtime Facade (run surface)

> **Status:** done
> **Verification:** checklist
> **Effort:** s
> **Priority:** critical
> **Depends on:** T1
> **Satisfies REQs:** R2 (generic extension point), R9 (uniform error surface, one outcome line per run)
> **Footprint slice:** New: src/app/graphs/registry.py, src/app/graphs/runtime.py (run surface only — resume/decide/chaining are T7's declared slice)
> **High-risk areas touched:** None (M/H) — additive library surface, no live code changes

### Description

The extension point and the facade every workflow goes through: `WorkflowDefinition` (`name`, `description`, `factory: Callable[[], CompiledStateGraph]`, `input_schema`) + `WORKFLOW_REGISTRY` with `register()`, and `run_workflow(name, tenant_id, input)` — validate input against the schema, inject the envelope (`tenant_id` only from the caller, fresh `run_id`), compile with cache, `ainvoke`, map to `WorkflowResult`. Infrastructure exceptions are caught here and only here → `failed`. A new workflow must be a subpackage plus one registration, touching nothing else (R2) — proven here with a scratch toy workflow, then consumed by T4–T7.

### Verification Checklist

_A scratch one-node toy workflow registered via `register()` in the verification script (never committed):_

- **Toy run** — expected: `run_workflow` returns `WorkflowResult(completed)` carrying the node's output.
- **Compile cache** — expected: a factory call counter shows exactly 1 compile across two runs.
- **Unknown workflow** — expected: immediate fail-fast error listing known names (embedding-factory style).
- **Uniform error surface** — expected: a node that raises → `WorkflowResult(failed, error=…)`; the exception never escapes the facade.
- **Envelope injection + outcome line** — expected: the toy echoes its state — `tenant_id` came only from the runtime argument, `run_id` generated; exactly one scannable log line `workflow=… tenant=… run_id=… status=… duration_ms=…` per run.
- **Input validation** — expected: a payload violating `input_schema` → clear error before the graph runs.

### Implementation Notes

- **Pattern references:** registry mirrors `infrastructure/embedding/factory.py` (dict + fail-fast unknown name); the facade is the single cross-graph wiring point (ARCH Module Boundaries — graphs never import each other).
- The runtime is async; **no checkpointer in this slice** (T7 attaches it for the approval graph).
- `register()` is callable from graph subpackage `__init__` files (self-registration); `graphs/__init__.py` imports subpackages as they land (T4–T7 each add their import line).
- `recursion_limit` from settings as the global backstop (A12).
- **Key decisions:** A2 (registry + single runtime facade = the genericity mechanism).

### Scope Boundaries

- Do NOT implement `resume_workflow` / `decide_approval` / cross-graph chaining (T7's declared slice).
- Do NOT build any real graph here (T4–T7).
- Do NOT touch `api/` or the HTTP surface (T8).
- Do NOT create test files.

### Files Expected

**New files:**
- `src/app/graphs/registry.py` (`WorkflowDefinition`, `WORKFLOW_REGISTRY`, `register()`)
- `src/app/graphs/runtime.py` (`run_workflow` + compile cache + error mapping)

**Modified files:** none.

**Must NOT modify:**
- T1/T2 files and the entire ingestion surface.

---

## Task T4: Self-RAG Knowledge Graph

> **Status:** done
> **Verification:** checklist
> **Effort:** l
> **Priority:** high
> **Depends on:** T2, T3
> **Satisfies REQs:** R4 (Self-RAG exactly per the reference architecture), R3 (tenant-scoped retrieval)
> **Footprint slice:** New: src/app/graphs/knowledge/{__init__, state, graph, nodes, graders}.py. Modified: src/app/graphs/__init__.py (import/registration line only)
> **High-risk areas touched:** None (M/H); documented LLM-call budget ≤ ~25 at default bounds is the performance note

### Description

The knowledge-retrieval graph, exactly per `self_rag.png` with ARCH-normalized names: `decide_retrieval → (generate_direct | retrieve → is_relevant → generate_from_context → is_sup ⇄ revise_answer → is_use → (END | rewrite_question → retrieve))`, with `no_answer_found` terminals and both loops bounded by settings counters. Graders are structured-output chat models from the purpose-keyed factory; the retrieve node calls T2's service with the envelope's tenant_id and nothing else.

### Verification Checklist

_Fake chat models injected via the T1 seam script every decision; the deterministic stub embedding provider keeps retrieval reproducible:_

- **Topology** — expected: the compiled graph's nodes are exactly the 10 from the PNG (`decide_retrieval, retrieve, generate_direct, is_relevant, generate_from_context, is_sup, revise_answer, is_use, rewrite_question, no_answer_found`); edges match: start→decide_retrieval; decide→{retrieve, generate_direct}; retrieve→is_relevant; is_relevant→{generate_from_context, no_answer_found}; generate_from_context→is_sup; is_sup→{revise_answer, is_use}; revise_answer→is_sup; is_use→{END, rewrite_question}; rewrite_question→retrieve; no_answer_found→END; generate_direct→END.
- **Direct path** — expected: decide=no → `generate_direct` → END; exactly **2 LLM calls** (decide + generate).
- **Happy path** — expected: relevant + supported + useful → END with the answer; exactly **5 LLM calls** (decide, is_relevant, generate_from_context, is_sup, is_use).
- **Irrelevant** — expected: all retrieved docs graded irrelevant → `no_answer_found` terminal with reason.
- **Revision bound** — expected: a fake whose `is_sup` always says "unsupported" loops `revise_answer` exactly `SELF_RAG_MAX_REVISIONS` times → `no_answer_found`; `recursion_limit` backstop also configured.
- **Rewrite bound** — expected: a fake whose `is_use` always says "not useful" loops `rewrite_question` exactly `SELF_RAG_MAX_REWRITES` times → `no_answer_found`; never an unbounded cycle.
- **Bounds from settings** — expected: env-overridden bounds change the observed loop counts accordingly (nothing hardcoded).
- **Tenant scoping** — expected: a document matching the query exists only for tenant B; running as tenant A → B's content never enters the assembled context (empty/irrelevant → no-answer path). _(R3)_

### Implementation Notes

- **Module(s):** `graphs/knowledge/` — orchestration only; no business rules, no HTTP.
- **Pattern references:** standard LangGraph StateGraph; graders = `get_chat_model("grader").with_structured_output(...)` (A6); generation nodes use `"generator"`. **All** LLM access goes through the factory — never a direct `ChatOpenAI` import in graph code.
- Loop counters live in the state (`revision_count`, `rewrite_count`); conditional edges check the bounds (A12).
- The retrieve node calls `application/retrieval` via `asyncio.to_thread` with `SELF_RAG_TOP_K`, tenant taken from the envelope only (structural isolation).
- "No results" is a *valid* outcome feeding the no-answer/rewrite paths; Qdrant unavailability raises → runtime maps to `failed` (the two failure classes).
- Registered as `self_rag` in `knowledge/__init__.py`; `graphs/__init__.py` gains the import.

### Scope Boundaries

- Do NOT add chunk-level grading, adaptive retrieval beyond `decide_retrieval`, or any strategy not present in the PNG (R4 fixes the scope).
- Do NOT touch the live-data/actions/approval packages, the runtime, or `api/`.
- Do NOT create test files.

### Files Expected

**New files:**
- `src/app/graphs/knowledge/__init__.py` (registers `self_rag`)
- `src/app/graphs/knowledge/state.py` (envelope extension: question, context, generations, counters)
- `src/app/graphs/knowledge/graph.py` (StateGraph wiring per the PNG)
- `src/app/graphs/knowledge/nodes.py` (node functions)
- `src/app/graphs/knowledge/graders.py` (structured-output graders)

**Modified files:**
- `src/app/graphs/__init__.py` (import line only)

**Must NOT modify:**
- T1–T3 files, the ingestion surface, `api/`.

---

## Task T5: Live-Data Graph + MCP Infrastructure

> **Status:** not started
> **Verification:** checklist
> **Effort:** m
> **Priority:** medium
> **Depends on:** T1, T3
> **Satisfies REQs:** R6 (agent + ToolNode + tools_condition), R3 (tenant allowlist)
> **Footprint slice:** New: src/app/infrastructure/mcp.py, src/app/graphs/live_data/{__init__, state, graph}.py. Modified: src/app/graphs/__init__.py (import/registration line only)
> **High-risk areas touched:** Multi-tenancy model (M) — allowlist misconfiguration = cross-tenant tool exposure; deny-by-default is the structural mitigation

### Description

Operational-data retrieval: `infrastructure/mcp.py` loads tools from `MCP_SERVERS` via `langchain-mcp-adapters` and filters them per-tenant by `MCP_TENANT_ALLOWLIST` (deny-by-default). The graph is the canonical prebuilt loop — agent (`bind_tools`) → `tools_condition` → `ToolNode` → agent → … → END when no tool calls remain — bounded by `LIVE_DATA_MAX_STEPS`.

### Verification Checklist

_A scratch stdio MCP server exposing two tools (e.g. `get_order_status`, `get_customer_email`) is spun up for the run and deleted after; fake chat model via the seam:_

- **Tool load** — expected: both tools load from the scratch server via the settings JSON.
- **Allowlist filtering** — expected: a tenant allowlisted for one tool gets exactly that tool; a tenant absent from the allowlist gets none → `run_workflow` fails fast with a clear reason (deny-by-default, decision 4). _(R3)_
- **Unreachable server** — expected: a wrong command in `MCP_SERVERS` → tool load fails at run start → `WorkflowResult(failed)` with a clear reason; no partial agent loop.
- **Tool round-trip** — expected: a scripted fake (first response emits a tool call, second is the final answer) drives agent → `ToolNode` → agent → END; the scratch **server's own log shows the invocation**; the answer reflects the tool's output.
- **Step bound** — expected: a fake that always emits tool calls stops the run at `LIVE_DATA_MAX_STEPS` with a bounded terminal state; no runaway loop.

### Implementation Notes

- **Module(s):** `infrastructure/mcp.py` (providers — settings only) + `graphs/live_data/` (orchestration only).
- Tool loading is async (session per run, or a cached client — document the choice in the module); tools attach via `get_chat_model("agent").bind_tools(tools)`.
- The step counter lives in the state and is checked before re-entering the agent node (A12); `tools_condition` + prebuilt `ToolNode`, no custom dispatch (R6).
- **High-risk callouts:** the allowlist filter must run at tool-load time, structurally — never as a post-hoc prompt filter.

### Scope Boundaries

- Do NOT implement a per-tenant MCP server registry or dynamic provisioning (ARCH Open Question — static settings now).
- Do NOT add non-MCP tools or custom tool dispatch.
- Do NOT create test files; the scratch MCP server script is deleted after the run, never committed.

### Files Expected

**New files:**
- `src/app/infrastructure/mcp.py` (`MCP_SERVERS` → tools; tenant allowlist filter)
- `src/app/graphs/live_data/__init__.py` (registers `live_data`)
- `src/app/graphs/live_data/state.py` (messages + step counter)
- `src/app/graphs/live_data/graph.py` (agent + ToolNode + tools_condition)

**Modified files:**
- `src/app/graphs/__init__.py` (import line only)

**Must NOT modify:**
- T1–T4 files, the ingestion surface, `api/`.

---

## Task T6: Action-Execution Graph + Policies + Mock Handlers

> **Status:** not started
> **Verification:** checklist
> **Effort:** l
> **Priority:** critical
> **Depends on:** T1, T3
> **Satisfies REQs:** R8 (generic ActionHandler registry; business rules never in graphs), R5 (approval-gate side), R9 (two failure classes), R3
> **Footprint slice:** New: src/app/graphs/actions/{__init__, contracts, graph, nodes}.py, src/app/graphs/actions/handlers/{__init__, ecommerce}.py, src/app/graphs/approval/__init__.py, src/app/graphs/approval/policies.py. Modified: src/app/graphs/__init__.py (import/registration line only)
> **High-risk areas touched:** None (M/H) — mock handlers only; the LLM-never-executes boundary is enforced by the runtime chaining (T7)

### Description

The generic executor: the `ActionHandler` Protocol (`validate` / `execute` / `verify`), `ActionDefinition` (`name`, `description`, `input_schema`), `ActionRegistry`, and the graph `resolve_handler → validate → approval_gate → execute → verify → record_result`. Business rules live only in handlers (mock `process_refund` / `cancel_order`) and in `approval/policies.py` (pure functions; the reference threshold policy). The gate ends the run as `pending_approval` carrying the full request payload — record persistence and graph chaining are T7's.

### Verification Checklist

_No LLM in this graph — mock handlers with a pre-seeded in-memory store; scratch scripts drive the runtime facade directly:_

- **Registry** — expected: `process_refund` / `cancel_order` resolve; an unknown action fails fast listing known names; an input violating `input_schema` → clear error before the graph runs.
- **Happy, no approval** (refund ≤ 500) — expected: `completed`; `ActionResultRecord(succeeded)` persisted tenant-scoped; one outcome log line.
- **Ineligible `validate()`** — expected: terminal state with the handler's reason; `ActionResultRecord(failed)` written; **execute call count 0** (decision 6).
- **`verify()` → False** — expected: `ActionResultRecord(failed)` with the unverified reason — execute did run, the discrepancy is visible for reconciliation (ARCH forward risk scenario).
- **Approval gate** (refund > 500) — expected: `WorkflowResult(pending_approval)` with `request_id` + `reason` / `risk_level` / `approver_role` payload; execute count 0; **no `ApprovalRecord` persisted yet** (chaining is T7).
- **Policy boundaries** — expected: amount exactly 500 → not required; 501 → required, `refund_manager`, high risk; `cancel_order` (no policy entry) → never gated (decision 5).
- **R8 genericity proof** — expected: a scratch third action (`block_card`, different schema + handler) registered and run through the **unmodified** graph → `succeeded` record. _(the ARCH's own proof that the abstraction works)_

### Scope Boundaries

- Do NOT implement real business integrations (mocks only — banking/hospital handlers are the drop-in proof, not shipped systems).
- Do NOT persist ApprovalRecords or chain graphs here (T7).
- Do NOT put eligibility rules, thresholds, or any business logic into nodes — only into handlers and policies (R8).
- Do NOT create test files.

### Files Expected

**New files:**
- `src/app/graphs/actions/__init__.py` (registers `action_execution`)
- `src/app/graphs/actions/contracts.py` (`ActionHandler`, `ActionDefinition`, `ActionRegistry`)
- `src/app/graphs/actions/graph.py` (resolve → validate → approval_gate → execute → verify → record_result)
- `src/app/graphs/actions/nodes.py` (node functions)
- `src/app/graphs/actions/handlers/__init__.py`, `src/app/graphs/actions/handlers/ecommerce.py` (mock `process_refund`, `cancel_order`)
- `src/app/graphs/approval/__init__.py`
- `src/app/graphs/approval/policies.py` (policy engine — created here because the gate consumes it; the approval graph itself is T7)

**Modified files:**
- `src/app/graphs/__init__.py` (import line only)

**Must NOT modify:**
- T1–T5 files, the runtime, the ingestion surface, `api/`.

---

## Task T7: Approval Graph + Runtime Chaining (resume / decide)

> **Status:** not started
> **Verification:** checklist
> **Effort:** l
> **Priority:** critical
> **Depends on:** T6 (and T1's checkpointer)
> **Satisfies REQs:** R5 (approved → action executes; rejected terminal), R9 (auditable decisions, bounded), R3
> **Footprint slice:** New: src/app/graphs/approval/{state, graph}.py. Modified: src/app/graphs/runtime.py (resume + decide_approval + cross-graph chaining + per-request_id lock — the slice declared at generation time)
> **High-risk areas touched:** Interrupt/checkpoint lifecycle (M) — the first stateful component (thread_id discipline, durable resume, topology-sensitive checkpoints); Multi-tenancy model (M) — cross-tenant decisions must be indistinguishable from unknown ones

### Description

The human gate and the completed flagship flow: `approval_graph` = `create_request` (persist `ApprovalRecord`, `thread_id = request_id`) → `resolve_approver` → `await_decision` (`interrupt()` + checkpoint) → `record_decision` (audit fields). The runtime gains the rest of the facade: on `pending_approval` it launches the approval graph; `resume_workflow` continues past an interrupt; `decide_approval` looks up the record, takes the per-`request_id` asyncio lock, re-checks status, resumes, and on approval re-invokes the action graph — `approval_id` set → re-validate, gate skipped → execute → verify → record_result.

### Verification Checklist

_Sqlite checkpointer; two processes where noted; mock handlers as in T6:_

- **Flagship approve** — expected: run refund > 500 → `pending_approval`; `ApprovalRecord(pending)` persisted with `thread_id == request_id`; a checkpoint exists (graph paused at `await_decision`).
- **Durable resume** — expected: `decide_approval(approved)` issued from a **fresh process** → record `approved` + `decided_by`/`note`/`decided_at`; action graph re-invoked with `approval_id`, gate skipped; `ActionResultRecord(succeeded, approval_id)` persisted; `WorkflowResult(completed)` carrying the action output.
- **Reject** — expected: record `rejected` with audit fields; handler execute count 0; `WorkflowResult(rejected)` — terminal, no re-invocation.
- **Double decision** — expected: a second `decide_approval` on the same request → 409-equivalent; record unchanged. Concurrent variant: two simultaneous decides via `asyncio.gather` → exactly one succeeds, the other 409 (lock + status re-check).
- **Unknown request / cross-tenant** — expected: decide on an unknown `request_id` → 404-equivalent; tenant B deciding tenant A's request → the same 404, no oracle. _(R3)_
- **Topology** — expected: approval graph nodes are exactly `create_request, resolve_approver, await_decision, record_decision`; no exception leaks on any business path.

### Implementation Notes

- **Module(s):** `graphs/approval/` (graph + state) + the runtime's declared second slice. Graphs still never import each other — **all** chaining lives in the runtime (A3).
- `thread_id == request_id == record id` (ARCH data model); `interrupt()` pauses; the resume value carries `{decision, decided_by, note}`.
- Only `pending → approved | rejected` is valid; enforcement lives in `decide_approval` (lock → re-check → act), not in the record model. Second attempts → 409-equivalent.
- The checkpointer from T1 attaches at compile for this graph (compile cache stays per workflow); sqlite is single-writer — the documented dev limit; the postgres path exists for multi-worker (ARCH Open Question).
- Carry the deploy note into the README: **drain or expire pending approvals before structural changes to the approval graph** (checkpoints are node-topology-sensitive).

### Scope Boundaries

- Do NOT implement approval expiry (enum value reserved, no mechanism — ARCH Open Question), two-person/sequential/time-boxed policies, or approver-role enforcement (no auth exists).
- Do NOT change the action graph or its handlers (T6 files) — chaining is runtime-only.
- Do NOT create test files.

### Files Expected

**New files:**
- `src/app/graphs/approval/state.py`
- `src/app/graphs/approval/graph.py` (interrupt + checkpoint wiring)

**Modified files:**
- `src/app/graphs/runtime.py` (`resume_workflow`, `decide_approval`, chaining, per-`request_id` lock — the declared slice)
- `README.md` (deploy/drain note) — documentation only

**Must NOT modify:**
- T1–T6 files, the ingestion surface, `api/` (T8).

---

## Task T8: Approvals HTTP API

> **Status:** not started
> **Verification:** checklist
> **Effort:** s
> **Priority:** high
> **Depends on:** T7
> **Satisfies REQs:** R11 (thin approvals surface), R3 (tenant-scoped HTTP)
> **Footprint slice:** New: src/app/api/approvals.py. Modified: src/app/main.py (router include)
> **High-risk areas touched:** Multi-tenancy model (M) — a new tenant-scoped HTTP surface

### Description

The two endpoints that make the interrupt→resume flow exercisable end-to-end before any orchestrator exists: GET a tenant's pending-approval queue, and POST a decision — on approve, the action executes synchronously and its result is returned in the response body.

### Verification Checklist

_uvicorn + curl (or an httpx scratch script); fake chat model where any LLM is touched:_

- **OpenAPI** — expected: exactly the 2 new endpoints beside the existing 5; the app boots clean.
- **GET queue** — expected: pending approvals for tenants A + B → A's list shows only A's; `?status=pending` filters; an empty tenant → `[]`.
- **POST approve** — expected: 200 with the `ActionResultRecord` fields in the body (synchronous execution, R11).
- **POST reject** — expected: 200 with the rejected outcome.
- **Errors** — expected: unknown `request_id` → 404; a second decision → 409; **cross-tenant decision (tenant B on tenant A's request) → 404 byte-identical to the unknown-id case** (no tenant-existence oracle). _(R3)_
- **Flagship over HTTP** — expected: a scratch `run_workflow` produces a pending approval → it appears in the GET queue → POST decision → the action result fields come back in the response.
- **Infra down** — expected: with Qdrant stopped, both endpoints fail with the existing vague-to-caller 500 style.
- **No regressions** — expected: `uv run pytest` (isolation test) green; the existing five endpoints behave unchanged.

### Implementation Notes

- **Pattern reference:** router style per `api/documents.py`; tenant identity from the URL path only (A3/A4); `decided_by` is declared audit identity, never enforcement (ARCH auth note).
- The endpoint is async and awaits the runtime facade (the only `api/` dependency for workflows, per Module Boundaries).
- **Key decisions:** A4 (thin surface exists so interrupt→resume is exercisable pre-orchestrator).

### Scope Boundaries

- Do NOT add action-trigger endpoints (actions are runtime/library consumers — the orchestrator's job later).
- Do NOT add authentication (MVP convention; ARCH Open Question).
- Do NOT create test files.

### Files Expected

**New files:**
- `src/app/api/approvals.py` (GET queue + POST decision)

**Modified files:**
- `src/app/main.py` (`include_router(approvals.router)` beside the existing two)

**Must NOT modify:**
- T1–T7 files, the ingestion surface, `tests/test_isolation.py`.

---

_All eight tasks are `checklist` verification mode (R10/A10): run-once scratch commands, deleted after their runs; no test files beyond the pre-existing `tests/test_isolation.py`. LLM behavior is verified exclusively through fake chat models injected at the T1 seam (developer decision, 2026-11-17) — real-model runs are deliberately absent from these checklists._
