# Architecture: Predefined LangGraph Workflows

> **Date:** 2026-11-17
> **Phase:** 2 of 5 (System Architecture)
> **Requirements source:** Standalone brief — see Inferred Requirements
> **Tasks:** TASKS-predefined-langgraph-workflows.md
> **Type:** feature

## Architecture Summary

A new `src/app/graphs/` subsystem adds four predefined LangGraph workflows to the existing tenant-scoped FastAPI + Qdrant service: a Self-RAG knowledge-retrieval graph (reflection loops per the reference architecture), a one-node live-data-fetch graph over MCP tools, a generic action-execution graph driven by registered `ActionHandler`s, and a checkpointed approval graph that pauses via `interrupt()` and, on human approval, triggers the action-execution graph. Genericity comes from a `WorkflowDefinition` registry plus a single runtime facade that compiles, runs, resumes, and chains every workflow — a new workflow is a subpackage plus one registry entry, touching no existing graph. Cross-graph wiring lives only in the runtime; business rules live only in handlers and approval policies; graphs own orchestration and nothing else. All state and records are structurally tenant-scoped, reusing the existing OVM/Qdrant isolation guarantees. The future orchestrator (planner–generator) is out of scope but is the primary intended consumer of every contract defined here.

## Inferred Requirements (Mode B — no REQ)

| ID  | Inferred Requirement                                                                 | Source |
|-----|----------------------------------------------------------------------------------------|--------|
| R1  | Four predefined workflows: Self-RAG knowledge retrieval, live data fetch, action execution, approval | Brief |
| R2  | New workflows introducible later without modifying existing graphs (generic extension point) | Brief ("generic so that in future any new workflow can be introduced without much hassle") |
| R3  | All workflows tenant-scoped with structural isolation, consistent with the existing OVM pattern | Brief (multi-tenant) + codebase convention |
| R4  | Knowledge retrieval implements Self-RAG exactly per the reference architecture (`self_rag.png`): decide-retrieval, relevance grading, groundedness/revision loop, usefulness grading, question-rewrite loop, no-answer and direct paths | Brief + `self_rag.png` |
| R5  | Approval workflow invokes the action-execution graph when approved; rejected approvals are terminal | Brief ("Approval will call data execution graph / transaction graph when approved") |
| R6  | Live data fetch is a minimal graph: agent node + `ToolNode` + `tools_condition`, tools provided by MCP servers assumed available | Brief |
| R7  | Orchestrator (planner–generator) excluded now; contracts must accommodate it as the primary future consumer | Brief |
| R8  | Action execution is generic via `ActionHandler` protocol + registry; business rules never enter graph code; mock e-commerce handlers ship first | `dicussion` (AI discussion, §2–§6) |
| R9  | Production-grade patterns: bounded reflection loops, auditable approval decisions, uniform error surface, no unbounded cycles | Brief ("design principles and patterns of production grade systems") |
| R10 | No unit/integration test suite; verification is run-once checklist commands (consistent with the ingestion-pipeline convention; only `tests/test_isolation.py` exists) | Developer decision, 2026-11-17 |
| R11 | Thin HTTP surface for approvals: GET pending queue + POST decision, so the interrupt→resume flow is exercisable end-to-end before the orchestrator exists | Developer decision (Q2-B), 2026-11-17 |

## High-Level Structure

```
src/app/
├── api/
│   ├── documents.py, search.py          (search.py modified → delegates to retrieval service)
│   └── approvals.py                     ← NEW: GET pending queue + POST decision
│
├── graphs/                              ← NEW SUBSYSTEM
│   ├── runtime.py                       ← facade: run/resume/chain, compile cache, checkpointer
│   ├── registry.py                      ← WorkflowDefinition + WORKFLOW_REGISTRY (extension point)
│   ├── base/state.py                    ← shared state envelope + WorkflowResult
│   ├── knowledge/                       ← self_rag_graph (state, graph, nodes, graders)
│   ├── live_data/                       ← agent + ToolNode + tools_condition (state, graph)
│   ├── actions/                         ← contracts (ActionHandler/Definition/Registry),
│   │                                     graph, nodes, handlers/ecommerce.py (mocks)
│   └── approval/                        ← graph (interrupt + checkpoint), policies.py
│
├── application/
│   └── retrieval.py                     ← NEW: extracted tenant-scoped retrieval service
│                                           (embed + multi-format search + ready-join)
├── domain/
│   ├── approvals.py                     ← NEW: ApprovalRecord (payload-only, Qdrant)
│   ├── action_results.py                ← NEW: ActionResultRecord (payload-only, Qdrant)
│   └── enums.py                         ← MODIFIED: + ApprovalStatus, ActionStatus
│
├── infrastructure/
│   ├── llm/                             ← NEW: get_chat_model(purpose) factory
│   ├── checkpoint.py                    ← NEW: checkpointer wiring (sqlite | postgres)
│   ├── mcp.py                           ← NEW: MCP server config → tools, tenant allowlist
│   └── settings.py                      ← MODIFIED: new settings groups
│
└── main.py                              ← MODIFIED: approvals router + bootstrap list
```

**Flagship end-to-end flow (approval-gated action):**

```
run action_execution_graph {tenant, action, input}
   ├─ resolve_handler → validate → approval_gate (policy engine)
   ├─ no approval needed → execute → verify → record_result → END(succeeded)
   └─ approval needed → END(pending_approval + ApprovalRequest)
         → runtime launches approval_graph
             create_request (persist ApprovalRecord, thread_id=request_id)
             → resolve_approver → await_decision  ⏸ interrupt() + checkpoint
         → human: POST /tenants/{t}/approvals/{id}/decision
         → runtime resumes approval graph → record_decision (+audit fields)
             ├─ rejected → END(rejected)
             └─ approved → END(approved)
                   → runtime re-invokes action_execution_graph (approval_id set:
                     re-validate, skip approval gate) → execute → verify → record_result
```

**Self-RAG flow** (per `self_rag.png`, node names normalized):

```
__start__ → decide_retrieval ──no──→ generate_direct → END
               │yes
               ▼
            retrieve ←─────────────────────┐
               ▼                           │ (rewrite_question loop, ≤ SELF_RAG_MAX_REWRITES)
            is_relevant ──irrelevant──→ no_answer_found → END
               ▼ relevant
       generate_from_context → is_sup ──unsupported──→ revise_answer ─┐
                                ▲             (≤ SELF_RAG_MAX_REVISIONS, │
                                └────────────── then no_answer_found)   │
                              supported                                   │
                                ▼                                        │
                             is_use ──not useful──→ rewrite_question ────┘
                                ▼ useful
                                END
```

**Live-data flow:** `agent (ChatOpenAI.bind_tools(mcp_tools)) → tools_condition → ToolNode → agent → … → END` when no tool calls remain; bounded by `LIVE_DATA_MAX_STEPS`.

## Tech Choices

| Area | Decision | Alternatives Considered | Rationale |
|------|----------|------------------------|-----------|
| Graph framework | `langgraph` (StateGraph, prebuilt `ToolNode`, `interrupt()`/`Command(resume=…)`) | hand-rolled asyncio orchestration; other orchestration libs | Brief specifies LangGraph; interrupt+checkpoint is native; prebuilt ToolNode required by R6 |
| LLM access in graphs | `langchain-openai` behind `infrastructure/llm` factory, purpose-keyed (`grader` temp 0, `generator` 0.3, `agent` 0.2) | raw `openai` SDK behind own protocol (mirrors embedding pattern) | Self-RAG graders are all structured output — `.with_structured_output()` gives battle-tested parse/retry; factory still centralizes provider config so a swap touches one module (D1 decision) |
| Checkpointer | pluggable via `CHECKPOINT_BACKEND`: `AsyncSqliteSaver` default, `AsyncPostgresSaver` behind compose profile `postgres` | sqlite-only; postgres-only | Zero new infra in dev; honest prod path wired now, not retrofitted (D2) |
| Business records store | Qdrant via existing `VectorBaseDocument` OVM (payload-only, dummy 1-dim vector) | Postgres tables | Zero new infra; structural tenant isolation for free, consistent with `DocumentRecord` precedent; migration path is one module since all access flows through OVM (D3) |
| MCP integration | `langchain-mcp-adapters` (`MultiServerMCPClient`), servers declared in settings, per-tenant allowlist | direct `mcp` SDK; per-tenant client hardcoding | Official LangChain integration; loads MCP tools as LangChain tools that feed `ToolNode` directly (R6, T2) |
| Concurrency | async-first graphs (`ainvoke` throughout) | sync graphs | LLM/MCP calls are IO-bound; FastAPI already async (T3) |
| Pinning style | `>=` lower bounds, matching existing `pyproject.toml` | exact pins | Project convention (T4) |

## Patterns & Conventions

- **Registry/factory/dispatch** — the project's established pattern (`embedding/factory.py`, `preprocessing/dispatchers.py`) extended three times: `WORKFLOW_REGISTRY`, `ActionRegistry`, LLM provider dict. Unknown names fail fast at startup with a clear message, exactly like `get_embedding_provider`.
- **Payload-only Qdrant records** — `ApprovalRecord`/`ActionResultRecord` mirror `DocumentRecord`: dummy 1-dim vector, `tenant_id` required with no default, `bulk_find`-style tenant listing.
- **Structural tenant isolation** — `tenant_id` is a required field of the shared graph state envelope, injected by the runtime; no graph or handler may accept tenant identity from any other source. MCP tools are filtered by allowlist at load time.
- **Bounded loops** — every cycle in every graph has a state-counter bound checked on the conditional edge (`SELF_RAG_MAX_REWRITES`, `SELF_RAG_MAX_REVISIONS`, `LIVE_DATA_MAX_STEPS`), with LangGraph's `recursion_limit` as backstop. No unbounded cycles (R9).
- **Two failure classes** — expected business failures are state transitions to terminal nodes with reasons (never exceptions); infrastructure failures raise and are caught only at the runtime facade → `WorkflowResult(status=failed)`.
- **Cross-graph calls through the runtime only** — graphs never import each other; the approval graph ends with `status=approved` and the runtime performs the action-graph invocation. Single-purpose graphs, one testable wiring point.
- **Intentionally NOT applied:** automated tests (R10 — run-once checklist verification instead, per project convention); auth on new endpoints (consistent with MVP; see Open Questions); streaming responses (deferred with the orchestrator).

## Data Models

### ApprovalRecord (`domain/approvals.py`)

**Purpose:** The durable, auditable approval request — the queue entity humans see and the resume key holder.

**Key fields:**

| Field | Type / Constraint | Notes |
|-------|-------------------|-------|
| `id` | UUID, pk | equals `request_id`; also the LangGraph `thread_id` |
| `tenant_id` | str, required, no default | structural isolation |
| `action_name` | str | registered action, e.g. `process_refund` |
| `action_input` | dict | schema-validated payload the action will run with |
| `reason` | str | why approval is required (from policy) |
| `risk_level` | str | low/medium/high (from policy) |
| `approver_role` | str | role expected to decide |
| `status` | `ApprovalStatus` | `pending → approved \| rejected \| expired` |
| `decision_note` / `decided_by` | optional str | audit fields, set on decision |
| `thread_id` | str | resume key (mirrors `id`) |
| `created_at` / `decided_at` | datetime | audit |

**Lifecycle:** created `pending` by the approval graph's `create_request` node → `approved`/`rejected` by `record_decision` (only valid transition from `pending`; second decision attempt → HTTP 409) → `expired` reserved (no mechanism yet — see Open Questions).

**Relationships:** 1 ApprovalRecord → 0..1 ActionResultRecord (`approval_id` back-reference).

### ActionResultRecord (`domain/action_results.py`)

**Purpose:** Audit of every action execution — success, failure, or unverified.

**Key fields:** `id`, `tenant_id` (required), `action_name`, `approval_id?`, `status: ActionStatus` (`succeeded|failed`), `result: dict`, `error?`, `created_at`.

**Lifecycle:** written once by the action graph's `record_result` terminal node; immutable.

### Transient graph models (`graphs/base/state.py`)

**State envelope** (TypedDict, extended by every graph): `tenant_id: str` (required), `run_id: str`, `errors: list[str]`.
**WorkflowResult** (pydantic, the facade's universal return): `status: completed | pending_approval | rejected | failed`, `output`, `request_id?`, `error?`.

## API Contracts / Interfaces

### Runtime facade (`graphs/runtime.py`) — library API, the orchestrator's future entry point

| Op | Signature | Purpose | Returns / Errors |
|----|-----------|---------|------------------|
| run | `run_workflow(name, tenant_id, input) -> WorkflowResult` | compile (cached) + `ainvoke`; for actions, chains into approval graph when policy requires | `completed`/`pending_approval`(+`request_id`)/`failed`; unknown workflow → immediate error |
| resume | `resume_workflow(name, tenant_id, thread_id, resume_value) -> WorkflowResult` | continue a checkpointed graph past its interrupt | resumed result / `failed` |
| decide | `decide_approval(tenant_id, request_id, decision, decided_by, note) -> WorkflowResult` | look up record → per-`request_id` lock + status re-check → resume approval graph → on approved, invoke action execute phase | `completed`(action result) / `rejected` / 404-equivalent / already-decided (409-equivalent) |

### Approvals HTTP API (`api/approvals.py`)

**Boundary:** HTTP, tenant-scoped under `/tenants/{tenant_id}` (A3 convention).

| Method | Path | Purpose | Errors |
|--------|------|---------|--------|
| GET | `/tenants/{tenant_id}/approvals?status=pending` | queue view for a human | 500 infra unavailable |
| POST | `/tenants/{tenant_id}/approvals/{request_id}/decision` body `{decision: approved|rejected, decided_by: str, note?: str}` | submit the human decision; on approve the action executes synchronously and its result is returned | 404 unknown request · 409 already decided · 500 infra |

**Auth requirements:** none in MVP (consistent with the existing API); `decided_by` is declared identity recorded for audit, not enforcement — see Open Questions.

### Action contracts (`graphs/actions/contracts.py`) — library API

- **`ActionHandler` Protocol:** `async validate(inputs, context) -> {eligible: bool, reason?}` · `async execute(inputs, context) -> dict` · `async verify(result, context) -> bool`. Business rules live only in implementations.
- **`ActionDefinition`:** `name`, `description`, `input_schema` (pydantic class; inputs validated before the graph runs).
- **`ActionRegistry`:** `name → (definition, handler)`; ships mock handlers `process_refund`, `cancel_order` (`graphs/actions/handlers/ecommerce.py`). Adding a banking handler must not touch the graph (R8 proof).

### Approval policy engine (`graphs/approval/policies.py`) — library API, pure functions

Per-action declarative conditions → `{required: bool, reason?, risk_level, approver_role}`. Ships the reference policy (`process_refund` amount > 500 → `refund_manager`). Richer policies (two-person, sequential, time-boxed) extend the engine later without graph changes (per discussion §3.3).

### Workflow extension point (`graphs/registry.py`)

**`WorkflowDefinition`:** `name` · `description` · `factory: Callable[[], CompiledStateGraph]` · `input_schema`. **New workflow = subpackage + one registry entry** (R2).

## Module Boundaries

| Module / Package | Responsibility | Allowed Dependencies |
|------------------|----------------|----------------------|
| `graphs/*` (graphs, nodes, graders, policies) | orchestration only | domain, infrastructure, application/retrieval; **never** `api/`, never sibling-graph internals |
| `graphs/runtime.py` | the only cross-graph wiring, checkpointer wiring, compile cache | registry, domain records, infrastructure |
| `graphs/actions/handlers/*` | business adapters (mocks now) | domain records; no graph imports |
| `api/*` | HTTP surface | runtime facade only (for workflows) |
| `application/retrieval.py` | tenant-scoped retrieval (embed + search + ready-join) | domain, embedding factory |
| `domain/*` | records, enums, OVM | Qdrant connection only (existing rule) |
| `infrastructure/llm, checkpoint, mcp` | providers | settings |

## Change Footprint

### New files / modules

| Path | Purpose | Pattern reference |
|------|---------|-------------------|
| `src/app/graphs/__init__.py`, `graphs/base/__init__.py` | package wiring | existing package style |
| `src/app/graphs/base/state.py` | state envelope + `WorkflowResult` | — |
| `src/app/graphs/registry.py` | `WorkflowDefinition` + `WORKFLOW_REGISTRY` + `register()` | `embedding/factory.py` registry style |
| `src/app/graphs/runtime.py` | facade: run/resume/decide, compile cache, chaining, per-`request_id` lock | — |
| `src/app/graphs/knowledge/{__init__,state,graph,nodes,graders}.py` | `self_rag_graph`: 10 nodes per PNG, structured graders, bounded loops | node names normalized from `self_rag.png` |
| `src/app/graphs/live_data/{__init__,state,graph}.py` | agent + `ToolNode` + `tools_condition`, step bound | standard prebuilt pattern |
| `src/app/graphs/actions/{__init__,contracts,graph,nodes}.py` | `ActionHandler`/`Definition`/`Registry`, `action_execution_graph` (resolve → validate → approval_gate → execute → verify → record_result) | `preprocessing/dispatchers.py` dispatch style |
| `src/app/graphs/actions/handlers/{__init__,ecommerce}.py` | mock `process_refund`, `cancel_order` | discussion §2.1 examples |
| `src/app/graphs/approval/{__init__,state,graph,policies}.py` | `approval_graph` (create_request → resolve_approver → await_decision[interrupt] → record_decision) + policy engine | discussion §3 |
| `src/app/domain/approvals.py` | `ApprovalRecord`, collection `approvals` | `domain/records.py` (`DocumentRecord`) |
| `src/app/domain/action_results.py` | `ActionResultRecord`, collection `action_results` | `domain/records.py` |
| `src/app/application/retrieval.py` | extracted retrieval service (embed + 3-format search + ready-join) | logic moved from `api/search.py:50-99` |
| `src/app/infrastructure/llm/{__init__,factory}.py` | `get_chat_model(purpose)` | `infrastructure/embedding/factory.py` |
| `src/app/infrastructure/checkpoint.py` | `get_checkpointer()` sqlite/postgres | factory style |
| `src/app/infrastructure/mcp.py` | `MCP_SERVERS` → tools; tenant allowlist filter | — |
| `src/app/api/approvals.py` | GET queue + POST decision | `api/documents.py` router style |

### Modified files / modules

| Path | What changes here |
|------|-------------------|
| `src/app/domain/enums.py` | add `ApprovalStatus`, `ActionStatus` StrEnums |
| `src/app/infrastructure/settings.py` | add `LLM_PROVIDER`, `CHAT_MODEL`, `SELF_RAG_MAX_REWRITES`, `SELF_RAG_MAX_REVISIONS`, `SELF_RAG_TOP_K`, `LIVE_DATA_MAX_STEPS`, `MCP_SERVERS` (JSON), `MCP_TENANT_ALLOWLIST` (JSON), `CHECKPOINT_BACKEND`, `SQLITE_CHECKPOINT_PATH`, `POSTGRES_DSN` |
| `src/app/main.py` | (a) `include_router(approvals.router)` beside `main.py:105-106`; (b) `payload_only` bootstrap list at `main.py:56-59` gains `ApprovalRecord`, `ActionResultRecord` (else a never-inserted collection "reads as an outage" per the module's own comment) |
| `src/app/api/search.py` | `search()` delegates to `application/retrieval.py`; `SearchRequest`/`SearchHit` models and error strings stay byte-identical (durable v1 contract); `api.documents.tenant_records` import drops |
| `pyproject.toml` | + `langgraph`, `langchain-core`, `langchain-openai`, `langchain-mcp-adapters`, `langgraph-checkpoint-sqlite`, `langgraph-checkpoint-postgres` (exact floors verified against the existing `langchain-text-splitters` pin at implementation) |
| `docker-compose.yml` | postgres service under `profiles: ["postgres"]` (off by default) |
| `.env.example`, `README.md` | document new settings; add subsystem tree |

### Deleted / replaced

None.

### Touched but not changed (silent-regression hotspots)

| Path | Why it matters |
|------|----------------|
| `src/app/api/documents.py` | `tenant_records()` keeps its own callers (`:105`, `:219`); the extraction must not change its behavior |
| `src/app/domain/base/vector.py` | new records ride the OVM untouched; approvals listing inherits `bulk_find`'s default `limit=50` pagination (fine now, documented) |
| `src/app/api/search.py` callers | the v1 external contract has **zero existing test coverage** and this is the only live-code refactor — mitigation: run-once manual verification against the documented contract (R10) |
| `tests/test_isolation.py` | must keep passing; new collections add tenant-scoped surfaces, not weaken them |
| `src/app/ingestion/pipeline.py`, `uv.lock` | coexistence only / mechanical re-lock |

## Areas of Impact

| Area | Impact | Risk (L/M/H) | Why |
|------|--------|--------------|-----|
| Existing search API contract | behavior-preserving extraction of its body | **M** | no automated coverage; mitigated by manual contract verification |
| Dependency surface | langgraph ↔ langchain-core ↔ langchain-openai ↔ mcp-adapters version matrix enters the project, must co-resolve with existing `langchain-text-splitters` | **M** | coherent `>=` floors + lock check at implementation |
| Multi-tenancy model | approvals API + MCP allowlist are new tenant-scoped surfaces | **M** | allowlist misconfig = cross-tenant tool exposure; structural filtering + checklist verification |
| Interrupt/checkpoint lifecycle | first stateful component in the system (thread_id discipline, resume path) | **M** | single writer (dev) / postgres path (prod); drain rule documented |
| Qdrant instance | two payload-only collections + bootstrap entries | **L** | dummy-vector precedent, same conventions |
| App startup | one router + two bootstrap entries | **L** | failed bootstrap already degrades gracefully |

**Contract changes:** none external. `POST /tenants/{tenant_id}/search` responses and error semantics are preserved byte-identical; the approvals API is additive. The runtime facade is a new *library* contract intended for the future orchestrator.

**Cross-cutting ripples:** `pyproject.toml`/`uv.lock` (new deps), `docker-compose.yml` (optional postgres profile), app bootstrap list, `.env.example` documentation.

## Cross-Cutting Concerns

- **Errors:** business failures = terminal state transitions with reasons (never exceptions); infrastructure failures raise → caught only at the runtime facade → `WorkflowResult(status=failed)`; API maps to codes in the existing vague-to-caller style (404/409/500). Retrieval service distinguishes "no results" (valid → `no_answer` path) from "Qdrant unavailable" (raise → failed run) — same semantics as today's endpoint.
- **Logging & metrics:** stdlib logging, one scannable outcome line per workflow run (`workflow=… tenant=… run_id=… status=… duration_ms=…`), plus approval pause/resume events. **No `action_input` payloads in logs** (customer data) — names and ids only. `run_id` is the correlation key the orchestrator inherits. No LangSmith/tracing yet.
- **Auth / authz:** none in MVP (consistent with existing API); `decided_by` is declared audit identity, not enforcement; MCP tenant allowlist enforced structurally at tool-load time.
- **Performance:** graphs compile once per process; only the approval graph pays checkpoint writes. Self-RAG LLM call budget is computable: 2 calls on the direct path, ≤ ~25 at default bounds (1 decide + 3 attempts × 8 grading/generation calls) — documented as the latency budget. SQLite checkpointer is single-writer (fine single-process); postgres profile exists for multi-worker. Approvals decision fetch is by point-id (O(1)); queue listing is scroll-paginated.
- **Security:** the hard boundary — **LLM output can propose actions but never execute them**; execution happens only via runtime chaining after a recorded human approval. Retrieved documents and MCP results are prompt-injection surfaces; the action graph is the mitigating boundary. Action inputs are schema-validated before graphs run; secrets stay in settings/.env.
- **Migrations / rollout:** fully additive; collections auto-create via existing bootstrap/first-insert fallback; rollback = revert (collections droppable, sqlite checkpoint file deletable). No migration tooling. Known deploy constraint: LangGraph checkpoints are node-topology-sensitive — drain or expire pending approvals before structural changes to the approval graph.

## Architecture Decisions Log

| # | Decision | Alternatives | Chosen Because | Satisfies |
|---|----------|--------------|----------------|-----------|
| A1 | Four graphs now: `self_rag`, `live_data`, `action_execution`, `approval`; orchestrator later | build orchestrator first | brief scopes workflows only; contracts designed for the orchestrator as first consumer | R1, R7 |
| A2 | Genericity via `WorkflowDefinition` registry + single runtime facade; new workflow = subpackage + one entry | per-workgraph bespoke wiring | satisfies "new workflow without hassle"; one compile/resume/chaining point | R2 |
| A3 | Two chained graphs for approval (not one interrupt-embedded graph); runtime owns all chaining | single checkpointed action graph containing approval node | developer decision (Q1-b); single-purpose graphs; approval graph reusable for non-action gates later | R5 |
| A4 | Thin approvals HTTP surface (GET queue + POST decision) in addition to in-process facade | facade only | developer decision (Q2-B); interrupt→resume must be exercisable end-to-end pre-orchestrator | R11 |
| A5 | Naming: `action_execution_graph`, `self_rag_graph`, package `graphs/` | `transaction_graph`, `knowledge_graph`, package `workflows/` | developer decision (Q3); avoids financial-"transaction" and graph-DB ambiguity | — |
| A6 | `langchain-openai` behind `infrastructure/llm` factory (purpose-keyed) | raw `openai` SDK protocol mirroring embeddings | graders are all structured output; factory still centralizes provider config | R4, R9 |
| A7 | Checkpointer pluggable: sqlite default, postgres behind compose profile | sqlite-only; postgres-only | zero dev infra now, prod path wired not retrofitted | R9 |
| A8 | `ApprovalRecord`/`ActionResultRecord` in Qdrant via OVM (payload-only) | Postgres tables | zero new infra; structural isolation; migration = one module | R3 |
| A9 | Extract `application/retrieval.py` from `api/search.py` (contract-preserving) | duplicate logic in graph node | one implementation shared by HTTP route and `retrieve` node | R4 |
| A10 | No automated tests; run-once checklist verification | committed pytest suite | developer decision, consistent with project convention; risk accepted on the single live-code refactor | R10 |
| A11 | MCP tools via `langchain-mcp-adapters`, settings-declared servers, per-tenant allowlist | direct MCP SDK; hardcoded clients | official integration feeding `ToolNode` directly; structural tenant filtering | R3, R6 |
| A12 | Bounded loops everywhere (state counters + `recursion_limit` backstop) | trusting LLM judgment to terminate | production-grade: no unbounded cycles | R9 |

## Risk & Stress-Test Scenarios

### Forward — runtime failure scenarios

| Scenario | How the Design Handles It |
|----------|---------------------------|
| LLM provider down 30s mid-run | exception → runtime facade → `WorkflowResult(failed)`; no partial writes. Sharper edge: failure between execute and verify → action ran, `ActionResultRecord(failed/unverified)`, approval stays `approved` — records show the discrepancy for reconciliation (accepted v1) |
| Same approval decided twice concurrently | per-`request_id` in-process asyncio lock + status re-check in `decide_approval`; second caller gets 409. Correct single-worker; multi-worker needs postgres path (documented limit) |
| Approvals grow 10K → 10M | payload-only points; decision fetch by point-id O(1); listing scroll-paginated — fine |
| Deploy graph changes while approvals pending | checkpoints are node-topology-sensitive → drain/expire pending approvals before structural approval-graph changes (documented deploy rule) |
| Human never decides | record stays `pending`; `expired` enum value reserved, no mechanism yet (Open Question) |
| MCP server unreachable | tool load fails at run start → `failed` result with clear reason; no partial agent loop |

### Backward — regression risk per touched area

| Touched area | What could regress | How we'd know / mitigation |
|--------------|--------------------|----------------------------|
| `api/search.py` extraction | v1 contract drift (response shape, 500 error strings) | run-once manual verification: same queries before/after refactor return identical bodies; no automated net (accepted, A10) |
| Dependency matrix | `langchain-core` pin conflict with existing `langchain-text-splitters` | `uv lock` resolution + import smoke check at implementation |
| `main.py` bootstrap | missing collections reading as outages | both new records added to `payload_only` list (footprint item) |
| Tenant isolation | cross-tenant approval/tool access | structural (path-scoped records, allowlist filter); manual cross-tenant check in the checklist (tenant B deciding tenant A's request → 404) |

## Open Questions

- **Approval expiry:** should pending approvals expire (enum value reserved, no timeout mechanism)?
  - **Impact if unresolved:** stale approvals accumulate in the queue view; resume of very old approvals may act on stale state.
  - **Suggested default:** defer; add a policy-engine time condition when a real tenant needs it.
- **Approver-role enforcement:** `decided_by` is declared identity; who may actually approve is unchecked until auth exists.
  - **Impact if unresolved:** anyone reaching the API can approve — acceptable only while the MVP has no auth at all.
  - **Suggested default:** enforce role matching in `decide_approval` when authentication lands; keep the audit fields as-is.
- **Multi-worker decision locking:** the per-`request_id` asyncio lock is process-local.
  - **Impact if unresolved:** under multi-worker uvicorn before postgres checkpointer adoption, a same-moment double decision could race.
  - **Suggested default:** document single-worker until `CHECKPOINT_BACKEND=postgres`; then move the guard to a DB-level check.
- **MCP server provisioning model:** who defines servers per tenant (static settings vs. per-tenant registry) once real tenants exist.
  - **Impact if unresolved:** settings JSON is fine for dev/mocks; per-tenant self-service needs a store.
  - **Suggested default:** static settings now; revisit with the orchestrator phase.

## Out of Scope

- Orchestrator (planner–generator) — later phase; consumes the runtime facade (R7).
- Real business integrations — mock e-commerce handlers only; banking/hospital handlers are the drop-in proof, not shipped systems.
- Authentication/authorization — consistent with MVP; flagged in Open Questions.
- Approval expiry mechanism — policy-engine extension later.
- Automated test suite — per developer decision (R10); verification is run-once checklists.
- Streaming responses, LangSmith/tracing — orchestrator-phase concerns.
- Two-person / sequential / time-boxed approval policies — engine supports extension; only threshold policies ship.
