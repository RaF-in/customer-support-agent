"""Runtime facade — the one door every workflow runs through.

``run_workflow(name, tenant_id, input)`` is the library API the future
orchestrator consumes (A2): registry lookup → input-schema validation →
compile (cached, once per process) → ``ainvoke`` → ``WorkflowResult``. This
module is the *only* place cross-graph wiring may live (ARCH Module
Boundaries — graphs never import each other). T3 delivered the run surface;
T7 added the rest of the facade: approval chaining on ``pending_approval``
(``_pause_for_approval`` launches the approval graph, which persists the
``ApprovalRecord`` and pauses at its ``interrupt()``), the raw
``resume_workflow`` past an interrupt, and the guarded ``decide_approval``
(per-``request_id`` lock → status re-check → resume → action re-invocation
with ``approval_id`` set). The checkpointer attaches here, at compile, for
the approval graph — the only checkpointed workflow.

Conventions the graphs (T4–T7) build against:

- **Envelope injection (R3):** the runtime merges the validated payload with
  the ``WorkflowState`` envelope — ``tenant_id`` comes ONLY from the caller's
  argument (a same-named input key is overwritten, never honored), ``run_id``
  is a fresh UUID per run, ``errors`` starts empty. No graph or handler can
  read tenant identity from any other source.
- **Outcome keys:** a graph communicates its terminal outcome through
  optional final-state keys — ``status`` (a ``WorkflowStatus`` value; absent
  means ``completed``), ``output``, ``request_id``, ``error``. The facade maps
  exactly these onto ``WorkflowResult`` and nothing else (internal state such
  as loop counters never leaks into the result).
- **Two failure classes (R9):** expected business failures are state
  transitions to terminal nodes (statuses, never exceptions); infrastructure
  exceptions — from the factory, a node, or an LLM/tool call inside a node —
  are caught here and only here, surfacing as
  ``WorkflowResult(status=failed)``. An exception never escapes the facade.

Logging: exactly one scannable outcome line per run —
``workflow=… tenant=… run_id=… status=… duration_ms=…`` — with ``run_id`` as
the correlation key the orchestrator inherits. No input payloads are ever
logged (customer data); names and ids only.
"""

import asyncio
import logging
import time
import uuid
from typing import Any, Optional
from uuid import UUID

from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from app.domain.approvals import ApprovalRecord
from app.domain.enums import ApprovalStatus
from app.graphs.approval.graph import build_approval_graph
from app.graphs.approval.state import ApprovalGraphInput
from app.graphs.base.state import WorkflowResult, WorkflowState, WorkflowStatus
from app.graphs.registry import get_workflow
from app.infrastructure.checkpoint import get_checkpointer
from app.infrastructure.qdrant import get_connection
from app.infrastructure.settings import Settings

logger = logging.getLogger(__name__)

# Compile cache: one factory call per workflow per process (ARCH performance
# note — graphs compile once). A concurrent first-invocation race can only
# cost one redundant compile of an equivalent graph; last write wins.
_compiled_graphs: dict[str, CompiledStateGraph] = {}

# Flat slack added to the deepest legitimate path implied by the configured
# bounds (see ``_recursion_limit``).
_RECURSION_HEADROOM = 25


def _recursion_limit() -> int:
    """Global recursion backstop (A12), derived from the loop-bound settings.

    Every loop's real bound is enforced by its own graph's state counter on
    the conditional edge; this process-wide ``recursion_limit`` exists only so
    an unguarded or buggy cycle hits ``GraphRecursionError`` (caught by the
    facade → ``failed``) instead of spinning forever.

    No dedicated recursion setting exists and T3 may not add one —
    ``settings.py`` is T1's file — so the limit is computed as an upper bound
    on the deepest legitimate path the configured bounds can imply, plus
    headroom: Self-RAG's worst case is ``decide`` + ``(rewrites + 1)`` attempts
    of ``(2 × revisions + 6)`` nodes + terminals; live-data's is two nodes per
    step round. Any future loop-bounded graph stays comfortably under it at
    sane bound values.
    """
    settings = Settings()
    self_rag_depth = (settings.SELF_RAG_MAX_REWRITES + 1) * (
        2 * settings.SELF_RAG_MAX_REVISIONS + 6
    ) + 4
    live_data_depth = 2 * settings.LIVE_DATA_MAX_STEPS + 2
    return max(self_rag_depth, live_data_depth) + _RECURSION_HEADROOM


# --------------------------------------------------------------------------
# Approval chaining (T7) — the second facade slice.
# --------------------------------------------------------------------------

# The approval workflow's name. Resolvable by ``resume_workflow`` but never
# registry-run: this facade launches it on ``pending_approval`` and always
# compiles it with the checkpointer attached (a registry factory call would
# compile it checkpointer-less, and ``interrupt()`` without a checkpointer
# can never pause).
APPROVAL_WORKFLOW_NAME = "approval"

# Lazily-compiled checkpointed approval graph — once per process, like the
# generic compile cache.
_approval_graph: Optional[CompiledStateGraph] = None

# Per-``request_id`` asyncio locks: exactly one decision may act on an
# approval at a time (ARCH forward-risk scenario — same approval decided
# twice concurrently). Process-local by design: single-worker is the
# documented dev constraint; multi-worker needs the postgres checkpointer
# path plus a DB-level guard (ARCH Open Question). Entries are small and
# never cleaned — one lock object per request ever decided in-process.
_request_locks: dict[str, asyncio.Lock] = {}


class ApprovalNotFoundError(LookupError):
    """404-equivalent: unknown ``request_id`` — or another tenant's, which
    is deliberately indistinguishable (no tenant-existence oracle, R3)."""


class ApprovalAlreadyDecidedError(RuntimeError):
    """409-equivalent: the request is no longer ``pending`` (only
    ``pending → approved | rejected`` is valid, and the runtime — not the
    record model — enforces it)."""


def _build_graph_input(tenant_id: str, run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Merge the validated payload with the envelope — envelope keys win.

    ``tenant_id``/``run_id``/``errors`` are set last so a payload key of the
    same name is always overwritten: tenant identity is structurally
    runtime-injected (R3), never taken from the input.
    """
    merged: dict[str, Any] = dict(payload)
    merged.update(WorkflowState(tenant_id=tenant_id, run_id=run_id, errors=[]))
    return merged


def _to_result(name: str, final: Any) -> WorkflowResult:
    """Map a graph's final state onto ``WorkflowResult`` via the outcome keys.

    Raises RuntimeError (caught by the caller's uniform error surface) when
    the final state is not a dict or carries an invalid ``status`` value —
    both are graph bugs, surfaced as ``failed`` rather than escaping.
    """
    if not isinstance(final, dict):
        raise RuntimeError(
            f"workflow {name!r} ended with a non-dict final state "
            f"({type(final).__name__})"
        )
    raw_status = final.get("status", WorkflowStatus.COMPLETED)
    try:
        status = WorkflowStatus(raw_status)
    except ValueError:
        raise RuntimeError(
            f"workflow {name!r} ended with invalid status {raw_status!r}"
        ) from None
    return WorkflowResult(
        status=status,
        output=final.get("output"),
        request_id=final.get("request_id"),
        error=final.get("error"),
    )


async def run_workflow(
    name: str, tenant_id: str, input: dict[str, Any]
) -> WorkflowResult:
    """Validate, compile (cached), and run workflow ``name`` for one tenant.

    ``input`` is validated against the workflow's ``input_schema`` before
    anything runs; ``tenant_id`` is the only source of tenant identity and
    overrides any same-named input key (R3).

    Raises:
        ValueError: unknown workflow name (message lists the known ones).
        pydantic.ValidationError: ``input`` violates the workflow's schema.

    Returns:
        ``WorkflowResult`` — the graph's declared terminal status (default
        ``completed``) on a normal end, or ``failed`` carrying the exception
        message when anything inside the run raises; the exception itself
        never escapes this facade (R9). Exactly one outcome log line is
        emitted per run.
    """
    definition = get_workflow(name)  # fail-fast: unknown name raises here
    validated = definition.input_schema.model_validate(input)  # fail-fast: bad input raises here
    payload = validated.model_dump()

    run_id = str(uuid.uuid4())
    started = time.perf_counter()
    try:
        graph = _compiled_graphs.get(name)
        if graph is None:
            graph = definition.factory()
            _compiled_graphs[name] = graph

        final = await graph.ainvoke(
            _build_graph_input(tenant_id, run_id, payload),
            config={"recursion_limit": _recursion_limit()},
        )
        result = _to_result(name, final)
        if result.status == WorkflowStatus.PENDING_APPROVAL:
            # T7 chaining: a gated run hands off to the approval graph,
            # which persists the pending ``ApprovalRecord`` and pauses at
            # its interrupt. The caller still sees the original
            # pending_approval result (request_id unchanged); a failure to
            # persist the request is an infrastructure failure → failed —
            # a human must never be shown a request that does not exist.
            result = await _pause_for_approval(tenant_id, run_id, result)
    except Exception as error:
        # The single catch point (R9): infrastructure failures become a failed
        # result — the caller sees a WorkflowResult, never an exception.
        logger.exception("Workflow %s crashed during run %s", name, run_id)
        result = WorkflowResult(
            status=WorkflowStatus.FAILED,
            error=str(error) or repr(error),
        )
    duration_ms = int((time.perf_counter() - started) * 1000)
    logger.info(
        "workflow=%s tenant=%s run_id=%s status=%s duration_ms=%d",
        name,
        tenant_id,
        run_id,
        result.status.value,
        duration_ms,
    )
    return result


async def _get_approval_graph() -> CompiledStateGraph:
    """Compile the approval graph once per process, checkpointer attached.

    Checkpointer wiring is this facade's job (ARCH Module Boundaries); the
    graph builder itself stays checkpointer-agnostic. The process-wide
    entered saver (T1) is shared — sqlite is single-writer, the documented
    dev limit.
    """
    global _approval_graph
    if _approval_graph is None:
        _approval_graph = build_approval_graph(
            checkpointer=await get_checkpointer()
        )
    return _approval_graph


async def _fetch_approval(
    tenant_id: str, request_id: str
) -> Optional[ApprovalRecord]:
    """Point-id fetch of one approval record, tenant-scoped (O(1)).

    Returns ``None`` when the id is unknown OR belongs to another tenant —
    deliberately indistinguishable (R3: no tenant-existence oracle; the
    cross-tenant probe reads exactly like the unknown-id one). Qdrant
    failures raise — infrastructure class, mapped to ``failed`` by the
    caller.
    """
    def _retrieve() -> Optional[ApprovalRecord]:
        points = get_connection().client.retrieve(
            collection_name=ApprovalRecord.collection_name,
            ids=[request_id],
            with_payload=True,
        )
        if not points:
            return None
        return ApprovalRecord.from_record(points[0])

    record = await asyncio.to_thread(_retrieve)
    if record is None or record.tenant_id != tenant_id:
        return None
    return record


async def _pause_for_approval(
    tenant_id: str, run_id: str, result: WorkflowResult
) -> WorkflowResult:
    """Launch the approval graph for a gated run (A3 — runtime chaining).

    Validates the gate's payload against ``ApprovalGraphInput``, then runs
    ``create_request`` (persist ``ApprovalRecord`` pending, thread_id =
    request_id) → ``resolve_approver`` → ``await_decision``, which pauses
    at its ``interrupt()`` under the thread-id checkpoint. Returns the
    caller's original ``pending_approval`` result unchanged.
    """
    payload = ApprovalGraphInput.model_validate(
        {"request_id": result.request_id, **(result.output or {})}
    )
    graph = await _get_approval_graph()
    await graph.ainvoke(
        _build_graph_input(tenant_id, run_id, payload.model_dump(mode="json")),
        config={
            "recursion_limit": _recursion_limit(),
            "configurable": {"thread_id": str(payload.request_id)},
        },
    )
    logger.info(
        "approval request paused (tenant=%s run_id=%s request_id=%s)",
        tenant_id,
        run_id,
        payload.request_id,
    )
    return result


async def resume_workflow(
    name: str,
    tenant_id: str,
    thread_id: str,
    resume_value: dict[str, Any],
) -> WorkflowResult:
    """Continue a checkpointed workflow past its interrupt (facade op).

    The approval graph is currently the only checkpointed workflow; it is
    resolved by name here rather than through the public registry (it is
    facade-launched and must always compile with the checkpointer
    attached). ``resume_value`` is what ``interrupt()`` hands the paused
    node — for the approval graph ``{decision, decided_by, note}``.

    This is the *raw* resume; the guarded, lock-protected entry point for
    approvals is ``decide_approval``. Raises ValueError for a non-resumable
    name (fail-fast, registry style); infrastructure failures surface as
    ``failed`` like every facade op. One outcome line per resume.
    """
    if name != APPROVAL_WORKFLOW_NAME:
        raise ValueError(
            f"Workflow {name!r} has no checkpointer and cannot resume. "
            f"Resumable workflows: {APPROVAL_WORKFLOW_NAME}."
        )
    run_id = str(uuid.uuid4())
    started = time.perf_counter()
    try:
        graph = await _get_approval_graph()
        final = await graph.ainvoke(
            Command(resume=resume_value),
            config={
                "recursion_limit": _recursion_limit(),
                "configurable": {"thread_id": thread_id},
            },
        )
        decision = final.get("decision") if isinstance(final, dict) else None
        # "approved" is not a facade status (A3): the approval *workflow*
        # completed its job — the runtime chains the action separately — so
        # the raw resume maps it to completed and carries the decision in
        # output; only a rejection maps to rejected.
        result = WorkflowResult(
            status=(
                WorkflowStatus.REJECTED
                if decision == "rejected"
                else WorkflowStatus.COMPLETED
            ),
            output=final.get("output") if isinstance(final, dict) else None,
            request_id=thread_id,
        )
    except Exception as error:
        logger.exception("Workflow %s crashed during resume %s", name, run_id)
        result = WorkflowResult(
            status=WorkflowStatus.FAILED,
            request_id=thread_id,
            error=str(error) or repr(error),
        )
    duration_ms = int((time.perf_counter() - started) * 1000)
    logger.info(
        "workflow=%s tenant=%s run_id=%s status=%s duration_ms=%d resume=1",
        name,
        tenant_id,
        run_id,
        result.status.value,
        duration_ms,
    )
    return result


async def decide_approval(
    tenant_id: str,
    request_id: "str | UUID",
    decision: str,
    decided_by: str,
    note: Optional[str] = None,
) -> WorkflowResult:
    """Record one human decision and act on it (the guarded resume).

    Per-``request_id`` lock → tenant-scoped record fetch → ``pending``
    re-check → resume the approval graph (``record_decision`` persists the
    audit fields) → on approval re-invoke the action-execution graph with
    ``approval_id`` set (re-validate, gate skipped, execute, verify,
    record); a rejection is terminal — no re-invocation (R5). The
    lock-plus-recheck is what makes a concurrent double decision fail safe:
    the second caller sees the already-decided record and raises before
    touching any state.

    Raises (business class — the HTTP layer maps these in T8):
        ValueError: ``decision`` is not ``approved``/``rejected``.
        ApprovalNotFoundError: unknown or malformed ``request_id``, or
            another tenant's request — 404-equivalent, indistinguishable
            (R3).
        ApprovalAlreadyDecidedError: the request is no longer pending —
            409-equivalent; the record is unchanged.

    Infrastructure failures return ``failed`` — an exception never escapes
    the facade (R9).
    """
    if decision not in ("approved", "rejected"):
        raise ValueError(
            f"Invalid decision {decision!r}. Expected 'approved' or 'rejected'."
        )
    try:
        key = str(UUID(str(request_id)))
    except ValueError:
        raise ApprovalNotFoundError(
            f"No pending approval {str(request_id)!r} for this tenant."
        ) from None

    started = time.perf_counter()
    lock = _request_locks.setdefault(key, asyncio.Lock())
    async with lock:
        try:
            record = await _fetch_approval(tenant_id, key)
            if record is None:
                raise ApprovalNotFoundError(
                    f"No pending approval {key!r} for this tenant."
                )
            if record.status != ApprovalStatus.PENDING:
                raise ApprovalAlreadyDecidedError(
                    f"Approval {key!r} is already {record.status.value}."
                )
            resumed = await resume_workflow(
                APPROVAL_WORKFLOW_NAME,
                tenant_id,
                key,
                {"decision": decision, "decided_by": decided_by, "note": note},
            )
            if resumed.status == WorkflowStatus.FAILED:
                # The record was never updated (its persist raised) — the
                # request stays pending and a retry is safe.
                return resumed
            if decision == "rejected":
                outcome = WorkflowResult(
                    status=WorkflowStatus.REJECTED,
                    request_id=key,
                    output=resumed.output,
                )
            else:
                # Approved → execute the gated action now (R5). The action
                # graph re-validates (the world may have changed since the
                # gate) and skips its approval gate on approval_id; its own
                # facade call maps any failure to failed.
                outcome = await run_workflow(
                    "action_execution",
                    tenant_id,
                    {
                        "action_name": record.action_name,
                        "inputs": record.action_input,
                        "approval_id": key,
                    },
                )
        except (ApprovalNotFoundError, ApprovalAlreadyDecidedError):
            raise
        except Exception as error:
            logger.exception("Approval decision crashed for request %s", key)
            outcome = WorkflowResult(
                status=WorkflowStatus.FAILED,
                request_id=key,
                error=str(error) or repr(error),
            )
    duration_ms = int((time.perf_counter() - started) * 1000)
    logger.info(
        "workflow=%s.decide tenant=%s request_id=%s decision=%s status=%s"
        " duration_ms=%d",
        APPROVAL_WORKFLOW_NAME,
        tenant_id,
        key,
        decision,
        outcome.status.value,
        duration_ms,
    )
    return outcome
