"""Runtime facade — the one door every workflow runs through.

``run_workflow(name, tenant_id, input)`` is the library API the future
orchestrator consumes (A2): registry lookup → input-schema validation →
compile (cached, once per process) → ``ainvoke`` → ``WorkflowResult``. This
module is the *only* place cross-graph wiring may live (ARCH Module
Boundaries — graphs never import each other). T3 delivered the run surface;
the approval chaining lives here too, row-based since the checkpointer
subsystem was deleted: a gated run persists a ``pending`` ``ApprovalRecord``
(Postgres) and returns ``pending_approval`` — the row IS the pause. The
guarded ``decide_approval`` resumes it (per-``request_id`` advisory lock →
tenant-scoped fetch → pending re-check → guarded UPDATE → on approval,
re-invoke the action-execution graph with ``approval_id`` set). No
LangGraph checkpointer, no ``interrupt()``, no thread-id discipline — the
same pause/resume shape as the orchestrator's ``PlanRecord``.

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

import logging
import time
import uuid
from typing import Any, Optional
from uuid import UUID

from langgraph.graph.state import CompiledStateGraph

from app.domain import approvals
from app.domain.approvals import ApprovalRecord
from app.domain.enums import ApprovalStatus
from app.graphs.base.state import WorkflowResult, WorkflowState, WorkflowStatus
from app.graphs.registry import get_workflow
from app.infrastructure.postgres import advisory_lock
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
# Approval chaining — the second facade slice (row-based pause/resume).
# --------------------------------------------------------------------------


class ApprovalNotFoundError(LookupError):
    """404-equivalent: unknown ``request_id`` — or another tenant's, which
    is deliberately indistinguishable (no tenant-existence oracle, R3)."""


class ApprovalAlreadyDecidedError(RuntimeError):
    """409-equivalent: the request is no longer ``pending`` (only
    ``pending → approved | rejected`` is valid, and the runtime + the
    guarded UPDATE — not the record model — enforce it)."""


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
            # Approval chaining: a gated run persists the pending
            # ``ApprovalRecord`` row — the pause. The caller still sees the
            # original pending_approval result (request_id unchanged); a
            # failure to persist the request is an infrastructure failure →
            # failed — a human must never be shown a request that does not
            # exist.
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


async def _pause_for_approval(
    tenant_id: str, run_id: str, result: WorkflowResult
) -> WorkflowResult:
    """Persist the pending ``ApprovalRecord`` for a gated run (the pause).

    Validates the gate's payload against ``ApprovalRequestInput``, then
    inserts the request row under the gate-minted id (id = request_id —
    one identifier names the request everywhere). Returns the caller's
    original ``pending_approval`` result unchanged. Raises on persistence
    failure — run_workflow maps that to ``failed``.
    """
    payload = approvals.ApprovalRequestInput.model_validate(
        {"request_id": result.request_id, **(result.output or {})}
    )
    try:
        record = await approvals.create_approval(
            ApprovalRecord(
                id=payload.request_id,
                tenant_id=tenant_id,
                action_name=payload.action_name,
                action_input=dict(payload.action_input),
                reason=payload.reason,
                risk_level=payload.risk_level,
                approver_role=payload.approver_role,
            )
        )
    except Exception as error:
        raise RuntimeError(
            "approval request could not be persisted (postgres unavailable)"
        ) from error
    logger.info(
        "approval request created (tenant=%s run_id=%s request_id=%s"
        " action=%s risk=%s)",
        tenant_id,
        run_id,
        record.id,
        record.action_name,
        record.risk_level,
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

    Per-``request_id`` guard → tenant-scoped row fetch → ``pending``
    re-check → guarded UPDATE (``mark_decided``: only a still-pending row
    transitions, so a concurrent double decision matches zero rows) → on
    approval re-invoke the action-execution graph with ``approval_id`` set
    (re-validate, gate skipped, execute, verify, record); a rejection is
    terminal — no re-invocation (R5). The guard is a cross-worker
    ``pg_advisory_lock("approval:<id>")`` holding one pool connection for
    the decision (the A17 turn-lock pattern); Postgres is required — the
    approval rows live there, so a pool-down boot fails fast here rather
    than degrading (the old process-local lock fallback guarded
    checkpointer state that no longer exists).

    The decision row is written BEFORE the action runs: the audit is the
    durable truth even if execution then fails (the outcome carries the
    failure; the record stays ``approved``).

    Raises (business class — the HTTP layer maps these):
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
    # Cross-worker guard: the whole decision runs under
    # pg_advisory_lock("approval:<id>") on one held pool connection (the
    # A17 turn-lock pattern), so same-request decisions serialize across
    # every worker sharing the database. The guard-plus-recheck is what
    # makes a concurrent double decision fail safe: the second caller sees
    # the already-decided row and raises before touching any state.
    try:
        async with advisory_lock(f"approval:{key}"):
            record = await approvals.get_approval(tenant_id, UUID(key))
            if record is None:
                raise ApprovalNotFoundError(
                    f"No pending approval {key!r} for this tenant."
                )
            if record.status != ApprovalStatus.PENDING:
                raise ApprovalAlreadyDecidedError(
                    f"Approval {key!r} is already {record.status.value}."
                )
            decided = await approvals.mark_decided(
                tenant_id,
                UUID(key),
                (
                    ApprovalStatus.APPROVED
                    if decision == "approved"
                    else ApprovalStatus.REJECTED
                ),
                decided_by,
                note,
            )
            if decided is None:
                # Decided outside the guard (direct DB edit) — the guarded
                # UPDATE lost the row. Belt-and-braces beside the re-check.
                raise ApprovalAlreadyDecidedError(
                    f"Approval {key!r} is already decided."
                )
            logger.info(
                "approval decided (tenant=%s request_id=%s decision=%s"
                " decided_by=%s)",
                tenant_id,
                decided.id,
                decided.status.value,
                decided_by,
            )
            if decision == "rejected":
                outcome = WorkflowResult(
                    status=WorkflowStatus.REJECTED,
                    request_id=key,
                    output={
                        "request_id": key,
                        "decision": decision,
                        "decided_by": decided_by,
                    },
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
        "workflow=approval.decide tenant=%s request_id=%s decision=%s"
        " status=%s duration_ms=%d",
        tenant_id,
        key,
        decision,
        outcome.status.value,
        duration_ms,
    )
    return outcome
