"""Runtime facade (run surface) — the one door every workflow runs through.

``run_workflow(name, tenant_id, input)`` is the library API the future
orchestrator consumes (A2): registry lookup → input-schema validation →
compile (cached, once per process) → ``ainvoke`` → ``WorkflowResult``. This
module is the *only* place cross-graph wiring may live (ARCH Module
Boundaries — graphs never import each other). T3 delivers the run surface;
T7 extends this module with resume/decide/chaining and attaches the
checkpointer for the approval graph. No checkpointer is wired in this slice.

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
from typing import Any

from langgraph.graph.state import CompiledStateGraph

from app.graphs.base.state import WorkflowResult, WorkflowState, WorkflowStatus
from app.graphs.registry import get_workflow
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
