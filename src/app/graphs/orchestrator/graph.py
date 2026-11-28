"""Orchestrator topology — slices 1 + 2 (CR + PE graph shapes, A1/A7/A9).

    START ──(turn)──► route ─┬─ direct ──────► direct_reply ──► compose ──► END
      {message,              ├─ clarify ─────► clarify_reply ──► compose ──► END
       history_text,         ├─ workflow ────► run_workflow ───► compose ──► END
       catalog,              └─ multi_workf. ─► planner ─► validate ─invalid─┐
       must_commit}                                    ▲ (errors fed back)  │
                                                        └───────────────────┘
                                            valid ▼          exhausted → compose (apology)
                                              executor (todo per step · sequential ·
                                                $$STEP_X_OUTPUT$$ resolution · fail-stop ·
                                                approval → pause → compose)
                                                ▼
                                              verify ─open items─► executor (≤ EXECUTOR_MAX_REINVOCATIONS)
                                                │ all closed / takeover
                                                ▼
                                              compose_final ─► END

    START ──(resume)─► executor   (seeded from a persisted PlanRecord after an
                                   approval decision; the remainder runs and
                                   compose_final appends the final answer — A9)

Deterministic dispatch lives on the conditional edges, never in prompt
discipline: ``_start`` picks the resume entry when a seeded plan is present
(the service — the graph's only invoker — seeds it); ``_after_route`` sends
``multi_workflow`` to the planner (PE-R1) and forces a clarify under
``must_commit`` to the direct path (A7); ``_after_validate`` drives the
bounded planner retry cycle and its exhaustion branch; ``_after_executor``
routes an approval pause to the templated waiting reply; ``_after_verify``
re-invokes the executor on open todos within the bound, else hands the
takeover to ``compose_final``. Every cycle is counter-bounded in the plan
state (planner attempts, verify re-invocations) — PE-R13.

Built by ``build_orchestrator_graph()``, invoked only by the conversation
service — never registered, never run through the facade (A1).
"""

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.domain.enums import PlanStatus, TurnOutcome
from app.graphs.orchestrator.composer import compose_final, compose_reply
from app.graphs.orchestrator.executor import TodoList, executor_node, verify_node
from app.graphs.orchestrator.catalog import build_catalog
from app.graphs.orchestrator.paths import (
    clarify_reply_node,
    direct_reply_node,
    run_workflow_node,
)
from app.graphs.orchestrator.planner import (
    exhaustion_reply,
    generate_plan_text,
    validate_plan,
    PlanValidationError,
)
from app.graphs.orchestrator.router import route
from app.graphs.orchestrator.state import OrchestratorState
from app.infrastructure.settings import Settings

logger = logging.getLogger(__name__)

_DISPATCH = {"direct", "clarify", "workflow"}


def _start(state: OrchestratorState) -> str:
    """Turn entry vs. slice-2 resume entry: a seeded plan (steps present)
    resumes the executor from the persisted PlanRecord (A9); anything else
    is a fresh turn through the router."""
    if (state.get("plan") or {}).get("steps"):
        return "executor"
    return "route"


def _after_route(state: OrchestratorState) -> str:
    """Deterministic edge after classification (A7 + PE-R1)."""
    classification = state.get("classification") or {}
    outcome = classification.get("outcome", "direct")
    if outcome == "multi_workflow":
        return "planner"  # slice 2: the planner–executor path (PE-R1)
    if outcome == "clarify" and state.get("must_commit"):
        return "direct"  # A7: the clarify cap forces a commit
    return outcome if outcome in _DISPATCH else "direct"


async def planner_node(state: OrchestratorState) -> dict[str, Any]:
    """One planner attempt — the raw plan text (the validator owns parsing,
    so malformed output stays a handled class; planner.py primitives)."""
    plan = dict(state.get("plan") or {})
    plan["attempts"] = plan.get("attempts", 0) + 1
    plan["text"] = await generate_plan_text(
        state["message"],
        state.get("history_text"),
        state.get("catalog") or build_catalog(),
        feedback=plan.get("errors") or None,
        memories=state.get("memories"),
    )
    return {"plan": plan}


async def validate_node(state: OrchestratorState) -> dict[str, Any]:
    """Validate the attempt; on success seed the plan's working state (the
    shape the service persists as a ``PlanRecord`` — steps/todos/status),
    on invalidity feed the errors back, and on exhaustion produce the
    apologetic direct answer with **nothing executed** (PE-R4)."""
    plan = dict(state.get("plan") or {})
    try:
        validated = validate_plan(plan.get("text") or "")
    except PlanValidationError as error:
        plan["errors"] = error.errors
        if plan.get("attempts", 0) >= Settings().PLANNER_MAX_ATTEMPTS:
            plan["exhausted"] = True
            logger.info(
                "plan tenant=%s turn_id=%s attempts=%d valid=%s exhausted=1",
                state["tenant_id"],
                state["run_id"],
                plan["attempts"],
                False,
            )
            reply = await exhaustion_reply(
                state["message"],
                state.get("history_text"),
                memories=state.get("memories"),
            )
            return {
                "plan": plan,
                "reply": reply,
                "outcome": TurnOutcome.MULTI_WORKFLOW,
                "runs": [],
            }
        return {"plan": plan}
    seeded = {
        "message": state["message"],
        "status": PlanStatus.VALIDATED.value,
        "steps": [
            {
                "number": step["step"],
                "workflow": step["workflow"],
                "input": step["input"],
                "resolved_input": None,
                "run_id": None,
                "status": "pending",
                "output": None,
            }
            for step in validated
        ],
        "todos": [],
        "pending_request_id": None,
        "reinvocations": 0,
        "takeover": False,
    }
    logger.info(
        "plan tenant=%s turn_id=%s attempts=%d valid=%s steps=%d",
        state["tenant_id"],
        state["run_id"],
        plan.get("attempts", 0),
        True,
        len(validated),
    )
    return {"plan": seeded}


def _after_validate(state: OrchestratorState) -> str:
    plan = state.get("plan") or {}
    if plan.get("exhausted"):
        return "compose"  # apologetic direct answer, nothing ran (PE-R4)
    if plan.get("status") == PlanStatus.VALIDATED.value:
        return "executor"
    return "planner"  # bounded retry with the errors fed back (PE-R4)


def _after_executor(state: OrchestratorState) -> str:
    plan = state.get("plan") or {}
    if plan.get("status") == PlanStatus.PAUSED.value:
        return "compose"  # templated waiting reply (PE-R10 pause half)
    return "verify"


def _after_verify(state: OrchestratorState) -> str:
    plan = state.get("plan") or {}
    if plan.get("takeover"):
        return "compose_final"  # honest partial report (PE-R8)
    todos = TodoList.model_validate({"items": plan.get("todos") or []})
    if todos.is_complete():
        return "compose_final"  # system-verified completion (PE-R8)
    return "executor"  # bounded re-invocation (the node set the counter)


def build_orchestrator_graph() -> CompiledStateGraph:
    """Wire and compile the full orchestrator graph (slices 1 + 2)."""
    builder = StateGraph(OrchestratorState)
    builder.add_node("route", route)
    builder.add_node("direct_reply", direct_reply_node)
    builder.add_node("clarify_reply", clarify_reply_node)
    builder.add_node("run_workflow", run_workflow_node)
    builder.add_node("compose", compose_reply)
    builder.add_node("planner", planner_node)
    builder.add_node("validate", validate_node)
    builder.add_node("executor", executor_node)
    builder.add_node("verify", verify_node)
    builder.add_node("compose_final", compose_final)
    builder.add_conditional_edges(
        START, _start, {"route": "route", "executor": "executor"}
    )
    builder.add_conditional_edges(
        "route",
        _after_route,
        {
            "direct": "direct_reply",
            "clarify": "clarify_reply",
            "workflow": "run_workflow",
            "planner": "planner",  # multi_workflow classifications (PE-R1)
        },
    )
    builder.add_edge("planner", "validate")
    builder.add_conditional_edges(
        "validate",
        _after_validate,
        {"planner": "planner", "executor": "executor", "compose": "compose"},
    )
    builder.add_conditional_edges(
        "executor",
        _after_executor,
        {"verify": "verify", "compose": "compose"},
    )
    builder.add_conditional_edges(
        "verify",
        _after_verify,
        {"executor": "executor", "compose_final": "compose_final"},
    )
    builder.add_edge("direct_reply", "compose")
    builder.add_edge("clarify_reply", "compose")
    builder.add_edge("run_workflow", "compose")
    builder.add_edge("compose", END)
    builder.add_edge("compose_final", END)
    return builder.compile()
