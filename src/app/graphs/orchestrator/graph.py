"""Orchestrator topology — slice 1 (CR graph shape, A1/A7).

    START → route ─┬─ direct ──────► direct_reply ──► compose ──► END
                   ├─ clarify ─────► clarify_reply ──► compose ──► END
                   └─ workflow ────► run_workflow ───► compose ──► END

``_after_route`` is the deterministic dispatch after classification:

- ``multi_workflow`` degrades to the direct path until slice 2 lands (ARCH
  rollout note — the planner/executor wiring is T9's slice of this file);
- a clarify classification under ``must_commit`` is overridden to the
  direct path (A7 — the clarify-cap guard is this edge, not prompt
  discipline);
- anything unexpected reads as direct.

Built by ``build_orchestrator_graph()``, invoked only by the conversation
service — never registered, never run through the facade (A1).
"""

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.graphs.orchestrator.composer import compose_reply
from app.graphs.orchestrator.paths import (
    clarify_reply_node,
    direct_reply_node,
    run_workflow_node,
)
from app.graphs.orchestrator.router import route
from app.graphs.orchestrator.state import OrchestratorState

_DISPATCH = {"direct", "clarify", "workflow"}


def _after_route(state: OrchestratorState) -> str:
    """Deterministic edge after classification (A7 + rollout note)."""
    classification = state.get("classification") or {}
    outcome = classification.get("outcome", "direct")
    if outcome == "multi_workflow":
        return "direct"  # slice-1 degradation; planner lands in T9
    if outcome == "clarify" and state.get("must_commit"):
        return "direct"  # A7: the clarify cap forces a commit
    return outcome if outcome in _DISPATCH else "direct"


def build_orchestrator_graph() -> CompiledStateGraph:
    """Wire and compile the slice-1 orchestrator graph."""
    builder = StateGraph(OrchestratorState)
    builder.add_node("route", route)
    builder.add_node("direct_reply", direct_reply_node)
    builder.add_node("clarify_reply", clarify_reply_node)
    builder.add_node("run_workflow", run_workflow_node)
    builder.add_node("compose", compose_reply)
    builder.add_edge(START, "route")
    builder.add_conditional_edges(
        "route",
        _after_route,
        {
            "direct": "direct_reply",
            "clarify": "clarify_reply",
            "workflow": "run_workflow",
        },
    )
    builder.add_edge("direct_reply", "compose")
    builder.add_edge("clarify_reply", "compose")
    builder.add_edge("run_workflow", "compose")
    builder.add_edge("compose", END)
    return builder.compile()
