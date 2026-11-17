"""Action-execution graph wiring (T6/R8/R5/R9).

    START → resolve_handler → validate ──ineligible──→ record_result → END
                                      │eligible
                                      ▼
                               approval_gate ──required──→ END
                                      │not required   (pending_approval +
                                      ▼                full request payload)
                                   execute → verify → record_result → END

``validate`` and ``approval_gate`` are conditional forks; ``record_result``
is the single terminal for every executed or business-failed path; the
approval gate ends the run itself when a policy requires a human. T7
persists the ``ApprovalRecord`` and chains the two graphs through the
runtime — nothing in this graph knows about that (graphs never import each
other, A3).

The graph is acyclic, so no counter bounds apply (A12 governs cycles); the
process-wide ``recursion_limit`` backstop still wraps every run. Business
rules live in handlers and ``approval/policies.py`` — never in these nodes
(R8).
"""

import logging

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.graphs.actions import nodes
from app.graphs.actions.nodes import ActionExecutionState

logger = logging.getLogger(__name__)

# Conditional-edge targets (kept explicit for the mapping dicts below).
_GATE = "approval_gate"
_RECORD = "record_result"
_EXECUTE = "execute"


def route_after_validate(state: ActionExecutionState) -> str:
    """Ineligible actions go straight to their failed record (decision 6)."""
    return _GATE if state.get("eligible", True) else _RECORD


def route_after_approval_gate(state: ActionExecutionState) -> str:
    """A required approval ends the run at the gate; anything else executes."""
    outcome = state.get("policy_outcome")
    return END if outcome is not None and outcome.required else _EXECUTE


def build_action_execution_graph() -> CompiledStateGraph:
    """Wire and compile the action-execution graph (registry factory)."""
    builder = StateGraph(ActionExecutionState)

    builder.add_node("resolve_handler", nodes.resolve_handler_node)
    builder.add_node("validate", nodes.validate_action_node)
    builder.add_node("approval_gate", nodes.approval_gate_node)
    builder.add_node("execute", nodes.execute_action_node)
    builder.add_node("verify", nodes.verify_action_node)
    builder.add_node("record_result", nodes.record_result_node)

    builder.add_edge(START, "resolve_handler")
    builder.add_edge("resolve_handler", "validate")
    builder.add_conditional_edges(
        "validate",
        route_after_validate,
        {_GATE: "approval_gate", _RECORD: "record_result"},
    )
    builder.add_conditional_edges(
        "approval_gate",
        route_after_approval_gate,
        {_EXECUTE: "execute", END: END},
    )
    builder.add_edge("execute", "verify")
    builder.add_edge("verify", "record_result")
    builder.add_edge("record_result", END)

    return builder.compile()
