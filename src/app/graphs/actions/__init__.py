"""Action-execution workflow (T6/R8) — self-registers as ``action_execution``.

One registration entry is this workflow's entire integration with the rest
of the system (A2, mirroring the other graph subpackages). Importing this
package also lands the mock action handlers: actions self-register exactly
like workflows, so ``ACTION_REGISTRY`` is populated the moment this package
is imported.
"""

from app.graphs.actions import handlers  # noqa: F401  (import = mock-action registration)
from app.graphs.actions.graph import build_action_execution_graph
from app.graphs.actions.nodes import ActionExecutionInput, ActionExecutionState
from app.graphs.registry import WorkflowDefinition, register

register(
    WorkflowDefinition(
        name="action_execution",
        description=(
            "Generic action execution — any registered ActionHandler"
            " (validate → approval gate → execute → verify), audited as a"
            " tenant-scoped ActionResultRecord"
        ),
        factory=build_action_execution_graph,
        input_schema=ActionExecutionInput,
    )
)

__all__ = ["ActionExecutionInput", "ActionExecutionState", "build_action_execution_graph"]
