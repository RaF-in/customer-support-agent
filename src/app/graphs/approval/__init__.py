"""Approval workflow package (T7/R5).

Created in T6 with its policy engine only: the actions graph's approval
gate consumes ``policies.py``, so the package and the engine land with T6
(declared in the task's footprint). The graph itself (``state.py`` +
``graph.py``: create_request → resolve_approver → await_decision ⏸
interrupt → record_decision) is T7's slice; until then this package
registers nothing workflow-shaped.
"""

from app.graphs.approval.policies import (
    PolicyOutcome,
    ThresholdRule,
    evaluate_approval_policy,
)

__all__ = ["PolicyOutcome", "ThresholdRule", "evaluate_approval_policy"]
