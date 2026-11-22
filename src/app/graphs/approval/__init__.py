"""Approval package (T6/R5) — the policy engine.

The actions graph's approval gate consumes ``policies.py``. The former
``state.py`` + ``graph.py`` interrupt-based approval graph (create_request
→ resolve_approver → await_decision ⏸ interrupt → record_decision,
checkpointed under ``thread_id = request_id``) was deleted when approvals
consolidated onto the row-based pause: the runtime facade now persists the
``pending`` ``ApprovalRecord`` (Postgres) at the gate and resumes it with a
guarded UPDATE in ``decide_approval`` — the same pattern as the
orchestrator's ``PlanRecord``, with no LangGraph checkpointer anywhere.
The pending-approval payload contract lives beside the row model in
``domain/approvals.py`` (``ApprovalRequestInput``).
"""

from app.graphs.approval.policies import (
    PolicyOutcome,
    ThresholdRule,
    evaluate_approval_policy,
)

__all__ = ["PolicyOutcome", "ThresholdRule", "evaluate_approval_policy"]
