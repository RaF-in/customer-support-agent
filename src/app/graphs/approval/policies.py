"""Approval policy engine — pure functions, no graph, no IO (T6/R5/R8).

The actions graph's ``approval_gate`` node asks exactly one question here:
does this action, with these validated inputs, require human approval
before it may execute? Rules are declarative data (a per-action condition
table), so richer policies — two-person, sequential, time-boxed — extend
the engine by adding rule types and evaluator arms, never by editing graph
nodes (discussion §3.3).

Stance (developer decision 5, 2026-11-17): an action with NO policy entry
requires NO approval — the mock-e-commerce posture, to be revisited before
real tenants. The shipped reference policy is the threshold rule: a
``process_refund`` above 500 must be approved by a ``refund_manager``
(high risk).
"""

from dataclasses import dataclass
from typing import Any, Optional

from pydantic import BaseModel


class PolicyOutcome(BaseModel):
    """One policy evaluation's verdict.

    ``reason`` / ``risk_level`` / ``approver_role`` are populated when (and
    only when) approval is required — they ride the pending-approval payload
    into the ``ApprovalRecord`` row the runtime persists at the gate.
    """

    required: bool
    reason: Optional[str] = None
    risk_level: Optional[str] = None  # low / medium / high
    approver_role: Optional[str] = None


@dataclass(frozen=True)
class ThresholdRule:
    """Require approval when ``input_field`` exceeds ``threshold``.

    The threshold is the exclusive ceiling for auto-approval: a value equal
    to it passes free, one strictly above it gates (checklist boundary:
    500 free, 501 gated).
    """

    action_name: str
    input_field: str
    threshold: float
    risk_level: str
    approver_role: str


# The rule table — declarative, module-level, extensible in place. When an
# action has several rules, the FIRST matching one wins (evaluation order =
# table order per action).
RULES: dict[str, tuple[ThresholdRule, ...]] = {
    "process_refund": (
        ThresholdRule(
            action_name="process_refund",
            input_field="amount",
            threshold=500.0,
            risk_level="high",
            approver_role="refund_manager",
        ),
    ),
    # ``cancel_order`` is deliberately absent: no policy entry -> no approval
    # required (developer decision 5).
}


def evaluate_approval_policy(action_name: str, inputs: Any) -> PolicyOutcome:
    """Pure verdict for one (action, validated inputs) pair.

    ``inputs`` is the action's validated ``input_schema`` instance; rules
    read declared fields off it and mutate nothing.
    """
    rules = RULES.get(action_name, ())
    if not rules:
        return PolicyOutcome(required=False)
    for rule in rules:
        value = getattr(inputs, rule.input_field, None)
        if value is None:
            # The rule's field is absent from this action's schema — the
            # rule cannot apply.
            continue
        if value > rule.threshold:
            return PolicyOutcome(
                required=True,
                reason=(
                    f"{action_name}: {rule.input_field} {value} exceeds the"
                    f" auto-approval threshold of {rule.threshold}"
                ),
                risk_level=rule.risk_level,
                approver_role=rule.approver_role,
            )
    return PolicyOutcome(required=False)
