"""Node functions + state/input contracts for the action-execution graph
(T6/R8).

Orchestration only — eligibility rules live in handlers, approval
conditions in ``approval/policies.py``; nodes sequence, gate, persist, and
terminate. No LLM anywhere in this graph: actions are deterministic
business adapters, which is exactly the ARCH security boundary — an LLM may
*propose* an action, but only this graph (after any required human
approval, T7) ever *executes* one.

State + input contracts live here rather than a ``state.py`` slice: T6's
file footprint names none, nodes are the state's only consumers, and
``graph.py`` imports both from this module alongside the node functions
(no import cycles).

Failure classes (R9):
- Ineligible ``validate()`` is a *business* failure (developer decision 6):
  a terminal state carrying the handler's reason, an ``ActionResultRecord``
  with status ``failed`` written by ``record_result``, and ``execute()``
  never called. Like every business terminal it ends the run
  ``completed`` with the action outcome + reason in ``output`` — the same
  honest-terminal pattern as Self-RAG's ``no_answer_found`` (T4).
  ``WorkflowResult(status=failed)`` stays reserved for the infrastructure
  class (facade-caught exceptions).
- ``verify()`` returning False is likewise business: execute DID run, the
  record says ``failed`` with the unverified reason, and the discrepancy
  stays visible for reconciliation (ARCH forward-risk scenario).
- Infrastructure failures (handler crash, record persistence failure) raise
  — the runtime facade is the only catcher.
"""

import logging
import uuid
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from app.domain import action_results
from app.domain.action_results import ActionResultRecord
from app.domain.enums import ActionStatus
from app.graphs.actions.contracts import (
    ActionContext,
    ActionDefinition,
    ActionHandler,
    get_action,
)
from app.graphs.approval.policies import PolicyOutcome, evaluate_approval_policy
from app.graphs.base.state import WorkflowState, WorkflowStatus

logger = logging.getLogger(__name__)


class ActionExecutionInput(BaseModel):
    """Input contract for ``action_execution`` runs — any registered action,
    its payload, and (on approved re-invocation, T7) the gate's approval id.

    ``tenant_id`` is deliberately absent: the runtime injects it from the
    caller's argument (R3).

    The model validator resolves the action and validates ``inputs``
    against the action's own ``input_schema`` at the facade — so an unknown
    action or a payload violating its schema fails with a clear error
    *before the graph runs*, the same pre-run guarantee the workflow-level
    schema gives every workflow. Pydantic surfaces both as
    ``ValidationError`` carrying the fail-fast message (unknown actions
    list the known names).
    """

    action_name: str = Field(min_length=1)
    inputs: dict[str, Any] = Field(default_factory=dict)
    approval_id: Optional[UUID] = None  # set by the runtime on approved re-invocation (T7)

    @model_validator(mode="after")
    def _action_must_resolve_and_validate(self) -> "ActionExecutionInput":
        definition, _handler = get_action(self.action_name)
        definition.input_schema.model_validate(self.inputs)
        return self


class ActionExecutionState(WorkflowState, total=False):
    """Envelope extension for the action-execution graph.

    ``definition`` / ``handler`` / ``validated_inputs`` are live objects
    riding state in-process (the T5 tools precedent); this graph is never
    checkpointed, so nothing here needs to be serializable.
    """

    # workflow input channels
    action_name: str
    inputs: dict[str, Any]
    approval_id: Optional[UUID]
    # resolve_handler
    definition: ActionDefinition
    handler: ActionHandler
    validated_inputs: Any  # the action's input_schema instance
    # validate
    eligible: bool
    validation_reason: Optional[str]
    # approval_gate
    policy_outcome: PolicyOutcome
    # execute / verify
    execution_result: dict[str, Any]
    verified: bool
    # outcome keys — written by terminal nodes only (runtime contract)
    status: WorkflowStatus
    output: Any
    request_id: str


def _context(state: ActionExecutionState) -> ActionContext:
    """Handler call context — identity from the envelope only (R3)."""
    return ActionContext(
        tenant_id=state["tenant_id"],
        run_id=state["run_id"],
        approval_id=state.get("approval_id"),
    )


async def resolve_handler_node(state: ActionExecutionState) -> dict[str, Any]:
    """Resolve the registered action; keep its typed inputs in state.

    The facade already proved the pair resolves and the payload validates —
    this re-resolution builds the typed instance nodes hand to the handler
    (and keeps the graph safe when invoked off-facade).
    """
    definition, handler = get_action(state["action_name"])
    validated = definition.input_schema.model_validate(state.get("inputs", {}))
    return {"definition": definition, "handler": handler, "validated_inputs": validated}


async def validate_action_node(state: ActionExecutionState) -> dict[str, Any]:
    """Ask the handler whether the action may run (business eligibility)."""
    outcome = await state["handler"].validate(
        state["validated_inputs"], _context(state)
    )
    update: dict[str, Any] = {"eligible": outcome.eligible}
    if outcome.reason is not None:
        update["validation_reason"] = outcome.reason
    return update


async def approval_gate_node(state: ActionExecutionState) -> dict[str, Any]:
    """Policy verdict: gate the run, or let it through.

    Required → this node writes the pending-approval terminal state itself:
    a fresh ``request_id`` plus the full request payload (action, input,
    reason, risk level, approver role) that the runtime facade persists as
    a pending ``ApprovalRecord`` row — the pause.

    Runtime chaining: when the facade re-invokes this graph on an
    *approved* request (``approval_id`` set), a recorded human decision has
    already satisfied the policy — the gate does not re-evaluate and the run
    proceeds to execute (re-validation happened in ``validate`` above).
    This is the one sanctioned T6-file edit of the T7 slice (developer
    decision, 2026-11-17): the ``approval_id`` channel this check reads was
    declared here in T6 for exactly this purpose.
    """
    if state.get("approval_id") is not None:
        return {"policy_outcome": PolicyOutcome(required=False)}
    outcome = evaluate_approval_policy(state["action_name"], state["validated_inputs"])
    if not outcome.required:
        return {"policy_outcome": outcome}
    request_id = str(uuid.uuid4())
    logger.info(
        "action %s gated for approval (tenant=%s run_id=%s risk=%s)",
        state["action_name"],
        state["tenant_id"],
        state["run_id"],
        outcome.risk_level,
    )
    return {
        "policy_outcome": outcome,
        "status": WorkflowStatus.PENDING_APPROVAL,
        "request_id": request_id,
        "output": {
            "action_name": state["action_name"],
            "action_input": dict(state.get("inputs", {})),
            "reason": outcome.reason,
            "risk_level": outcome.risk_level,
            "approver_role": outcome.approver_role,
        },
    }


async def execute_action_node(state: ActionExecutionState) -> dict[str, Any]:
    """Run the action. Only reachable when eligible and ungated."""
    result = await state["handler"].execute(
        state["validated_inputs"], _context(state)
    )
    return {"execution_result": dict(result)}


async def verify_action_node(state: ActionExecutionState) -> dict[str, Any]:
    """Confirm the executed result (reconciliation check)."""
    verified = await state["handler"].verify(
        state["execution_result"], _context(state)
    )
    return {"verified": verified}


def _outcome(
    state: ActionExecutionState,
) -> tuple[ActionStatus, Optional[str], dict[str, Any]]:
    """Classify the terminal outcome from the state flags.

    Order matters: ineligibility (execute never ran) beats an unverified
    execution (it did) beats success.
    """
    if not state.get("eligible", True):
        reason = state.get("validation_reason") or "action ineligible"
        return ActionStatus.FAILED, reason, {}
    if not state.get("verified", True):
        return (
            ActionStatus.FAILED,
            "executed but verify() did not confirm the result",
            dict(state.get("execution_result", {})),
        )
    return (
        ActionStatus.SUCCEEDED,
        None,
        dict(state.get("execution_result", {})),
    )


async def record_result_node(state: ActionExecutionState) -> dict[str, Any]:
    """Terminal: write the immutable ``ActionResultRecord``, then end.

    The single writer of action audit records (ARCH data model). A failed
    persistence raises — the facade maps it to ``failed`` — because a run
    whose durable outcome could not be recorded must not report success.
    """
    action_status, reason, result_payload = _outcome(state)
    record = ActionResultRecord(
        tenant_id=state["tenant_id"],
        action_name=state["action_name"],
        approval_id=state.get("approval_id"),
        status=action_status,
        result=result_payload,
        error=reason,
    )
    try:
        record = await action_results.record_action_result(record)
    except Exception as error:
        raise RuntimeError(
            "action result could not be persisted (postgres unavailable)"
        ) from error
    logger.info(
        "action %s %s (tenant=%s run_id=%s record_id=%s)",
        state["action_name"],
        action_status.value,
        state["tenant_id"],
        state["run_id"],
        record.id,
    )
    return {
        "status": WorkflowStatus.COMPLETED,
        "output": {
            "action_name": state["action_name"],
            "action_status": action_status.value,
            "result": result_payload,
            "reason": reason,
            "record_id": str(record.id),
        },
    }
