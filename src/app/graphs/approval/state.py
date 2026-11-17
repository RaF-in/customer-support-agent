"""Approval workflow state + input contract (T7/R5).

The approval graph is the system's first — and so far only — *checkpointed*
workflow: it pauses mid-run at ``await_decision`` via ``interrupt()`` and
continues when the runtime resumes the thread (``thread_id = request_id``).
Because its state is serialized into checkpoints, every value riding it
must stay serializer-friendly (str / dict — no live objects, unlike the
never-checkpointed action-graph state).
"""

from typing import Any, Optional, TypedDict
from uuid import UUID

from pydantic import BaseModel, Field

from app.graphs.base.state import WorkflowState


class ApprovalGraphInput(BaseModel):
    """Input contract for an approval-graph launch.

    Launched only by the runtime facade, never registry-run: a gated
    workflow's ``pending_approval`` payload — the action graph's gate
    output plus the request id it minted. ``tenant_id`` is deliberately
    absent (runtime-injected envelope, R3).
    """

    request_id: UUID  # == ApprovalRecord id == LangGraph thread_id
    action_name: str = Field(min_length=1)
    action_input: dict[str, Any] = Field(default_factory=dict)
    reason: str = Field(min_length=1)  # why approval is required (policy)
    risk_level: str  # low / medium / high (policy)
    approver_role: str  # role expected to decide (policy)


class ApprovalState(WorkflowState, total=False):
    """Envelope extension for the approval graph (checkpoint-serialized)."""

    # workflow input channels (ApprovalGraphInput, json-dumped)
    request_id: str
    action_name: str
    action_input: dict[str, Any]
    reason: str
    risk_level: str
    approver_role: str
    # create_request
    created_at: str  # ISO — the original creation time survives the pause
    # await_decision — the resume value interrupt() hands back
    decision: str  # "approved" | "rejected"
    decided_by: str
    decision_note: Optional[str]
    # record_decision — outcome keys the runtime maps. Deliberately never a
    # WorkflowStatus: "approved" is a graph-internal terminal, not a facade
    # status; the runtime interprets it and chains (A3).
    output: Any
