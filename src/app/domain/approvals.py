"""Operator-facing approval records — the human-gate queue entity (R3/R5)."""

from datetime import datetime
from typing import Any, ClassVar, Optional

from pydantic import model_validator

from app.domain.base.vector import VectorBaseDocument
from app.domain.enums import ApprovalStatus


class ApprovalRecord(VectorBaseDocument):
    """One approval request. Payload-only point (dummy 1-dim vector, never
    similarity-searched).

    ``id`` equals ``request_id`` and the LangGraph ``thread_id`` — a single
    identifier names the request everywhere (ARCH data model). Lifecycle:
    created ``pending`` by the approval graph's ``create_request`` node ->
    ``approved``/``rejected`` by a recorded decision. Transitions are
    enforced by the runtime (T7), not by this model.
    """

    action_name: str
    action_input: dict[str, Any]  # schema-validated payload the action runs with
    reason: str  # why approval is required (from the policy engine)
    risk_level: str  # low / medium / high (from the policy engine)
    approver_role: str  # role expected to decide
    status: ApprovalStatus = ApprovalStatus.PENDING
    decision_note: Optional[str] = None  # audit — set on decision
    decided_by: Optional[str] = None  # declared audit identity, not enforcement
    decided_at: Optional[datetime] = None  # audit — set on decision
    thread_id: Optional[str] = None  # resume key; mirrors id

    collection_name: ClassVar[str] = "approvals"

    @model_validator(mode="after")
    def _thread_id_mirrors_id(self) -> "ApprovalRecord":
        """The resume key always equals the record id (invariant held
        structurally — callers may not supply a divergent thread_id)."""
        if self.thread_id is None:
            self.thread_id = str(self.id)
        return self
