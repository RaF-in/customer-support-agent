"""Immutable audit of every action execution (R3/R9)."""

from typing import Any, ClassVar, Optional
from uuid import UUID

from pydantic import Field

from app.domain.base.vector import VectorBaseDocument
from app.domain.enums import ActionStatus


class ActionResultRecord(VectorBaseDocument):
    """Outcome of one action execution — success, failure, or unverified
    (execute ran but ``verify()`` disagreed; the discrepancy stays visible
    for reconciliation). Written once by the action graph's
    ``record_result`` node; immutable. Payload-only point (dummy 1-dim
    vector, never similarity-searched).
    """

    action_name: str
    approval_id: Optional[UUID] = None  # back-reference when gated
    status: ActionStatus
    result: dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None

    collection_name: ClassVar[str] = "action_results"
