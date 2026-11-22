"""Shared graph state envelope + the runtime facade's universal return type.

Every workflow state extends ``WorkflowState``. The runtime (T3) injects
the envelope on every invocation — ``tenant_id`` comes only from the caller
and a fresh ``run_id`` is generated per run — so no graph or handler can
read tenant identity from any other source (structural isolation, R3).
"""

from enum import StrEnum
from typing import Any, Optional, TypedDict

from pydantic import BaseModel


class WorkflowStatus(StrEnum):
    """Terminal outcome of one workflow run — the only statuses the facade
    ever returns (business failures are statuses, not exceptions; R9)."""

    COMPLETED = "completed"
    PENDING_APPROVAL = "pending_approval"
    REJECTED = "rejected"
    FAILED = "failed"


class WorkflowState(TypedDict):
    """Envelope injected by the runtime into every graph invocation."""

    tenant_id: str
    run_id: str
    errors: list[str]


class WorkflowResult(BaseModel):
    """What ``run_workflow`` / ``decide_approval`` return — the single
    outcome type the future orchestrator consumes (R7/R9)."""

    status: WorkflowStatus
    output: Any = None
    request_id: Optional[str] = None  # present when status is pending_approval
    error: Optional[str] = None  # present when status is failed
