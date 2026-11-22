"""Tenant-scoped approval endpoints — the human gate's thin HTTP surface (T8/R11).

    GET  /tenants/{tenant_id}/approvals[?status=pending]
    POST /tenants/{tenant_id}/approvals/{request_id}/decision

This surface is deliberately thin (A4): it exists so the pause→resume flow
is exercisable end-to-end before any orchestrator. All semantics live
behind the runtime facade — ``decide_approval`` is this module's only
workflow dependency (ARCH Module Boundaries: ``api/`` touches the runtime
facade only); the module itself is parsing + error mapping, the same shape
as ``api/search.py``. On approve, the gated action executes synchronously
inside the facade call and its result fields ride back in the 200 body
(R11); a rejection is terminal — no re-invocation (R5). The queue itself is
one indexed Postgres query (``approval_requests_queue_idx``) — never a full
collection read.

``tenant_id`` comes from the URL path only (A3/A4); ``decided_by`` is
declared audit identity recorded on the decision, never enforcement (no
auth in MVP — ARCH Open Question). An unknown ``request_id`` and another
tenant's are indistinguishable — no tenant-existence oracle (R3): both are
the same 404 with a byte-identical detail, raised in one place.
"""

import logging
from datetime import datetime
from typing import Any, Literal, Optional

from fastapi import APIRouter, HTTPException, Path
from pydantic import BaseModel, Field

from app.application import conversations as conversation_service
from app.domain import approvals
from app.domain.enums import ActionStatus, ApprovalStatus
from app.graphs.base.state import WorkflowStatus
from app.graphs.runtime import (
    ApprovalAlreadyDecidedError,
    ApprovalNotFoundError,
    decide_approval,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tenants/{tenant_id}", tags=["approvals"])


class ApprovalSummary(BaseModel):
    """Queue item — what a human needs to decide, plus audit once decided."""

    request_id: str
    action_name: str
    action_input: dict[str, Any]  # the payload the action will run with
    reason: str
    risk_level: str
    approver_role: str
    status: ApprovalStatus
    decision_note: Optional[str] = None
    decided_by: Optional[str] = None
    decided_at: Optional[datetime] = None
    created_at: datetime


async def _tenant_approvals(
    tenant_id: str, status: Optional[ApprovalStatus]
) -> list["approvals.ApprovalRecord"]:
    """The tenant's approval requests — one indexed query (status filter
    and recency ordering pushed down to Postgres).

    Raises HTTPException 500 when the approval store is unavailable
    (infrastructure outage — the one case ARCH allows a 500 for).
    """
    try:
        return await approvals.list_approvals(tenant_id, status=status)
    except Exception:
        logger.exception(
            "approval queue read failed (tenant=%s)", tenant_id
        )
        raise HTTPException(
            status_code=500,
            detail="approval store unavailable: could not list approvals",
        ) from None


class DecisionRequest(BaseModel):
    """The human's decision. ``decided_by`` is declared audit identity
    recorded on the ``ApprovalRecord`` — not enforcement (no auth exists)."""

    decision: Literal["approved", "rejected"]
    decided_by: str = Field(min_length=1)
    note: Optional[str] = None


class DecisionResponse(BaseModel):
    """Recorded decision outcome. On approve, the action has already run
    synchronously (R11) and carries its ``ActionResultRecord`` fields:
    ``action_name`` / ``action_status`` / ``result`` / ``reason`` /
    ``record_id`` (None on reject — nothing executed)."""

    request_id: str
    decision: str
    decided_by: str
    note: Optional[str] = None
    action_name: Optional[str] = None
    action_status: Optional[ActionStatus] = None
    result: Optional[dict[str, Any]] = None
    reason: Optional[str] = None
    record_id: Optional[str] = None


@router.get("/approvals", response_model=list[ApprovalSummary])
async def list_approvals(
    tenant_id: str = Path(min_length=1), status: Optional[ApprovalStatus] = None
) -> list[ApprovalSummary]:
    """List this tenant's approval requests — the decision queue (R11).

    ``?status=pending`` narrows the view to what awaits a human; unfiltered,
    decided requests are included with their audit fields. Newest first
    (``created_at DESC`` — the queue index's ordering).
    """
    records = await _tenant_approvals(tenant_id, status)
    return [
        ApprovalSummary(
            request_id=str(record.id),
            action_name=record.action_name,
            action_input=record.action_input,
            reason=record.reason,
            risk_level=record.risk_level,
            approver_role=record.approver_role,
            status=record.status,
            decision_note=record.decision_note,
            decided_by=record.decided_by,
            decided_at=record.decided_at,
            created_at=record.created_at,
        )
        for record in records
    ]


@router.post(
    "/approvals/{request_id}/decision", response_model=DecisionResponse
)
async def decide(
    decision_request: DecisionRequest,
    tenant_id: str = Path(min_length=1),
    request_id: str = Path(min_length=1),
) -> DecisionResponse:
    """Record the human decision; on approve the gated action executes
    synchronously and its result rides back here (R11/R5).

    404 for an unknown ``request_id`` — or another tenant's (the same
    response, no oracle, R3); 409 when the request is no longer pending;
    500 for infrastructure failures. The decision row is written before the
    action runs, so a 500 here usually means execution failed after the
    decision was recorded — check the request's status via GET.
    """
    try:
        result = await decide_approval(
            tenant_id=tenant_id,
            request_id=request_id,
            decision=decision_request.decision,
            decided_by=decision_request.decided_by,
            note=decision_request.note,
        )
    except ApprovalNotFoundError:
        # Unknown and cross-tenant land here alike — one detail string,
        # byte-identical for both (R3).
        raise HTTPException(
            status_code=404, detail="approval request not found"
        ) from None
    except ApprovalAlreadyDecidedError:
        raise HTTPException(
            status_code=409, detail="approval request is no longer pending"
        ) from None
    if result.status == WorkflowStatus.FAILED:
        # Infrastructure class (the facade never leaks exceptions): with the
        # decision row written before execution, a failure here is most
        # often the action run — point at the queue for the recorded state.
        logger.error(
            "approval decision failed over HTTP (tenant=%s request_id=%s)",
            tenant_id,
            request_id,
        )
        raise HTTPException(
            status_code=500,
            detail=(
                "infrastructure failure while deciding — the decision, if "
                "reached, is recorded; check the request via GET"
            ),
        ) from None
    output = result.output if isinstance(result.output, dict) else {}
    # The CR-R10 conversational hand-off (A8): after a successful decision,
    # append the outcome to the linked conversation's transcript. Failure-
    # isolated by design — this can never change the decision response
    # above; the approval + action records remain the durable truth.
    try:
        await conversation_service.on_approval_decided(
            tenant_id=tenant_id, request_id=request_id, result=result
        )
    except Exception:
        logger.error(
            "approval outcome transcript append failed (tenant=%s request_id=%s)",
            tenant_id,
            request_id,
            exc_info=True,
        )
    return DecisionResponse(
        request_id=result.request_id or request_id,
        decision=decision_request.decision,
        decided_by=decision_request.decided_by,
        note=decision_request.note,
        action_name=output.get("action_name"),
        action_status=output.get("action_status"),
        result=output.get("result"),
        reason=output.get("reason"),
        record_id=output.get("record_id"),
    )
