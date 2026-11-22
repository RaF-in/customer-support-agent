"""Operator-facing approval records — the human-gate queue entity (R3/R5).

Postgres row model + accessors (the ``domain/plans.py`` pattern). The
pending row IS the pause state of a gated run — no LangGraph checkpointer,
no ``interrupt()``: the runtime inserts a ``pending`` row at the gate and
``decide_approval`` resumes by guarded UPDATE, exactly the row-based
pause/resume ``PlanRecord`` uses. (These records previously rode Qdrant as
payload-only points with a dummy 1-dim vector; they moved to Postgres when
it became a default boot dependency — one storage paradigm for row state.)

Accessor discipline: unlike ``plans.py``'s swallow-and-log public twins,
every accessor here RAISES on infrastructure failure. Each caller is its
own catch point by design — the runtime facade maps a raise to
``WorkflowResult(status=failed)`` (R9) and the HTTP layer to 500 — because
a human must never be shown an approval request that was not durably
persisted, and a decision that could not be recorded must never report
success.

Tenant isolation is structural: every WHERE clause carries ``tenant_id``;
an unknown id and another tenant's read identically as absent (no
tenant-existence oracle, R3).
"""

import logging
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID, uuid4

from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field

from app.domain.enums import ApprovalStatus
from app.infrastructure.postgres import get_postgres_pool

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ApprovalRequestInput(BaseModel):
    """The pending-approval payload contract between the action graph's
    gate and the runtime's request-row persistence.

    Produced by ``approval_gate_node`` (plus the request id it minted) and
    consumed by the runtime facade — ``tenant_id`` is deliberately absent
    (runtime-injected envelope, R3). Formerly ``ApprovalGraphInput``, the
    input contract of the deleted interrupt-based approval graph.
    """

    request_id: UUID  # == ApprovalRecord id (one identifier everywhere)
    action_name: str = Field(min_length=1)
    action_input: dict[str, Any] = Field(default_factory=dict)
    reason: str = Field(min_length=1)  # why approval is required (policy)
    risk_level: str  # low / medium / high (policy)
    approver_role: str  # role expected to decide (policy)


class ApprovalRecord(BaseModel):
    """Row model for one ``approval_requests`` row.

    ``id`` equals ``request_id`` (the gate mints it) — a single identifier
    names the request everywhere (ARCH data model). Lifecycle: inserted
    ``pending`` by the runtime when a gated run pauses -> ``approved`` /
    ``rejected`` by ``mark_decided`` under the decision guard. Transitions
    are enforced by the runtime + the guarded UPDATE, not by this model.
    """

    id: UUID = Field(default_factory=uuid4)
    tenant_id: str
    action_name: str
    action_input: dict[str, Any]  # schema-validated payload the action runs with
    reason: str  # why approval is required (from the policy engine)
    risk_level: str  # low / medium / high (from the policy engine)
    approver_role: str  # role expected to decide
    status: ApprovalStatus = ApprovalStatus.PENDING
    decision_note: Optional[str] = None  # audit — set on decision
    decided_by: Optional[str] = None  # declared audit identity, not enforcement
    decided_at: Optional[datetime] = None  # audit — set on decision
    created_at: datetime = Field(default_factory=_utc_now)


# ---------------------------------------------------------------------------
# Accessors — raising on infrastructure failure (see module docstring)
# ---------------------------------------------------------------------------


def _to_record(row: dict[str, Any]) -> ApprovalRecord:
    """Validate one ``SELECT *`` row (psycopg already decoded JSONB)."""
    return ApprovalRecord.model_validate(row)


async def create_approval(record: ApprovalRecord) -> ApprovalRecord:
    """Insert one pending request row; returns the stored row (DB ``now()``
    as ``created_at``). Raises on infrastructure failure."""
    async with get_postgres_pool().connection() as conn:
        cursor = await conn.execute(
            "INSERT INTO approval_requests "
            "(id, tenant_id, action_name, action_input, reason, risk_level,"
            " approver_role, status) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING *",
            (
                record.id,
                record.tenant_id,
                record.action_name,
                Jsonb(record.action_input),
                record.reason,
                record.risk_level,
                record.approver_role,
                record.status.value,
            ),
        )
        row = await cursor.fetchone()
    assert row is not None
    return _to_record(row)


async def get_approval(
    tenant_id: str, request_id: UUID
) -> Optional[ApprovalRecord]:
    """Fetch one request (unknown and cross-tenant ids both read as
    absent — no oracle, R3). Raises on infrastructure failure."""
    async with get_postgres_pool().connection() as conn:
        cursor = await conn.execute(
            "SELECT * FROM approval_requests "
            "WHERE id = %s AND tenant_id = %s",
            (request_id, tenant_id),
        )
        row = await cursor.fetchone()
    return _to_record(row) if row else None


async def mark_decided(
    tenant_id: str,
    request_id: UUID,
    status: ApprovalStatus,
    decided_by: str,
    decision_note: Optional[str] = None,
) -> Optional[ApprovalRecord]:
    """Record one human decision — the guarded resume.

    ``WHERE ... AND status = 'pending'`` makes the pending→decided
    transition atomic at the row: a second decision (or one that raced past
    the advisory lock via a direct DB edit) matches zero rows and returns
    ``None`` — the caller raises ``ApprovalAlreadyDecidedError``. The row
    keeps its original ``created_at``; only the audit fields change.
    Raises on infrastructure failure.
    """
    async with get_postgres_pool().connection() as conn:
        cursor = await conn.execute(
            "UPDATE approval_requests "
            "SET status = %s, decided_by = %s, decision_note = %s, "
            "decided_at = now() "
            "WHERE id = %s AND tenant_id = %s AND status = 'pending' "
            "RETURNING *",
            (status.value, decided_by, decision_note, request_id, tenant_id),
        )
        row = await cursor.fetchone()
    return _to_record(row) if row else None


async def list_approvals(
    tenant_id: str, status: Optional[ApprovalStatus] = None
) -> list[ApprovalRecord]:
    """The decision queue (R11): this tenant's requests, newest first,
    optionally narrowed by status — one indexed query
    (``approval_requests_queue_idx``), never a full-collection read.
    Raises on infrastructure failure."""
    query = "SELECT * FROM approval_requests WHERE tenant_id = %s"
    params: tuple[Any, ...] = (tenant_id,)
    if status is not None:
        query += " AND status = %s"
        params = (tenant_id, status.value)
    query += " ORDER BY created_at DESC, id DESC"
    async with get_postgres_pool().connection() as conn:
        cursor = await conn.execute(query, params)
        rows = await cursor.fetchall()
    return [_to_record(row) for row in rows]
