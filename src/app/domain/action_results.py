"""Immutable audit of every action execution (R3/R9).

Postgres row model + one raising accessor (the ``domain/approvals.py``
discipline: the caller — the action graph's ``record_result`` node — maps a
raise to a failed run, because a run whose durable outcome could not be
recorded must not report success). Formerly a Qdrant payload-only point
with a dummy 1-dim vector; moved to Postgres with the approvals queue.

Tenant isolation is the ``tenant_isolation`` row-level-security policy
(``infrastructure/postgres.py``): the connection comes from
``tenant_connection`` with the record's tenant, so Postgres itself rejects
a row written under another tenant.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID, uuid4

from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field

from app.domain.enums import ActionStatus
from app.infrastructure.postgres import tenant_connection

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ActionResultRecord(BaseModel):
    """Outcome of one action execution — success, failure, or unverified
    (execute ran but ``verify()`` disagreed; the discrepancy stays visible
    for reconciliation). Written once by the action graph's
    ``record_result`` node; immutable. ``approval_id`` back-references the
    satisfied request when the run was gated."""

    id: UUID = Field(default_factory=uuid4)
    tenant_id: str
    action_name: str
    approval_id: Optional[UUID] = None  # back-reference when gated
    status: ActionStatus
    result: dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=_utc_now)


async def record_action_result(record: ActionResultRecord) -> ActionResultRecord:
    """Append one immutable audit row; returns the stored row (DB ``now()``
    as ``created_at``). Raises on infrastructure failure."""
    async with tenant_connection(record.tenant_id) as conn:
        cursor = await conn.execute(
            "INSERT INTO action_results "
            "(id, tenant_id, action_name, approval_id, status, result, error) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING *",
            (
                record.id,
                record.tenant_id,
                record.action_name,
                record.approval_id,
                record.status.value,
                Jsonb(record.result),
                record.error,
            ),
        )
        row = await cursor.fetchone()
    assert row is not None
    return ActionResultRecord.model_validate(row)
