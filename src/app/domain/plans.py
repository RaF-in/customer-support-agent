"""Plan row models + SQL accessors (A9) — the slice-2 working state.

The ``conversation_plans`` row is the *mutable* working state of one
orchestrated compound request: pause/resume and the executor mutate it via
UPDATE (more natural than Qdrant's whole-point upsert), while the
transcript keeps the *immutable* audit snapshot (a ``plan``-kind message
appended at validation — that is ``domain/conversations.py``'s business).

Statuses: ``validated -> executing -> (paused <-> executing) -> done``,
terminal via failure-exhaustion or takeover; rejection of a step's approval
counts as that step failing (PE decision 8). Todo semantics are exactly
``specs/context.txt`` (statuses pending/in_progress/completed/abandoned;
abandonment carries a reason).

Same disciplines as ``domain/conversations.py``: structural tenant
isolation in every WHERE clause (cross-tenant reads as unknown, PE-N2),
public-swallow / private-raise accessor twins, process-wide psycopg pool.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Literal, Optional
from uuid import UUID, uuid4

from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field

from app.domain.enums import PlanStatus, TodoStatus
from app.infrastructure.postgres import get_postgres_pool

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


# Step status vocabulary (ARCH data model) — distinct from TodoStatus: a
# step that cannot run is *skipped*, its todo is *abandoned*.
StepStatus = Literal["pending", "completed", "failed", "skipped"]


class Todo(BaseModel):
    """One executor todo — semantics per ``specs/context.txt``."""

    id: int
    text: str
    status: TodoStatus = TodoStatus.PENDING
    abandon_reason: Optional[str] = None  # required when abandoning


class PlanStep(BaseModel):
    """One planned workflow run. ``input`` may carry ``$$STEP_X_OUTPUT$$``
    placeholders; ``resolved_input`` is what actually ran after resolution
    against earlier step outputs (both kept — the plan is the audit)."""

    number: int
    workflow: str
    input: dict[str, Any]
    resolved_input: Optional[dict[str, Any]] = None
    run_id: Optional[str] = None
    status: StepStatus = "pending"
    output: Any = None


class PlanRecord(BaseModel):
    """Row model for one ``conversation_plans`` row — the mutable working
    state the resume path re-seeds the graph from (deploy-safe pauses: the
    row survives redeploys, unlike checkpoint state)."""

    id: UUID = Field(default_factory=uuid4)
    tenant_id: str
    conversation_id: UUID
    status: PlanStatus = PlanStatus.VALIDATED
    steps: list[PlanStep] = Field(default_factory=list)
    todos: list[Todo] = Field(default_factory=list)
    pending_request_id: Optional[UUID] = None  # set while paused
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)


# ---------------------------------------------------------------------------
# Public API — swallow-and-log infrastructure failures (OVM twin style)
# ---------------------------------------------------------------------------


async def create_plan(
    tenant_id: str, conversation_id: UUID, steps: list[PlanStep]
) -> Optional[PlanRecord]:
    """Insert a validated plan's working record; None on failure."""
    try:
        return await _create_plan(tenant_id, conversation_id, steps)
    except Exception:
        logger.exception(
            "create_plan failed for conversation %s", conversation_id
        )
        return None


async def get_plan(tenant_id: str, plan_id: UUID) -> Optional[PlanRecord]:
    """Fetch one plan (unknown and cross-tenant ids both read as absent)."""
    try:
        return await _get_plan(tenant_id, plan_id)
    except Exception:
        logger.exception("get_plan failed for plan %s", plan_id)
        return None


async def get_plan_by_pending_request(
    tenant_id: str, pending_request_id: UUID
) -> Optional[PlanRecord]:
    """The PE-R10 resume lookup: the paused plan carrying this pending
    request id (partial-key query on the partial index)."""
    try:
        return await _get_plan_by_pending_request(tenant_id, pending_request_id)
    except Exception:
        logger.exception(
            "get_plan_by_pending_request failed for request %s",
            pending_request_id,
        )
        return None


async def update_plan(plan: PlanRecord) -> Optional[PlanRecord]:
    """Persist the mutable working state (status, steps, todos, pending
    request linkage) and return the updated row; None on failure."""
    try:
        return await _update_plan(plan)
    except Exception:
        logger.exception("update_plan failed for plan %s", plan.id)
        return None


# ---------------------------------------------------------------------------
# Private implementations — raise on failure
# ---------------------------------------------------------------------------


async def _create_plan(
    tenant_id: str, conversation_id: UUID, steps: list[PlanStep]
) -> PlanRecord:
    async with get_postgres_pool().connection() as conn:
        cursor = await conn.execute(
            "INSERT INTO conversation_plans "
            "(tenant_id, conversation_id, status, steps, todos) "
            "VALUES (%s, %s, %s, %s, '[]'::jsonb) RETURNING *",
            (
                tenant_id,
                conversation_id,
                PlanStatus.VALIDATED.value,
                Jsonb([step.model_dump(mode="json") for step in steps]),
            ),
        )
        row = await cursor.fetchone()
    assert row is not None
    return PlanRecord.model_validate(row)


async def _get_plan(tenant_id: str, plan_id: UUID) -> Optional[PlanRecord]:
    async with get_postgres_pool().connection() as conn:
        cursor = await conn.execute(
            "SELECT * FROM conversation_plans WHERE id = %s AND tenant_id = %s",
            (plan_id, tenant_id),
        )
        row = await cursor.fetchone()
    return PlanRecord.model_validate(row) if row else None


async def _get_plan_by_pending_request(
    tenant_id: str, pending_request_id: UUID
) -> Optional[PlanRecord]:
    async with get_postgres_pool().connection() as conn:
        cursor = await conn.execute(
            "SELECT * FROM conversation_plans "
            "WHERE tenant_id = %s AND pending_request_id = %s "
            "ORDER BY created_at LIMIT 1",
            (tenant_id, pending_request_id),
        )
        row = await cursor.fetchone()
    return PlanRecord.model_validate(row) if row else None


async def _update_plan(plan: PlanRecord) -> PlanRecord:
    async with get_postgres_pool().connection() as conn:
        cursor = await conn.execute(
            "UPDATE conversation_plans SET status = %s, steps = %s, "
            "todos = %s, pending_request_id = %s, updated_at = now() "
            "WHERE id = %s AND tenant_id = %s RETURNING *",
            (
                plan.status.value,
                Jsonb([step.model_dump(mode="json") for step in plan.steps]),
                Jsonb([todo.model_dump(mode="json") for todo in plan.todos]),
                plan.pending_request_id,
                plan.id,
                plan.tenant_id,
            ),
        )
        row = await cursor.fetchone()
    return PlanRecord.model_validate(row) if row else None
