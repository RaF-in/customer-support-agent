"""Long-term-memory fact-card row models + SQL accessors (LM v1).

One row is one durable fact on one user's card ("Pays with bKash"); the
set of ``status='active'`` rows for ``(tenant_id, user_id)`` *is* the card
— current truth — while superseded rows stay behind as the change audit
trail (append + supersede, never in-place edits; the PlanRecord/transcript
split pattern, LM A8). ``updated_at`` is the recency ordering key and is
bumped on supersede so the audit stays honest about when truth changed.

Retrieval is deliberately pure-recency in v1 (capped by
``MEMORY_MAX_FACTS``): a curated card is 1–5 KB, small enough that full
prompt inclusion beats approximate search; the v2 semantic upgrade swaps
this module's ``list_active_facts`` internals behind a measured tripwire
(LM-N3) — the contract stays row-shaped either way.

Same disciplines as ``domain/plans.py`` / ``domain/conversations.py``:
structural tenant isolation in every WHERE clause (a cross-tenant read is
byte-identical to an unknown one — no oracle, LM-R4), public-swallow /
private-raise accessor twins over the process-wide psycopg pool (the
application service is the single catch point), and no FK on ``user_id``
— MVP identity is caller-supplied strings, so the join to
``conversations.user_id`` is by value convention only.
"""

import logging
from datetime import datetime, timezone
from typing import Literal, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from app.infrastructure.postgres import get_postgres_pool

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


# Card vocabulary (LM data model): only these two values in v1 — a fact is
# current truth or superseded-by-a-newer-one; erasure is a hard DELETE.
FactStatus = Literal["active", "superseded"]


class MemoryFact(BaseModel):
    """Row model for one ``memory_facts`` row.

    ``superseded_by`` is a loose self-reference (no constraint — the audit
    trail is convention, same looseness as ``source_conversation_id``):
    set when a later fact replaces this one, pointing at the replacement.
    """

    id: UUID = Field(default_factory=uuid4)
    tenant_id: str
    user_id: str
    content: str
    status: FactStatus = "active"
    source_conversation_id: Optional[UUID] = None  # which exchange produced it
    superseded_by: Optional[UUID] = None  # set on supersede: the replacing fact
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)


# ---------------------------------------------------------------------------
# Public API — swallow-and-log infrastructure failures (OVM twin style)
# ---------------------------------------------------------------------------


async def add_fact(
    tenant_id: str,
    user_id: str,
    content: str,
    source_conversation_id: Optional[UUID] = None,
) -> Optional[MemoryFact]:
    """Append one active fact to the user's card; None on failure."""
    try:
        return await _add_fact(tenant_id, user_id, content, source_conversation_id)
    except Exception:
        logger.exception(
            "add_fact failed for tenant %s user %s", tenant_id, user_id
        )
        return None


async def list_active_facts(
    tenant_id: str, user_id: str, limit: int
) -> list[MemoryFact]:
    """The user's card — active facts in ``updated_at DESC`` recency order,
    capped by ``limit``. Empty list when the card is empty OR the read
    broke (degrade-to-empty is the read path's failure signal, LM-R5)."""
    try:
        return await _list_active_facts(tenant_id, user_id, limit)
    except Exception:
        logger.exception(
            "list_active_facts failed for tenant %s user %s", tenant_id, user_id
        )
        return []


async def supersede_fact(
    tenant_id: str, fact_id: UUID, superseded_by: Optional[UUID] = None
) -> Optional[MemoryFact]:
    """Flip one active fact to ``superseded`` (``superseded_by`` set,
    ``updated_at`` bumped) and return the updated row; None on failure or
    when the fact is unknown, already superseded, or another tenant's
    (all three read identically — no oracle)."""
    try:
        return await _supersede_fact(tenant_id, fact_id, superseded_by)
    except Exception:
        logger.exception(
            "supersede_fact failed for tenant %s fact %s", tenant_id, fact_id
        )
        return None


async def forget_user(tenant_id: str, user_id: str) -> bool:
    """GDPR erase: hard-delete every fact (active and superseded) for this
    ``(tenant, user)``. True iff rows were deleted — an immediate second
    call returns False (LM-R6)."""
    try:
        return await _forget_user(tenant_id, user_id)
    except Exception:
        logger.exception(
            "forget_user failed for tenant %s user %s", tenant_id, user_id
        )
        return False


# ---------------------------------------------------------------------------
# Private implementations — raise on failure
# ---------------------------------------------------------------------------


async def _add_fact(
    tenant_id: str,
    user_id: str,
    content: str,
    source_conversation_id: Optional[UUID],
) -> MemoryFact:
    async with get_postgres_pool().connection() as conn:
        cursor = await conn.execute(
            "INSERT INTO memory_facts "
            "(tenant_id, user_id, content, source_conversation_id) "
            "VALUES (%s, %s, %s, %s) RETURNING *",
            (tenant_id, user_id, content, source_conversation_id),
        )
        row = await cursor.fetchone()
    assert row is not None
    return MemoryFact.model_validate(row)


async def _list_active_facts(
    tenant_id: str, user_id: str, limit: int
) -> list[MemoryFact]:
    async with get_postgres_pool().connection() as conn:
        cursor = await conn.execute(
            "SELECT * FROM memory_facts "
            "WHERE tenant_id = %s AND user_id = %s AND status = 'active' "
            "ORDER BY updated_at DESC LIMIT %s",
            (tenant_id, user_id, limit),
        )
        rows = await cursor.fetchall()
    return [MemoryFact.model_validate(row) for row in rows]


async def _supersede_fact(
    tenant_id: str, fact_id: UUID, superseded_by: Optional[UUID]
) -> Optional[MemoryFact]:
    # Atomic by construction: the WHERE carries status='active', so a lost
    # race (or a second supersede) matches zero rows and reads as the no-op
    # None at the statement itself — belt-and-braces before the lock that
    # makes extraction races structurally impossible (A4/LM-N4).
    async with get_postgres_pool().connection() as conn:
        cursor = await conn.execute(
            "UPDATE memory_facts SET status = 'superseded', superseded_by = %s, "
            "updated_at = now() "
            "WHERE id = %s AND tenant_id = %s AND status = 'active' RETURNING *",
            (superseded_by, fact_id, tenant_id),
        )
        row = await cursor.fetchone()
    return MemoryFact.model_validate(row) if row else None


async def _forget_user(tenant_id: str, user_id: str) -> bool:
    async with get_postgres_pool().connection() as conn:
        cursor = await conn.execute(
            "DELETE FROM memory_facts WHERE tenant_id = %s AND user_id = %s",
            (tenant_id, user_id),
        )
        deleted = cursor.rowcount
    return deleted > 0
