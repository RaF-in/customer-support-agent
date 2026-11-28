"""Conversational transcript row models + SQL accessors (A3).

Two tables owned here: the ``conversations`` header (server-issued identity,
transcript sequence source, O(1) no-oracle existence check) and the
append-only ``conversation_messages`` rows (every transcript-visible entry —
user messages, assistant replies, orchestration artifacts — in one table,
one sort, one read path).

Structural tenant isolation lives in the WHERE clauses by construction:
every statement carries ``tenant_id``, so a cross-tenant id returns no row —
reading exactly like an unknown one (no tenant-existence oracle,
CR-R12/N3). Postgres enforces the same filter a second time through the
``tenant_isolation`` row-level-security policy
(``infrastructure/postgres.py``): every accessor takes its connection via
``tenant_connection``, so a statement without the WHERE clause still cannot
cross tenants. Accessor style mirrors the OVM's public-swallow /
private-raise twins, expressed in SQL over the process-wide psycopg pool:
public functions catch and log infrastructure failures and return ``None``
(a fetch finding nothing and a fetch that broke are indistinguishable here;
the conversation service — the single catch point — decides what that
means). The turn write is ONE transaction: seq bump and message inserts
commit together or not at all (ARCH A3 — no partially-visible turns).
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID, uuid4

from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field

from app.domain.enums import MessageKind, MessageRole
from app.infrastructure.postgres import advisory_lock, tenant_connection

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ConversationRecord(BaseModel):
    """Row model for one ``conversations`` header.

    Created implicitly by the first message (CR decision 5); append-only
    thereafter. ``last_seq`` is the monotonic transcript sequence — bumped
    only inside the turn transaction, so there are never gaps or collisions.
    """

    id: UUID = Field(default_factory=uuid4)
    tenant_id: str
    # The long-term-memory owner, bound once at creation from the
    # X-User-Id header (LM A2). NULL = anonymous — no fact card, behavior
    # identical to today; pre-existing rows stay NULL forever (no backfill).
    user_id: Optional[str] = None
    last_seq: int = 0
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)


class MessageDraft(BaseModel):
    """One message to append, before the turn transaction assigns its seq."""

    role: MessageRole
    kind: MessageKind = MessageKind.TEXT
    content: str
    data: Optional[dict[str, Any]] = None
    pending_request_id: Optional[UUID] = None


class ConversationMessageRecord(BaseModel):
    """Row model for one ``conversation_messages`` row (written once,
    immutable). ``pending_request_id`` is set on the assistant reply of a
    pending-approval turn — the CR-R10/PE-R10 decision-time linkage."""

    id: UUID = Field(default_factory=uuid4)
    tenant_id: str
    conversation_id: UUID
    seq: int
    role: MessageRole
    kind: MessageKind
    content: str
    data: Optional[dict[str, Any]] = None
    pending_request_id: Optional[UUID] = None
    created_at: datetime = Field(default_factory=_utc_now)


# ---------------------------------------------------------------------------
# Public API — swallow-and-log infrastructure failures (OVM twin style)
# ---------------------------------------------------------------------------


async def create_conversation(
    tenant_id: str, user_id: Optional[str] = None
) -> Optional[ConversationRecord]:
    """Insert a conversation header; returns None on failure."""
    try:
        return await _create_conversation(tenant_id, user_id)
    except Exception:
        logger.exception("create_conversation failed for tenant %s", tenant_id)
        return None


async def get_conversation(
    tenant_id: str, conversation_id: UUID
) -> Optional[ConversationRecord]:
    """Fetch one header (unknown and cross-tenant ids both read as absent)."""
    try:
        return await _get_conversation(tenant_id, conversation_id)
    except Exception:
        logger.exception(
            "get_conversation failed for conversation %s", conversation_id
        )
        return None


async def append_messages(
    tenant_id: str,
    conversation_id: UUID,
    drafts: list[MessageDraft],
) -> Optional[list[ConversationMessageRecord]]:
    """Append a turn's messages in ONE transaction.

    ``UPDATE conversations SET last_seq = last_seq + N … RETURNING`` plus the
    N message inserts commit together or not at at all — a failure anywhere
    (including between the bump and the inserts) rolls the whole turn back
    and ``last_seq`` is unchanged. Seqs are assigned here, in write order:
    base+1 … base+N. Returns the created rows (DB ids and timestamps), or
    None on failure — nothing partial is ever visible.
    """
    try:
        return await _append_messages(tenant_id, conversation_id, drafts)
    except Exception:
        logger.exception(
            "append_messages failed for conversation %s (%d messages)",
            conversation_id,
            len(drafts),
        )
        return None


async def get_transcript(
    tenant_id: str, conversation_id: UUID
) -> Optional[list[ConversationMessageRecord]]:
    """Read one conversation's transcript, chronological by ``seq``."""
    try:
        return await _get_transcript(tenant_id, conversation_id)
    except Exception:
        logger.exception(
            "get_transcript failed for conversation %s", conversation_id
        )
        return None


async def find_message_by_pending_request(
    tenant_id: str, pending_request_id: UUID
) -> Optional[ConversationMessageRecord]:
    """The CR-R10 linkage lookup: the assistant reply carrying this pending
    request id (partial-key query on the partial index)."""
    try:
        return await _find_message_by_pending_request(tenant_id, pending_request_id)
    except Exception:
        logger.exception(
            "find_message_by_pending_request failed for request %s",
            pending_request_id,
        )
        return None


@asynccontextmanager
async def conversation_turn_lock(conversation_id: UUID) -> AsyncIterator[None]:
    """Turn-spanning advisory lock (A17) — the T5 declared additive accessor.

    Delegates to ``infrastructure.postgres.advisory_lock`` keyed on the
    conversation id: same-conversation turns serialize at any worker count
    (CR-N2), and the lock releases at block exit — or automatically at
    session death, so a crashed process can never strand a conversation.
    Unlike the record accessors this primitive *raises* on infrastructure
    failure (the conversation service, the single catch point, maps that
    to its one user-meaningful error); a lock held by another turn blocks
    here server-side — that wait is the serialization working, not an
    outage.
    """
    async with advisory_lock(str(conversation_id)):
        yield


# ---------------------------------------------------------------------------
# Private implementations — raise on failure
# ---------------------------------------------------------------------------

_MESSAGE_COLUMNS = (
    "id, tenant_id, conversation_id, seq, role, kind, content, data, "
    "pending_request_id, created_at"
)


async def _create_conversation(
    tenant_id: str, user_id: Optional[str] = None
) -> ConversationRecord:
    async with tenant_connection(tenant_id) as conn:
        cursor = await conn.execute(
            "INSERT INTO conversations (tenant_id, user_id) "
            "VALUES (%s, %s) RETURNING *",
            (tenant_id, user_id),
        )
        row = await cursor.fetchone()
    assert row is not None
    return ConversationRecord.model_validate(row)


async def _get_conversation(
    tenant_id: str, conversation_id: UUID
) -> Optional[ConversationRecord]:
    async with tenant_connection(tenant_id) as conn:
        cursor = await conn.execute(
            "SELECT * FROM conversations WHERE id = %s AND tenant_id = %s",
            (conversation_id, tenant_id),
        )
        row = await cursor.fetchone()
    return ConversationRecord.model_validate(row) if row else None


async def _append_messages(
    tenant_id: str,
    conversation_id: UUID,
    drafts: list[MessageDraft],
) -> list[ConversationMessageRecord]:
    async with tenant_connection(tenant_id) as conn:
        async with conn.transaction():
            cursor = await conn.execute(
                "UPDATE conversations SET last_seq = last_seq + %s, "
                "updated_at = now() WHERE id = %s AND tenant_id = %s "
                "RETURNING last_seq",
                (len(drafts), conversation_id, tenant_id),
            )
            header = await cursor.fetchone()
            if header is None:
                # Unknown or cross-tenant conversation — the service checks
                # the header before appending, so this is a guard, not a
                # control-flow path.
                raise LookupError(
                    f"Conversation {conversation_id} not found for this tenant."
                )
            base_seq = header["last_seq"] - len(drafts)
            return await _insert_message_rows(
                conn, tenant_id, conversation_id, drafts, base_seq
            )


async def _insert_message_rows(
    conn: Any,
    tenant_id: str,
    conversation_id: UUID,
    drafts: list[MessageDraft],
    base_seq: int,
) -> list[ConversationMessageRecord]:
    """Insert the turn's messages at seq base+1…base+N (patch-separable so
    verification can induce a mid-transaction failure)."""
    records: list[ConversationMessageRecord] = []
    for offset, draft in enumerate(drafts, start=1):
        cursor = await conn.execute(
            "INSERT INTO conversation_messages "
            "(tenant_id, conversation_id, seq, role, kind, content, data, "
            "pending_request_id) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
            f"RETURNING {_MESSAGE_COLUMNS}",
            (
                tenant_id,
                conversation_id,
                base_seq + offset,
                draft.role.value,
                draft.kind.value,
                draft.content,
                Jsonb(draft.data) if draft.data is not None else None,
                draft.pending_request_id,
            ),
        )
        row = await cursor.fetchone()
        assert row is not None
        records.append(ConversationMessageRecord.model_validate(row))
    return records


async def _get_transcript(
    tenant_id: str, conversation_id: UUID
) -> list[ConversationMessageRecord]:
    async with tenant_connection(tenant_id) as conn:
        cursor = await conn.execute(
            "SELECT " + _MESSAGE_COLUMNS + " FROM conversation_messages "
            "WHERE tenant_id = %s AND conversation_id = %s ORDER BY seq",
            (tenant_id, conversation_id),
        )
        rows = await cursor.fetchall()
    return [ConversationMessageRecord.model_validate(row) for row in rows]


async def _find_message_by_pending_request(
    tenant_id: str, pending_request_id: UUID
) -> Optional[ConversationMessageRecord]:
    async with tenant_connection(tenant_id) as conn:
        cursor = await conn.execute(
            "SELECT " + _MESSAGE_COLUMNS + " FROM conversation_messages "
            "WHERE tenant_id = %s AND pending_request_id = %s "
            "ORDER BY seq LIMIT 1",
            (tenant_id, pending_request_id),
        )
        row = await cursor.fetchone()
    return ConversationMessageRecord.model_validate(row) if row else None
