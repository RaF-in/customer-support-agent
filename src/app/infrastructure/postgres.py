"""Process-wide async Postgres pool + schema bootstrap (A18, extended).

The row-state substrate: one ``psycopg_pool.AsyncConnectionPool`` at
``POSTGRES_DSN``, opened and entered at app lifespan startup beside the
Qdrant bootstrap and closed on shutdown (``close_postgres()``).
``bootstrap_schema()`` creates every table idempotently (``CREATE TABLE IF
NOT EXISTS`` — additive zero-migration MVP schema; no migration framework
by decision A18): the three conversational tables (header, append-only
messages, mutable plans) plus the two human-gate tables (``approval_requests``
— the pause state of a gated run — and the immutable ``action_results``
audit), which moved here off Qdrant payload-only points when Postgres
became a default boot dependency.

Degrade-gracefully, exactly like the Qdrant bootstrap: Postgres unreachable
at boot ⇒ one warning logged here, ``open_postgres()`` returns False, the
pool stays closed (``get_postgres_pool()`` then raises — the conversational
and approval surfaces map that to 500), and the rest of the app is
unaffected. Recovery is a restart with Postgres reachable — deliberate MVP
simplicity.

Connections are autocommit with ``dict_row`` rows: the *only* transactions
are the explicit ``conn.transaction()`` blocks the domain accessors open —
most importantly the one-transaction turn write (seq bump + message
inserts commit together or not at all, ARCH A3). ``max_size`` comes from
``POSTGRES_POOL_MAX_SIZE`` — the concurrent-turns lever, because a
turn-spanning advisory lock holds one connection for the turn's duration
(A17).

``advisory_lock(key)`` is the generic cross-worker serialization primitive
built on that pool: conversation turns key it on the conversation id, the
approval decision guard on ``approval:<request_id>``.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Optional

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.infrastructure.settings import Settings

logger = logging.getLogger(__name__)

# Process-wide pool singleton — None until a successful open, and again
# after close. `get_postgres_pool()` is the single handle the domain
# accessors use; nothing else constructs connections.
_pool: Optional[AsyncConnectionPool] = None

# Seconds to wait for the first pool connection at boot — past this,
# Postgres reads as unreachable and the app degrades gracefully instead of
# hanging startup.
_OPEN_TIMEOUT_SECONDS = 5.0

# Additive MVP schema (A18): header, append-only messages, mutable plans,
# then the human-gate pair — ``approval_requests`` (the pending row IS the
# pause of a gated run; the queue index serves the decision queue's
# ``?status=pending`` view directly) and ``action_results`` (immutable
# execution audit, written once). The UNIQUE (conversation_id, seq)
# constraint is the no-gaps/no-collisions guarantee behind the turn
# transaction; the partial indexes are the decision-time pending-request
# lookups (CR-R10 / PE-R10).
_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS conversations (
        id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        tenant_id  TEXT NOT NULL,
        last_seq   INTEGER NOT NULL DEFAULT 0,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS conversation_messages (
        id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        tenant_id          TEXT NOT NULL,
        conversation_id    UUID NOT NULL REFERENCES conversations (id),
        seq                INTEGER NOT NULL,
        role               TEXT NOT NULL,
        kind               TEXT NOT NULL,
        content            TEXT NOT NULL,
        data               JSONB,
        pending_request_id UUID,
        created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
        UNIQUE (conversation_id, seq)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS conversation_messages_pending_request_idx
        ON conversation_messages (tenant_id, pending_request_id)
        WHERE pending_request_id IS NOT NULL
    """,
    """
    CREATE TABLE IF NOT EXISTS conversation_plans (
        id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        tenant_id          TEXT NOT NULL,
        conversation_id    UUID NOT NULL,
        status             TEXT NOT NULL,
        steps              JSONB NOT NULL DEFAULT '[]'::jsonb,
        todos              JSONB NOT NULL DEFAULT '[]'::jsonb,
        pending_request_id UUID,
        created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at         TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS conversation_plans_pending_request_idx
        ON conversation_plans (tenant_id, pending_request_id)
        WHERE pending_request_id IS NOT NULL
    """,
    """
    CREATE TABLE IF NOT EXISTS approval_requests (
        id            UUID PRIMARY KEY,
        tenant_id     TEXT NOT NULL,
        action_name   TEXT NOT NULL,
        action_input  JSONB NOT NULL,
        reason        TEXT NOT NULL,
        risk_level    TEXT NOT NULL,
        approver_role TEXT NOT NULL,
        status        TEXT NOT NULL DEFAULT 'pending',
        decided_by    TEXT,
        decision_note TEXT,
        decided_at    TIMESTAMPTZ,
        created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    # The decision queue: GET /approvals[?status=pending] is served by this
    # index (tenant + status + recency), never a full-table read.
    """
    CREATE INDEX IF NOT EXISTS approval_requests_queue_idx
        ON approval_requests (tenant_id, status, created_at)
    """,
    """
    CREATE TABLE IF NOT EXISTS action_results (
        id          UUID PRIMARY KEY,
        tenant_id   TEXT NOT NULL,
        action_name TEXT NOT NULL,
        approval_id UUID,
        status      TEXT NOT NULL,
        result      JSONB NOT NULL DEFAULT '{}'::jsonb,
        error       TEXT,
        created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS action_results_approval_idx
        ON action_results (tenant_id, approval_id)
    """,
    # Long-term memory v1 (LM ARCH): the per-user fact card. All additive
    # (A18) — the ALTER leaves pre-existing conversations with NULL
    # ``user_id`` (anonymous forever, by design; no backfill), and the
    # card index serves the turn-start recency read directly.
    """
    ALTER TABLE conversations ADD COLUMN IF NOT EXISTS user_id TEXT
    """,
    """
    CREATE TABLE IF NOT EXISTS memory_facts (
        id                     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        tenant_id              TEXT NOT NULL,
        user_id                TEXT NOT NULL,
        content                TEXT NOT NULL,
        status                 TEXT NOT NULL DEFAULT 'active',
        source_conversation_id UUID,
        superseded_by          UUID,
        created_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at             TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    # The card read path: (tenant, user) active facts by recency —
    # index-bounded at any table size (LM-N3).
    """
    CREATE INDEX IF NOT EXISTS memory_facts_card_idx
        ON memory_facts (tenant_id, user_id, status, updated_at DESC)
    """,
)


def get_postgres_pool() -> AsyncConnectionPool:
    """Return the process-wide pool.

    Raises RuntimeError when the pool is not open — unreachable at boot, or
    already closed at shutdown. The conversation service maps that to the
    single user-meaningful 500 (infrastructure class; CR-N1).
    """
    if _pool is None:
        raise RuntimeError(
            "Postgres pool is not open (unreachable at boot, or the app is "
            "shutting down); conversational surfaces are unavailable."
        )
    return _pool


@asynccontextmanager
async def advisory_lock(key: str) -> AsyncIterator[None]:
    """Session-scoped Postgres advisory lock under one pool connection.

    Holds ONE connection for the whole block with
    ``pg_advisory_lock(hashtext(<key>))`` taken on it: callers keyed on the
    same string serialize at any worker count (every worker shares the one
    Postgres lock manager), and the lock releases at block exit — or
    automatically at session death, so a crashed process can never strand a
    key. The registry is Postgres-internal memory (visible through the
    ``pg_locks`` system view), not a table — nothing persists and nothing
    needs cleanup. Callers must namespace their keys (``approval:<id>``,
    a bare conversation id) so independent guards never collide.

    Raises on infrastructure failure (pool closed, or the lock statement
    itself broke) — callers decide what that means; a lock held by another
    caller blocks here server-side, which is the serialization working,
    not an outage.
    """
    pool = get_postgres_pool()
    async with pool.connection() as conn:
        await conn.execute("SELECT pg_advisory_lock(hashtext(%s))", (key,))
        try:
            yield
        finally:
            try:
                await conn.execute(
                    "SELECT pg_advisory_unlock(hashtext(%s))", (key,)
                )
            except Exception:
                # The connection is about to return to the pool (or die);
                # a session close releases the lock anyway (A17).
                logger.warning(
                    "advisory unlock failed for key %r; the session close "
                    "releases it",
                    key,
                    exc_info=True,
                )


async def _bootstrap_schema(pool: AsyncConnectionPool) -> None:
    """Run every schema statement (idempotent; raises on failure)."""
    async with pool.connection() as conn:
        async with conn.transaction():
            for statement in _SCHEMA_STATEMENTS:
                await conn.execute(statement)


async def bootstrap_schema() -> None:
    """(Re)run the idempotent schema bootstrap on the open pool.

    Raises on failure (infrastructure class) — safe to call on every boot:
    existing tables and indexes are left untouched.
    """
    await _bootstrap_schema(get_postgres_pool())


async def open_postgres(settings: Optional[Settings] = None) -> bool:
    """Open the pool and bootstrap the schema (lifespan startup hook).

    Degrade-gracefully: any failure here (endpoint unreachable, auth, schema
    bootstrap) logs exactly one warning and returns False — the app still
    starts; only the conversational surfaces are down. Returns True when the
    pool is ready. Calling again after a failed open retries cleanly.
    """
    global _pool
    if _pool is not None:
        return True
    settings = settings or Settings()
    pool: Optional[AsyncConnectionPool] = None
    try:
        pool = AsyncConnectionPool(
            settings.POSTGRES_DSN,
            min_size=1,
            max_size=settings.POSTGRES_POOL_MAX_SIZE,
            open=False,
            # Acquire timeout: a starved caller (every connection held by a
            # long turn) fails in seconds, not the 30s default.
            timeout=_OPEN_TIMEOUT_SECONDS,
            # Extra connect kwargs: autocommit + dict rows, so the explicit
            # transaction blocks in the accessors are the only transactions.
            kwargs={"autocommit": True, "row_factory": dict_row},
        )
        # wait=True + short timeout is the boot-time reachability probe:
        # the first connection is ready, or the open fails fast.
        await pool.open(wait=True, timeout=_OPEN_TIMEOUT_SECONDS)
        await _bootstrap_schema(pool)
    except Exception:
        logger.warning(
            "Postgres unavailable at boot (endpoint unreachable or schema "
            "bootstrap failed); conversational and approval surfaces will "
            "return 500 until the app is restarted with Postgres reachable. "
            "Documents and search are unaffected.",
            exc_info=True,
        )
        if pool is not None:
            try:
                await pool.close()
            except Exception:  # pragma: no cover - shutdown best effort
                logger.debug("Pool cleanup after failed open raised", exc_info=True)
        return False
    _pool = pool
    logger.info(
        "Postgres pool ready (max_size=%d)", settings.POSTGRES_POOL_MAX_SIZE
    )
    return True


async def close_postgres() -> None:
    """Close the process-wide pool (shutdown hook; safe if unused or closed).

    Session-scoped advisory locks die with their connections, so closing
    the pool also releases any lock a crashed turn was holding — a
    conversation can never stay stranded (A17).
    """
    global _pool
    if _pool is not None:
        await _pool.close()
    _pool = None
