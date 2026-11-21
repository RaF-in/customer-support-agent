"""Process-wide async Postgres pool + conversational schema bootstrap (A18).

The conversational subsystem's storage substrate: one
``psycopg_pool.AsyncConnectionPool`` at ``POSTGRES_DSN``, opened and entered
at app lifespan startup beside the Qdrant bootstrap and closed on shutdown
(``close_postgres()`` — the ``infrastructure/checkpoint.py`` enter/exit
pattern). ``bootstrap_schema()`` creates the three conversational tables
idempotently (``CREATE TABLE IF NOT EXISTS`` — additive zero-migration MVP
schema; no migration framework by decision A18).

Degrade-gracefully, exactly like the Qdrant bootstrap: Postgres unreachable
at boot ⇒ one warning logged here, ``open_postgres()`` returns False, the
pool stays closed (``get_postgres_pool()`` then raises — the conversational
surfaces map that to 500), and the rest of the app is unaffected. Recovery
is a restart with Postgres reachable — deliberate MVP simplicity.

Connections are autocommit with ``dict_row`` rows: the *only* transactions
are the explicit ``conn.transaction()`` blocks the domain accessors open —
most importantly the one-transaction turn write (seq bump + message
inserts commit together or not at all, ARCH A3). ``max_size`` comes from
``POSTGRES_POOL_MAX_SIZE`` — the concurrent-turns lever, because a
turn-spanning advisory lock holds one connection for the turn's duration
(A17).
"""

import logging
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

# Additive MVP schema (A18): header, append-only messages, mutable plans.
# The UNIQUE (conversation_id, seq) constraint is the no-gaps/no-collisions
# guarantee behind the turn transaction; the partial index is the CR-R10
# decision-time pending-request lookup.
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
            "bootstrap failed); conversational surfaces will return 500 "
            "until the app is restarted with Postgres reachable. Documents, "
            "search, and approvals are unaffected.",
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
