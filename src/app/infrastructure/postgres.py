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
pool stays closed, and the rest of the app is unaffected. Recovery needs no
restart: ``get_postgres_pool()`` — now a coroutine — re-attempts
``open_postgres()`` whenever it finds the pool closed (spaced by a short
cooldown so an outage costs one probe per window, not one per request), and
raises only while Postgres is still unreachable — the conversational and
approval surfaces map that to 500.

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

Tenant isolation is enforced twice. The domain accessors still write
``tenant_id`` into every WHERE clause, and every tenant table also carries
the ``tenant_isolation`` row-level-security policy: Postgres itself drops
any row whose ``tenant_id`` differs from the ``app.tenant_id`` session
setting, so a statement that forgets the WHERE clause reads and writes
nothing outside its tenant. ``tenant_connection(tenant_id)`` is how an
accessor takes a connection for tenant-table SQL — it writes that setting on
checkout and clears it on return.

Two roles. The pool connects as the ordinary role in ``POSTGRES_DSN``
(default ``app_rw``), which is subject to the policies; the schema bootstrap
and grants run over one short-lived connection as the owner in
``POSTGRES_ADMIN_DSN`` (default ``app``). ``open_postgres`` refuses a
``POSTGRES_DSN`` role that would bypass the policies (superuser, BYPASSRLS,
or table owner) and degrades as below rather than serve unisolated rows.
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Optional

from psycopg import AsyncConnection, sql
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.infrastructure.settings import Settings

logger = logging.getLogger(__name__)

# Process-wide pool singleton — None until a successful open, and again
# after close. `get_postgres_pool()` is the single handle the domain
# accessors use; nothing else constructs connections.
_pool: Optional[AsyncConnectionPool] = None

# Serializes opens: concurrent callers that find the pool closed wait here
# so exactly one of them builds a pool (the others would each leak one).
_open_lock = asyncio.Lock()

# Set by `close_postgres()` (the shutdown hook) and cleared by the next
# explicit `open_postgres()` — while set, the lazy re-open is off, so a late
# caller cannot build a pool that nothing would ever close.
_shutdown = False

# Monotonic clock time before which `get_postgres_pool()` fails fast instead
# of probing again. Each failed lazy probe pays the full open timeout, so an
# outage must cost one probe per cooldown window, not one per request.
_retry_not_before = 0.0

# Seconds to wait for the first pool connection at boot — past this,
# Postgres reads as unreachable and the app degrades gracefully instead of
# hanging startup. Also the spacing between lazy re-open probes.
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

# Every table with a ``tenant_id`` column — the second, database-enforced
# tenant filter behind the accessors' WHERE clauses. Each gets the same
# row-level-security policy: Postgres drops any row whose ``tenant_id``
# differs from the session setting ``tenant_connection`` writes, so a
# statement that forgets ``WHERE tenant_id = %s`` matches zero rows outside
# its tenant instead of leaking. ``USING`` filters rows read / updated /
# deleted; ``WITH CHECK`` rejects rows inserted under another tenant.
#
# The policy reads the setting through ``app_current_tenant()``, which
# RAISES when the setting is empty or unset. That is what makes a connection
# taken without ``tenant_connection`` fail loudly on a tenant-table
# statement: ``RESET`` leaves a custom setting defined-but-empty (so the bare
# ``current_setting`` would not error there), and an empty label must never
# be mistaken for "no filter".
_TENANT_TABLES: tuple[str, ...] = (
    "conversations",
    "conversation_messages",
    "conversation_plans",
    "approval_requests",
    "action_results",
    "memory_facts",
)

TENANT_SETTING = "app.tenant_id"

_TENANT_FUNCTION = f"""
    CREATE OR REPLACE FUNCTION app_current_tenant() RETURNS text
    LANGUAGE plpgsql STABLE AS $$
    BEGIN
        IF COALESCE(current_setting('{TENANT_SETTING}', true), '') = '' THEN
            RAISE EXCEPTION
                '{TENANT_SETTING} is not set on this connection';
        END IF;
        RETURN current_setting('{TENANT_SETTING}');
    END;
    $$
"""

_RLS_STATEMENTS: tuple[str, ...] = (
    _TENANT_FUNCTION,  # first: CREATE POLICY validates that it exists
) + tuple(
    statement
    for table in _TENANT_TABLES
    for statement in (
        f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY",
        f"DROP POLICY IF EXISTS tenant_isolation ON {table}",
        "CREATE POLICY tenant_isolation ON "
        f"{table} USING (tenant_id = app_current_tenant()) "
        f"WITH CHECK (tenant_id = app_current_tenant())",
    )
)

# Row rights for the ordinary role, re-granted on every bootstrap so a table
# added here is reachable by the next boot without a manual step. The role
# name is interpolated as an identifier (never a DSN or SQL string).
_APP_ROLE_GRANTS: tuple[str, ...] = (
    "GRANT USAGE ON SCHEMA public TO {}",
    "GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public "
    "TO {}",
)

# How the connected role could ignore the policies — each is a bypass
# ``open_postgres`` refuses to serve with.
_RLS_BYPASS_QUERY = (
    "SELECT current_user AS role, "
    "(SELECT rolsuper OR rolbypassrls FROM pg_roles "
    " WHERE rolname = current_user) AS privileged, "
    "EXISTS (SELECT 1 FROM pg_tables "
    " WHERE schemaname = current_schema() "
    " AND tableowner = current_user) AS owns_tables"
)


async def get_postgres_pool() -> AsyncConnectionPool:
    """Return the process-wide pool, re-opening it on demand.

    When the pool is closed — Postgres was unreachable at boot, or dropped
    since — this awaits one ``open_postgres()`` attempt, so the app recovers
    as soon as Postgres is reachable, without a restart. Concurrent callers
    share a single attempt, and a failed one starts a cooldown during which
    callers fail fast instead of each paying the open timeout.

    Raises RuntimeError when Postgres is still unreachable, a probe just
    failed, or the app is shutting down. The conversation service maps that
    to the single user-meaningful 500 (infrastructure class; CR-N1).
    """
    if _pool is not None:
        return _pool
    if not _shutdown:
        global _retry_not_before
        async with _open_lock:
            if _pool is not None:  # the caller we waited for opened it
                return _pool
            if time.monotonic() >= _retry_not_before:
                await open_postgres()
                if _pool is not None:
                    return _pool
                _retry_not_before = time.monotonic() + _OPEN_TIMEOUT_SECONDS
    raise RuntimeError(
        "Postgres pool is not open (Postgres unreachable, or the app is "
        "shutting down); conversational surfaces are unavailable."
    )


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
    pool = await get_postgres_pool()
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


@asynccontextmanager
async def tenant_connection(tenant_id: str) -> AsyncIterator[AsyncConnection]:
    """One pooled connection labelled with the tenant, for tenant-table SQL.

    A label, not a lock: nothing waits here, and any number of connections
    may carry the same tenant at once (serialization is ``advisory_lock``).
    On checkout this writes the ``app.tenant_id`` session setting the
    ``tenant_isolation`` policies read, so every statement on this
    connection — including one that forgot its WHERE clause — is confined to
    the tenant; on return the label is cleared, because pool connections are
    reused and must never carry a tenant into the next checkout.

    Raises ValueError on a blank tenant (never a silent zero-row read) and
    RuntimeError when the pool is closed (see ``get_postgres_pool``).
    """
    if not tenant_id:
        raise ValueError("tenant_id must be a non-empty string")
    pool = await get_postgres_pool()
    async with pool.connection() as conn:
        await conn.execute(
            "SELECT set_config(%s, %s, false)", (TENANT_SETTING, tenant_id)
        )
        try:
            yield conn
        finally:
            try:
                # Cleared to "no tenant": a later tenant-table statement on
                # this connection without a label raises (the policy calls
                # ``app_current_tenant()``) instead of reading everything.
                await conn.execute(f"RESET {TENANT_SETTING}")
            except Exception:
                # The connection is about to return to the pool (or die), and
                # every tenant-table checkout overwrites the label anyway.
                logger.warning(
                    "could not clear %s on a pooled connection",
                    TENANT_SETTING,
                    exc_info=True,
                )


async def _bootstrap_schema(settings: Settings) -> None:
    """Run every schema, policy and grant statement as the owner role.

    One short-lived connection at ``POSTGRES_ADMIN_DSN`` (the pool's role
    cannot create tables or policies). Idempotent; raises on failure.
    """
    conn = await AsyncConnection.connect(
        settings.POSTGRES_ADMIN_DSN, autocommit=True
    )
    try:
        async with conn.transaction():
            for statement in _SCHEMA_STATEMENTS + _RLS_STATEMENTS:
                await conn.execute(statement)
            role = sql.Identifier(settings.POSTGRES_APP_ROLE)
            for grant in _APP_ROLE_GRANTS:
                await conn.execute(sql.SQL(grant).format(role))
    finally:
        await conn.close()


async def bootstrap_schema() -> None:
    """(Re)run the idempotent schema bootstrap (owner role, own connection).

    Raises on failure (infrastructure class) — safe to call on every boot:
    existing tables, indexes and the policies are left untouched.
    """
    await _bootstrap_schema(Settings())


async def _rls_bypass_reason(pool: AsyncConnectionPool) -> Optional[str]:
    """Why the pool's role would ignore the tenant policies; None if it can't.

    Three roles skip row-level security however the policies are written: a
    superuser, a role with ``BYPASSRLS``, and the role that owns the table
    (without ``FORCE``). System catalogs carry no tenant policy, so the
    pool's own connection may read them.
    """
    async with pool.connection() as conn:
        cursor = await conn.execute(_RLS_BYPASS_QUERY)
        row = await cursor.fetchone()
    if row is None:
        return "its privileges could not be read"
    if row["privileged"]:
        return "it is a superuser or holds BYPASSRLS"
    if row["owns_tables"]:
        return "it owns the tenant tables"
    return None


async def open_postgres(settings: Optional[Settings] = None) -> bool:
    """Open the pool and bootstrap the schema (lifespan startup hook).

    Degrade-gracefully: any failure here (endpoint unreachable, auth, schema
    bootstrap, or a ``POSTGRES_DSN`` role that would bypass the tenant
    policies) logs exactly one warning and returns False — the app still
    starts; only the conversational surfaces are down until a later call
    (the next ``get_postgres_pool()``) succeeds. Returns True when the pool
    is ready. Calling again after a failed open retries cleanly; it also
    starts a new lifecycle after ``close_postgres()``.

    The owner role bootstraps first, on its own connection, bounded by the
    same short timeout as the reachability probe; then the pool opens as the
    ordinary role and is refused if that role could ignore row-level
    security.

    Not itself serialized: the lifespan startup calls it once, and
    ``get_postgres_pool()`` calls it under ``_open_lock``.
    """
    global _pool, _shutdown
    if _pool is not None:
        return True
    _shutdown = False
    settings = settings or Settings()
    pool: Optional[AsyncConnectionPool] = None
    try:
        await asyncio.wait_for(
            _bootstrap_schema(settings), timeout=_OPEN_TIMEOUT_SECONDS
        )
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
        reason = await _rls_bypass_reason(pool)
        if reason is not None:
            raise RuntimeError(
                "POSTGRES_DSN connects as a role that ignores the tenant "
                f"row-level-security policies ({reason}); tenant isolation "
                "would not be enforced. Point POSTGRES_DSN at the ordinary "
                f"role ({settings.POSTGRES_APP_ROLE})."
            )
    except BaseException as open_error:
        # BaseException: a lazy attempt runs inside a request, which can be
        # cancelled mid-open — the half-open pool must not leak then either.
        logger.warning(
            "Postgres unavailable (endpoint unreachable, schema bootstrap "
            "failed, or the POSTGRES_DSN role bypasses tenant row-level "
            "security); conversational and approval surfaces will return 500 "
            "until it is fixed — retried on the next request, "
            "no restart needed. Documents and search are unaffected.",
            exc_info=True,
        )
        if pool is not None:
            try:
                await pool.close()
            except Exception:  # pragma: no cover - shutdown best effort
                logger.debug("Pool cleanup after failed open raised", exc_info=True)
        if not isinstance(open_error, Exception):
            raise
        return False
    _pool = pool
    logger.info(
        "Postgres pool ready (max_size=%d, role=%s)",
        settings.POSTGRES_POOL_MAX_SIZE,
        settings.POSTGRES_APP_ROLE,
    )
    return True


async def close_postgres() -> None:
    """Close the process-wide pool (shutdown hook; safe if unused or closed).

    Session-scoped advisory locks die with their connections, so closing
    the pool also releases any lock a crashed turn was holding — a
    conversation can never stay stranded (A17). Also stops the lazy re-open
    in ``get_postgres_pool()`` — this is the end of the lifecycle.
    """
    global _pool, _shutdown
    async with _open_lock:
        if _pool is not None:
            await _pool.close()
        _pool = None
        _shutdown = True
