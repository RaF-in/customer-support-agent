"""Pluggable LangGraph checkpointer wiring (A7).

``CHECKPOINT_BACKEND`` selects the backend: ``sqlite`` (default —
``AsyncSqliteSaver`` at ``SQLITE_CHECKPOINT_PATH``, zero extra infra) or
``postgres`` (``AsyncPostgresSaver`` at ``POSTGRES_DSN``, started with
``docker compose --profile postgres up -d``). Unknown backends fail fast
with a clear message, exactly like the embedding factory.

Async savers must be *entered* before use, so ``get_checkpointer()``
returns one process-wide **entered** instance (parent directory and sqlite
file created, ``setup()`` run — tables exist before the first write).
Exit it with ``close_checkpointer()`` on shutdown.

Known dev constraint: the sqlite checkpointer is single-writer — fine for
a single-process deployment; use the postgres path for multi-worker
(ARCH Open Question). The sqlite file is local runtime state (gitignored).
"""

import logging
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import Any, Optional

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.infrastructure.settings import Settings

logger = logging.getLogger(__name__)


def _select_checkpointer_cm(
    settings: Settings,
) -> AbstractAsyncContextManager[Any]:
    """Map ``CHECKPOINT_BACKEND`` to an *unentered* context manager.

    Split out from ``get_checkpointer`` so a backend can be selected and
    inspected (class + DSN/path) without opening a live connection — the
    connection happens on enter, which for postgres requires the compose
    profile to be up.
    """
    backend = settings.CHECKPOINT_BACKEND
    if backend == "sqlite":
        path = Path(settings.SQLITE_CHECKPOINT_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        return AsyncSqliteSaver.from_conn_string(str(path))
    if backend == "postgres":
        # The DSN is never logged (may carry credentials — ARCH security).
        return AsyncPostgresSaver.from_conn_string(settings.POSTGRES_DSN)
    raise ValueError(
        f"Unknown checkpoint backend {backend!r}. Known backends: sqlite, postgres."
    )


# Process-wide entered singleton (plus the context manager that owns it).
_checkpointer: Optional[BaseCheckpointSaver] = None
_checkpointer_cm: Optional[AbstractAsyncContextManager[Any]] = None


async def get_checkpointer() -> BaseCheckpointSaver:
    """Return the process-wide entered checkpointer (``setup()`` done)."""
    global _checkpointer, _checkpointer_cm
    if _checkpointer is not None:
        return _checkpointer
    settings = Settings()
    _checkpointer_cm = _select_checkpointer_cm(settings)
    saver = await _checkpointer_cm.__aenter__()
    await saver.setup()
    _checkpointer = saver
    logger.info(
        "Checkpointer ready: backend=%s", settings.CHECKPOINT_BACKEND
    )
    return _checkpointer


async def close_checkpointer() -> None:
    """Exit the process-wide checkpointer (shutdown hook; safe if unused)."""
    global _checkpointer, _checkpointer_cm
    if _checkpointer_cm is not None:
        await _checkpointer_cm.__aexit__(None, None, None)
    _checkpointer = None
    _checkpointer_cm = None
