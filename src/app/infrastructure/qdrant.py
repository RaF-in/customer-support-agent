"""Qdrant connection and collection bootstrap.

All collection creation and payload indexing lives here; every other module
reaches Qdrant only through ``VectorBaseDocument`` (``app.domain.base.vector``),
which calls ``get_connection()`` from this module.
"""

import logging
from typing import Optional

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http import exceptions as qdrant_exceptions

from app.infrastructure.settings import Settings

logger = logging.getLogger(__name__)

# Payload field every entity filters on — indexed on every collection so
# tenant filters never degrade to full scans.
TENANT_ID_FIELD = "tenant_id"

# HTTP 409 Conflict — the single outcome the race-safe creators below treat
# as success: the collection/index already exists because a concurrent
# creator (gathered bootstrap, another worker) won the race.
_ALREADY_EXISTS_STATUS = 409


def _is_already_exists(exc: qdrant_exceptions.UnexpectedResponse) -> bool:
    """True only for Qdrant's already-exists response — genuine create
    failures must propagate, never be blanket-swallowed."""
    return exc.status_code == _ALREADY_EXISTS_STATUS


class QdrantConnection:
    """Lazily-created connection to the Qdrant instance defined by settings.

    The constructor does no I/O: the async transport is built on first
    ``client`` access, so importing and booting stay side-effect-free even
    with Qdrant down. ``aclose()`` closes the transport at lifespan
    shutdown; a later ``client`` access reconnects lazily.
    """

    def __init__(self, url: str) -> None:
        self._url = url
        self._client: Optional[AsyncQdrantClient] = None

    @property
    def client(self) -> AsyncQdrantClient:
        if self._client is None:
            logger.info("Connecting to Qdrant at %s", self._url)
            self._client = AsyncQdrantClient(url=self._url)
        return self._client

    async def aclose(self) -> None:
        """Close the underlying transport and drop it (no-op when never
        connected)."""
        if self._client is None:
            return
        client, self._client = self._client, None
        await client.close()


_connection: Optional[QdrantConnection] = None


def get_connection() -> QdrantConnection:
    """Process-wide Qdrant connection singleton."""
    global _connection
    if _connection is None:
        _connection = QdrantConnection(url=Settings().QDRANT_URL)
    return _connection


async def aclose_connection() -> None:
    """Lifespan-shutdown entrypoint: close the singleton's transport, if any."""
    if _connection is not None:
        await _connection.aclose()


async def create_collection_if_missing(
    collection_name: str,
    vector_size: int,
    distance: models.Distance = models.Distance.COSINE,
) -> None:
    """Create a collection (vector size taken from the first batch of data)
    with a ``tenant_id`` payload index. No-op when the collection exists.

    Race-safe under concurrent callers (gathered bootstrap, multi-worker):
    two creators that both pass ``collection_exists`` collide on the create
    call, and only the already-exists (409) outcome counts as success —
    any other failure propagates.
    """
    client = get_connection().client
    if await client.collection_exists(collection_name):
        return
    try:
        await client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=distance),
        )
        logger.info(
            "Created Qdrant collection %r (vector_size=%d, distance=%s)",
            collection_name,
            vector_size,
            distance,
        )
    except qdrant_exceptions.UnexpectedResponse as exc:
        if not _is_already_exists(exc):
            raise
        logger.debug(
            "Qdrant collection %r already exists (concurrent create won the race)",
            collection_name,
        )
    await create_tenant_id_index(collection_name)


async def create_tenant_id_index(collection_name: str) -> None:
    """Index ``tenant_id`` as a keyword so tenant filters stay fast.

    Race-safe like collection creation: a concurrent creator that already
    indexed the field is a success, anything else propagates.
    """
    try:
        await get_connection().client.create_payload_index(
            collection_name=collection_name,
            field_name=TENANT_ID_FIELD,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
    except qdrant_exceptions.UnexpectedResponse as exc:
        if not _is_already_exists(exc):
            raise
        return
    logger.info(
        "Indexed %s.%s as keyword for tenant filters", collection_name, TENANT_ID_FIELD
    )
