"""Qdrant connection and collection bootstrap.

All collection creation and payload indexing lives here; every other module
reaches Qdrant only through ``VectorBaseDocument`` (``app.domain.base.vector``),
which calls ``get_connection()`` from this module.
"""

import logging
from typing import Optional

from qdrant_client import QdrantClient, models

from app.infrastructure.settings import Settings

logger = logging.getLogger(__name__)

# Payload field every entity filters on — indexed on every collection so
# tenant filters never degrade to full scans.
TENANT_ID_FIELD = "tenant_id"


class QdrantConnection:
    """Lazily-created connection to the Qdrant instance defined by settings."""

    def __init__(self, url: str) -> None:
        self._url = url
        self._client: Optional[QdrantClient] = None

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            logger.info("Connecting to Qdrant at %s", self._url)
            self._client = QdrantClient(url=self._url)
        return self._client


_connection: Optional[QdrantConnection] = None


def get_connection() -> QdrantConnection:
    """Process-wide Qdrant connection singleton."""
    global _connection
    if _connection is None:
        _connection = QdrantConnection(url=Settings().QDRANT_URL)
    return _connection


def create_collection_if_missing(
    collection_name: str,
    vector_size: int,
    distance: models.Distance = models.Distance.COSINE,
) -> None:
    """Create a collection (vector size taken from the first batch of data)
    with a ``tenant_id`` payload index. No-op when the collection exists."""
    client = get_connection().client
    if client.collection_exists(collection_name):
        return
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=vector_size, distance=distance),
    )
    logger.info(
        "Created Qdrant collection %r (vector_size=%d, distance=%s)",
        collection_name,
        vector_size,
        distance,
    )
    create_tenant_id_index(collection_name)


def create_tenant_id_index(collection_name: str) -> None:
    """Index ``tenant_id`` as a keyword so tenant filters stay fast."""
    get_connection().client.create_payload_index(
        collection_name=collection_name,
        field_name=TENANT_ID_FIELD,
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    logger.info("Indexed %s.%s as keyword for tenant filters", collection_name, TENANT_ID_FIELD)
