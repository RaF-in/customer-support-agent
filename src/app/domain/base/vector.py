"""Object-vector mapping (OVM) base — every Qdrant read/write in the system
flows through ``VectorBaseDocument`` subclasses.

The public methods (``bulk_insert`` / ``bulk_find`` / ``search`` /
``delete_by_filter``) swallow and log Qdrant errors and return a failure
result; their ``_``-prefixed private twins raise. Query methods take
``tenant_id`` with **no default**, so a cross-tenant read cannot be expressed
by accident (decision A4 — default-to-nothing).
"""

import logging
from abc import ABC
from datetime import datetime, timezone
from typing import ClassVar, Generic, Optional, TypeVar
from uuid import UUID, uuid4

from pydantic import BaseModel, Field
from qdrant_client import models

from app.domain.enums import DataCategory
from app.infrastructure.qdrant import (
    TENANT_ID_FIELD,
    create_collection_if_missing,
    get_connection,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="VectorBaseDocument")

# Qdrant upserts are batched to keep requests small (ARCH performance note).
UPSERT_BATCH_SIZE = 100

# Placeholder vector stored for payload-only points (use_vector_index=False):
# one dimension, never similarity-searched.
DUMMY_VECTOR = [0.0]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class VectorBaseDocument(BaseModel, Generic[T], ABC):
    """Base class mapping a Pydantic entity to a Qdrant point.

    Per-entity Qdrant declaration lives in class-level attributes
    (``collection_name`` / ``category`` / ``use_vector_index``) that
    subclasses override.
    """

    id: UUID = Field(default_factory=uuid4)
    tenant_id: str
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    collection_name: ClassVar[Optional[str]] = None
    category: ClassVar[Optional[DataCategory]] = None
    use_vector_index: ClassVar[bool] = False

    # ------------------------------------------------------------------
    # Mapping (pure functions — no Qdrant access)
    # ------------------------------------------------------------------

    def to_point(self) -> models.PointStruct:
        """Serialize to a Qdrant point. Payload-only entities get a dummy
        1-dim vector; vector-indexed entities carry their embedding."""
        if self.use_vector_index:
            payload = self.model_dump(mode="json", exclude={"embedding"})
            vector = list(getattr(self, "embedding"))
        else:
            payload = self.model_dump(mode="json")
            vector = list(DUMMY_VECTOR)
        return models.PointStruct(id=str(self.id), vector=vector, payload=payload)

    @classmethod
    def from_record(cls, record: models.Record) -> T:
        """Rebuild an entity from a Qdrant scroll record."""
        payload = dict(record.payload or {})
        if cls.use_vector_index and record.vector is not None:
            payload["embedding"] = record.vector
        return cls.model_validate(payload)

    @classmethod
    def from_scored_point(cls, point: models.ScoredPoint) -> T:
        """Rebuild an entity from a search result (score lives on the point)."""
        payload = dict(point.payload or {})
        if cls.use_vector_index and point.vector is not None:
            payload["embedding"] = point.vector
        return cls.model_validate(payload)

    # ------------------------------------------------------------------
    # Public API — swallow-and-log Qdrant failures
    # ------------------------------------------------------------------

    @classmethod
    def bulk_insert(cls, documents: list[T]) -> bool:
        """Insert documents, auto-creating the collection on first use.
        Returns False on failure."""
        try:
            return cls._bulk_insert(documents)
        except Exception:
            logger.exception("Qdrant bulk_insert failed for %s", cls.__name__)
            return False

    @classmethod
    def bulk_find(
        cls, tenant_id: str, limit: int = 50, offset: Optional[str] = None
    ) -> Optional[list[T]]:
        """Fetch this tenant's documents. Returns None on failure."""
        try:
            return cls._bulk_find(tenant_id, limit, offset)
        except Exception:
            logger.exception("Qdrant bulk_find failed for %s", cls.__name__)
            return None

    @classmethod
    def search(
        cls, query_vector: list[float], tenant_id: str, limit: int = 10
    ) -> Optional[list[models.ScoredPoint]]:
        """Similarity search over this tenant's points. Chunk classes with a
        vector index only. Returns None on failure."""
        try:
            return cls._search(query_vector, tenant_id, limit)
        except Exception:
            logger.exception("Qdrant search failed for %s", cls.__name__)
            return None

    @classmethod
    def delete_by_filter(
        cls, tenant_id: str, document_id: Optional[UUID] = None
    ) -> bool:
        """Delete points by tenant, optionally narrowed to one document.
        Returns False on failure."""
        try:
            cls._delete_by_filter(tenant_id, document_id)
            return True
        except Exception:
            logger.exception("Qdrant delete_by_filter failed for %s", cls.__name__)
            return False

    @classmethod
    def delete_by_ids(cls, ids: list[UUID]) -> bool:
        """Delete points by id. Records carry no ``document_id`` payload
        (their own id *is* the document id), so single-point deletes —
        replace-by-name and delete in the pipeline — go by id.
        Returns False on failure."""
        try:
            cls._delete_by_ids(ids)
            return True
        except Exception:
            logger.exception("Qdrant delete_by_ids failed for %s", cls.__name__)
            return False

    # ------------------------------------------------------------------
    # Private implementations — raise on failure
    # ------------------------------------------------------------------

    @classmethod
    def _get_collection_name(cls) -> str:
        if cls.collection_name is None:
            raise ValueError(
                f"{cls.__name__} declares no collection_name; it is not storable."
            )
        return cls.collection_name

    @classmethod
    def _tenant_condition(cls, tenant_id: str) -> models.FieldCondition:
        return models.FieldCondition(
            key=TENANT_ID_FIELD, match=models.MatchValue(value=tenant_id)
        )

    @classmethod
    def _bulk_insert(cls, documents: list[T]) -> bool:
        if not documents:
            logger.warning("bulk_insert called with an empty list for %s", cls.__name__)
            return True
        collection_name = cls._get_collection_name()
        points = [doc.to_point() for doc in documents]
        # Vector size comes from the first batch so collection creation stays
        # independent of the embedding provider (T2/T3 decoupling).
        create_collection_if_missing(
            collection_name, vector_size=len(points[0].vector)
        )
        for start in range(0, len(points), UPSERT_BATCH_SIZE):
            get_connection().client.upsert(
                collection_name=collection_name,
                points=points[start : start + UPSERT_BATCH_SIZE],
                wait=True,
            )
        return True

    @classmethod
    def _bulk_find(
        cls, tenant_id: str, limit: int, offset: Optional[str]
    ) -> list[T]:
        records, _ = get_connection().client.scroll(
            collection_name=cls._get_collection_name(),
            scroll_filter=models.Filter(must=[cls._tenant_condition(tenant_id)]),
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=cls.use_vector_index,
        )
        return [cls.from_record(record) for record in records]

    @classmethod
    def _search(
        cls, query_vector: list[float], tenant_id: str, limit: int
    ) -> list[models.ScoredPoint]:
        if not cls.use_vector_index:
            raise ValueError(
                f"{cls.__name__} is not vector-indexed; search() is only "
                "available on EmbeddedChunk subclasses (A8)."
            )
        response = get_connection().client.query_points(
            collection_name=cls._get_collection_name(),
            query=query_vector,
            query_filter=models.Filter(must=[cls._tenant_condition(tenant_id)]),
            limit=limit,
            with_payload=True,
        )
        return response.points

    @classmethod
    def _delete_by_filter(cls, tenant_id: str, document_id: Optional[UUID]) -> None:
        must = [cls._tenant_condition(tenant_id)]
        if document_id is not None:
            must.append(
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=str(document_id)),
                )
            )
        get_connection().client.delete(
            collection_name=cls._get_collection_name(),
            points_selector=models.FilterSelector(filter=models.Filter(must=must)),
        )

    @classmethod
    def _delete_by_ids(cls, ids: list[UUID]) -> None:
        get_connection().client.delete(
            collection_name=cls._get_collection_name(),
            points_selector=models.PointIdsList(points=[str(i) for i in ids]),
        )
