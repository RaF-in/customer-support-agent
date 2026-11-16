"""Tenant-scoped retrieval endpoint — the agent's surface (T7).

``POST /tenants/{tenant_id}/search`` is the durable v1 external contract
(ARCH Areas of Impact): responses are ``{content, document_id, score,
metadata}`` passages drawn from this tenant's documents only. Isolation is
structural (A3/A4): ``tenant_id`` comes from the URL path and every Qdrant
call below is a tenant-scoped OVM method.

R4 ("only ``ready`` documents are retrievable") is enforced as a join on
live records: chunk points are only returned when a ``ready`` record for
their ``document_id`` currently exists for this tenant. That also keeps
orphan chunks invisible when a delete/replace purge partially failed (A7)
and hides chunks of a document still mid-pipeline (A8).
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Path
from pydantic import BaseModel, Field

from app.api.documents import tenant_records
from app.domain.chunks import EmbeddedDocxChunk, EmbeddedPdfChunk, EmbeddedTextChunk
from app.domain.enums import DocumentStatus
from app.infrastructure.embedding.factory import get_embedding_model

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tenants/{tenant_id}", tags=["search"])

# All chunk collections are searched and the best passages win — a tenant's
# documents may span formats.
_CHUNK_CLASSES = (EmbeddedPdfChunk, EmbeddedTextChunk, EmbeddedDocxChunk)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=100)


class SearchHit(BaseModel):
    """One retrieved passage — shape is the v1 agent contract."""

    content: str
    document_id: str
    score: float
    metadata: dict[str, Any]


@router.post("/search", response_model=list[SearchHit])
def search(
    search_request: SearchRequest, tenant_id: str = Path(min_length=1)
) -> list[SearchHit]:
    """Embed the query and return this tenant's best passages from live
    ``ready`` documents (R1/R4/R5). Metadata carries ``unreliable_table``
    for flagged table chunks (R9)."""
    ready_ids = {
        str(record.id)
        for record in tenant_records(tenant_id)
        if record.status == DocumentStatus.READY
    }
    if not ready_ids:
        # No live document can match — skip the embedding call entirely.
        return []

    try:
        query_vector = get_embedding_model().embed([search_request.query])[0]
    except Exception:
        logger.exception("Query embedding failed for tenant %s", tenant_id)
        raise HTTPException(
            status_code=500, detail="embedding service unavailable"
        ) from None

    hits: list[SearchHit] = []
    for chunk_class in _CHUNK_CLASSES:
        points = chunk_class.search(
            query_vector, tenant_id=tenant_id, limit=search_request.top_k
        )
        if points is None:
            raise HTTPException(
                status_code=500, detail="vector db unavailable: search failed"
            )
        for point in points:
            payload = dict(point.payload or {})
            document_id = str(payload.get("document_id"))
            if document_id not in ready_ids:
                # Orphan (A7) or mid-pipeline (A8) chunk — never agent-visible.
                continue
            hits.append(
                SearchHit(
                    content=str(payload.get("content", "")),
                    document_id=document_id,
                    score=point.score,
                    metadata=dict(payload.get("metadata") or {}),
                )
            )

    hits.sort(key=lambda hit: hit.score, reverse=True)
    return hits[: search_request.top_k]
