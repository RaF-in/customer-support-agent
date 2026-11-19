"""Tenant-scoped retrieval endpoint — the agent's surface (T7).

``POST /tenants/{tenant_id}/search`` is the durable v1 external contract
(ARCH Areas of Impact): responses are ``{content, document_id, score,
metadata}`` passages drawn from this tenant's documents only. Isolation is
structural (A3/A4): ``tenant_id`` comes from the URL path and every Qdrant
call underneath is a tenant-scoped OVM method.

The retrieval logic itself lives in ``application/retrieval.py`` (A9 — one
implementation shared with the future Self-RAG retrieve node); this module
is parsing + error mapping only, translating the service's infrastructure
exceptions into the durable v1 error strings.
"""

from typing import Any

from fastapi import APIRouter, HTTPException, Path
from pydantic import BaseModel, Field

from app.application.retrieval import (
    QueryEmbeddingError,
    RecordListingError,
    VectorSearchError,
    retrieve_passages,
)

router = APIRouter(prefix="/tenants/{tenant_id}", tags=["search"])


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
async def search(
    search_request: SearchRequest, tenant_id: str = Path(min_length=1)
) -> list[SearchHit]:
    """Embed the query and return this tenant's best passages from live
    ``ready`` documents (R1/R4/R5). Metadata carries ``unreliable_table``
    for flagged table chunks (R9)."""
    try:
        hits = await retrieve_passages(
            tenant_id=tenant_id,
            query=search_request.query,
            top_k=search_request.top_k,
        )
    except QueryEmbeddingError:
        raise HTTPException(
            status_code=500, detail="embedding service unavailable"
        ) from None
    except VectorSearchError:
        raise HTTPException(
            status_code=500, detail="vector db unavailable: search failed"
        ) from None
    except RecordListingError:
        raise HTTPException(
            status_code=500, detail="vector db unavailable: could not list documents"
        ) from None
    return [SearchHit(**hit) for hit in hits]
