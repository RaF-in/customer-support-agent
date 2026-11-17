"""Tenant-scoped retrieval service (A9) — the one implementation shared by
the HTTP search route and the Self-RAG ``retrieve`` node (T4).

Extracted contract-preserving from ``api/search.py``: ready-record join,
query embedding, three-format search, orphan/mid-pipeline filtering, top-k.
HTTP-agnostic by design (plain function, no FastAPI imports) so graph nodes
can call it via ``asyncio.to_thread``; infrastructure failures raise the
exceptions below and the API layer maps them to the durable v1 error
strings.
"""

import logging
from typing import Any, Optional

from app.domain.chunks import EmbeddedDocxChunk, EmbeddedPdfChunk, EmbeddedTextChunk
from app.domain.enums import DocumentStatus
from app.domain.records import DocumentRecord
from app.infrastructure.embedding.factory import get_embedding_model

logger = logging.getLogger(__name__)

# All chunk collections are searched and the best passages win — a tenant's
# documents may span formats.
_CHUNK_CLASSES = (EmbeddedPdfChunk, EmbeddedTextChunk, EmbeddedDocxChunk)

# Record scroll page size for tenant-wide listings (mirrors the paging in
# ``api/documents.tenant_records``, which keeps its own copy for its own
# callers).
_PAGE_SIZE = 100


class RetrievalServiceError(Exception):
    """Infrastructure failure inside the retrieval service (never a
    business outcome — "no results" is a valid, empty return)."""


class RecordListingError(RetrievalServiceError):
    """The tenant's document records could not be listed (Qdrant down)."""


class QueryEmbeddingError(RetrievalServiceError):
    """The query could not be embedded."""


class VectorSearchError(RetrievalServiceError):
    """A chunk-collection similarity search failed (Qdrant down)."""


def _tenant_records(tenant_id: str) -> list[DocumentRecord]:
    """Scroll every record of one tenant, paging via point-id offset.

    Raises RecordListingError when Qdrant is unavailable.
    """
    records: list[DocumentRecord] = []
    offset: Optional[str] = None
    while True:
        page = DocumentRecord.bulk_find(tenant_id, limit=_PAGE_SIZE, offset=offset)
        if page is None:
            raise RecordListingError("vector db unavailable: could not list documents")
        records.extend(page)
        if len(page) < _PAGE_SIZE:
            return records
        offset = str(page[-1].id)


def retrieve_passages(tenant_id: str, query: str, top_k: int) -> list[dict[str, Any]]:
    """Embed ``query`` and return this tenant's best passages from live
    ``ready`` documents (R1/R4/R5) — ``{content, document_id, score,
    metadata}`` dicts, score-sorted descending, truncated to ``top_k``.

    R4 ("only ``ready`` documents are retrievable") is enforced as a join on
    live records: chunk points are only returned when a ``ready`` record for
    their ``document_id`` currently exists for this tenant. That also keeps
    orphan chunks invisible when a delete/replace purge partially failed
    (A7) and hides chunks of a document still mid-pipeline (A8).
    """
    ready_ids = {
        str(record.id)
        for record in _tenant_records(tenant_id)
        if record.status == DocumentStatus.READY
    }
    if not ready_ids:
        # No live document can match — skip the embedding call entirely.
        return []

    try:
        query_vector = get_embedding_model().embed([query])[0]
    except Exception:
        logger.exception("Query embedding failed for tenant %s", tenant_id)
        raise QueryEmbeddingError("embedding service unavailable") from None

    hits: list[dict[str, Any]] = []
    for chunk_class in _CHUNK_CLASSES:
        points = chunk_class.search(query_vector, tenant_id=tenant_id, limit=top_k)
        if points is None:
            raise VectorSearchError("vector db unavailable: search failed")
        for point in points:
            payload = dict(point.payload or {})
            document_id = str(payload.get("document_id"))
            if document_id not in ready_ids:
                # Orphan (A7) or mid-pipeline (A8) chunk — never agent-visible.
                continue
            hits.append(
                {
                    "content": str(payload.get("content", "")),
                    "document_id": document_id,
                    "score": point.score,
                    "metadata": dict(payload.get("metadata") or {}),
                }
            )

    hits.sort(key=lambda hit: hit["score"], reverse=True)
    return hits[:top_k]
