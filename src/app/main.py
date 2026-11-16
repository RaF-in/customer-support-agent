"""FastAPI application factory — the HTTP surface (T7).

Routers are tenant-scoped under ``/tenants/{tenant_id}`` (A3): the path
segment is the only tenant identity the system ever accepts. This module
is the composition root: it wires routers and runs the collection
bootstrap at startup. Booting itself stays resilient — infrastructure
connections are lazy and a failed bootstrap only logs a warning, so the
app starts even with Qdrant or the embedding provider down.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from app.api import documents, search

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Make pipeline outcome lines visible under ``uvicorn`` (ARCH
    observability: one scannable line per document outcome)."""
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _bootstrap_collections() -> None:
    """Ensure every declared collection exists with its expected vector
    size and a ``tenant_id`` index (ARCH: indexes created at bootstrap).

    Collections also auto-create on first insert (OVM fallback), but the
    *search* path has no insert — without this, querying a tenant whose
    documents never used one of the three chunk formats would hit a
    missing collection and read as an outage. Payload-only entities get
    the 1-dim dummy vector size; chunk collections bind to the embedding
    provider's dim (switching provider = drop + recreate, documented
    constraint).
    """
    # Imported here so importing this module stays free of side effects.
    from app.domain.chunks import EmbeddedDocxChunk, EmbeddedPdfChunk, EmbeddedTextChunk
    from app.domain.documents import (
        CleanedDocxDocument,
        CleanedPdfDocument,
        CleanedTextDocument,
    )
    from app.domain.records import DocumentRecord
    from app.infrastructure.embedding.factory import get_embedding_model
    from app.infrastructure.qdrant import create_collection_if_missing

    payload_only: list[tuple[str, int]] = [
        (entity.collection_name, 1)  # dummy 1-dim vector (use_vector_index=False)
        for entity in (DocumentRecord, CleanedPdfDocument, CleanedTextDocument, CleanedDocxDocument)
    ]
    chunk_collections: list[tuple[str, int]] = []
    try:
        dim = get_embedding_model().dim
        chunk_collections = [
            (entity.collection_name, dim)
            for entity in (EmbeddedPdfChunk, EmbeddedTextChunk, EmbeddedDocxChunk)
        ]
    except Exception:
        logger.warning(
            "Could not determine embedding dimension for collection bootstrap "
            "(provider construction failed); chunk collections will be created "
            "on first insert",
            exc_info=True,
        )
    try:
        for collection_name, vector_size in payload_only + chunk_collections:
            create_collection_if_missing(collection_name, vector_size=vector_size)
    except Exception:
        logger.warning(
            "Collection bootstrap failed (is Qdrant up?); collections will be "
            "created on first insert. Searching a never-inserted format may "
            "return 500 until the server is restarted with Qdrant reachable.",
            exc_info=True,
        )


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    _bootstrap_collections()
    yield


def create_app() -> FastAPI:
    """Build the application and register the tenant-scoped routers."""
    _configure_logging()
    application = FastAPI(
        title="Customer Support Agent — Document Ingestion API",
        version="0.1.0",
        description=(
            "Tenant-scoped document ingestion (synchronous pipeline) and "
            "retrieval surface for the support agent. The tenant path "
            "segment is the only identity — no auth in MVP."
        ),
        lifespan=lifespan,
    )
    application.include_router(documents.router)
    application.include_router(search.router)
    return application


app = create_app()
