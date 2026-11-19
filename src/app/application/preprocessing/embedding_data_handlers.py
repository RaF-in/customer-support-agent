"""Per-format embedding handlers — the EMBED stage.

Each handler batches chunks through the ``EmbeddingProvider`` into
``Embedded*Chunk`` entities. Batching follows the ``EMBEDDING_BATCH_SIZE``
setting (one provider call per batch), chunks longer than the provider's
input limit are split before they are handed over, and the provider's
``model_id`` is written into every chunk's metadata — part of the durable
v1 payload contract the future agent relies on.
"""

import logging
from abc import ABC
from typing import ClassVar, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.domain.chunks import (
    Chunk,
    EmbeddedChunk,
    EmbeddedDocxChunk,
    EmbeddedPdfChunk,
    EmbeddedTextChunk,
)
from app.domain.documents import CleanedDocument
from app.domain.enums import DataCategory
from app.infrastructure.embedding.base import EmbeddingProvider
from app.infrastructure.embedding.factory import get_embedding_model
from app.infrastructure.settings import Settings

logger = logging.getLogger(__name__)


class EmbeddingDataHandler(ABC):
    """Strategy: embed one format's chunks via the provider."""

    category: ClassVar[DataCategory]
    embedded_chunk_class: ClassVar[type[EmbeddedChunk]]

    def __init__(self, embedding_provider: Optional[EmbeddingProvider] = None) -> None:
        self._provider = embedding_provider or get_embedding_model()

    async def embed(
        self, *, data_model: CleanedDocument, chunks: list[Chunk]
    ) -> list[EmbeddedChunk]:
        """Embed ``chunks`` in ``EMBEDDING_BATCH_SIZE`` batches, producing
        ``Embedded*Chunk`` entities with the model id in metadata. Batches
        are awaited sequentially; the provider gathers its own API calls
        under ``EMBEDDING_MAX_CONCURRENCY``."""
        if not chunks:
            raise ValueError(
                f"Document {data_model.document_id} has no chunks to embed."
            )
        chunks = self._enforce_input_limit(chunks)
        batch_size = Settings().EMBEDDING_BATCH_SIZE
        embedded: list[EmbeddedChunk] = []
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors = await self._provider.embed(
                [chunk.content for chunk in batch]
            )
            if len(vectors) != len(batch):
                raise ValueError(
                    "Embedding provider returned %d vectors for %d texts"
                    % (len(vectors), len(batch))
                )
            for chunk, vector in zip(batch, vectors):
                embedded.append(self._to_embedded_chunk(chunk, vector))
        logger.info(
            "Embedded %d chunks for document %s (model=%s, dim=%d)",
            len(embedded),
            data_model.document_id,
            self._provider.model_id,
            self._provider.dim,
        )
        return embedded

    def _to_embedded_chunk(self, chunk: Chunk, vector: list[float]) -> EmbeddedChunk:
        """Carry every chunk field into its embedded twin and stamp the
        provider's model id into metadata (durable model record)."""
        metadata = dict(chunk.metadata)
        metadata["embedding_model_id"] = self._provider.model_id
        return self.embedded_chunk_class(
            id=chunk.id,
            tenant_id=chunk.tenant_id,
            document_id=chunk.document_id,
            content=chunk.content,
            created_at=chunk.created_at,
            metadata=metadata,
            embedding=list(vector),
        )

    def _enforce_input_limit(self, chunks: list[Chunk]) -> list[Chunk]:
        """Split any chunk longer than the provider's input limit so nothing
        oversized is ever handed to the provider. The limit is a token count
        compared against character counts — conservative, as above."""
        limit = self._provider.max_input_length
        if all(len(chunk.content) <= limit for chunk in chunks):
            return chunks
        settings = Settings()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=limit,
            chunk_overlap=min(settings.CHUNK_OVERLAP, max(limit - 1, 0)),
            length_function=len,
            is_separator_regex=False,
        )
        safe: list[Chunk] = []
        for chunk in chunks:
            if len(chunk.content) <= limit:
                safe.append(chunk)
                continue
            logger.warning(
                "Splitting %d-char chunk over the provider input limit (%d)",
                len(chunk.content),
                limit,
            )
            for text in splitter.split_text(chunk.content):
                safe.append(
                    Chunk(
                        tenant_id=chunk.tenant_id,
                        document_id=chunk.document_id,
                        content=text,
                        metadata=dict(chunk.metadata),
                    )
                )
        return safe


class PdfEmbeddingHandler(EmbeddingDataHandler):
    category: ClassVar[DataCategory] = DataCategory.PDF
    embedded_chunk_class: ClassVar[type[EmbeddedChunk]] = EmbeddedPdfChunk


class TextEmbeddingHandler(EmbeddingDataHandler):
    category: ClassVar[DataCategory] = DataCategory.TEXT
    embedded_chunk_class: ClassVar[type[EmbeddedChunk]] = EmbeddedTextChunk


class DocxEmbeddingHandler(EmbeddingDataHandler):
    category: ClassVar[DataCategory] = DataCategory.DOCX
    embedded_chunk_class: ClassVar[type[EmbeddedChunk]] = EmbeddedDocxChunk
