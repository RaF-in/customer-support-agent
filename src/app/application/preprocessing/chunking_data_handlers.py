"""Per-format chunking handlers — the CHUNK stage.

Each handler splits a ``CleanedDocument`` into transient ``Chunk``
entities with LangChain's ``RecursiveCharacterTextSplitter`` (REQ Decision
10 preference). Size and overlap come from settings (A10); the effective
size also respects the embedding provider's ``max_input_length`` so no
chunk is ever produced oversized for the model. Chunk ids use the T2
MD5(content + document_id) derivation, making re-uploads upsert
idempotently.

The splitting logic is format-independent — subclasses only declare the
``DataCategory`` they serve (the extension slot for per-format chunk sizes).
"""

import logging
from abc import ABC
from typing import ClassVar, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.domain.chunks import Chunk
from app.domain.documents import CleanedDocument
from app.domain.enums import DataCategory
from app.infrastructure.embedding.base import EmbeddingProvider
from app.infrastructure.embedding.factory import get_embedding_model
from app.infrastructure.settings import Settings

logger = logging.getLogger(__name__)


class ChunkingDataHandler(ABC):
    """Strategy: split one cleaned format into chunks."""

    category: ClassVar[DataCategory]

    def __init__(self, embedding_provider: Optional[EmbeddingProvider] = None) -> None:
        # The provider is consulted for max_input_length only (never called
        # here); accepting an injected provider keeps scratch verification
        # stub-friendly without touching the singleton.
        self._provider = embedding_provider or get_embedding_model()

    def chunk(self, *, data_model: CleanedDocument) -> list[Chunk]:
        """Split ``data_model.content`` into chunks no larger than the
        smallest of the configured chunk size and the provider input limit."""
        if not data_model.content:
            raise ValueError(
                f"Document {data_model.document_id} has no content to chunk."
            )
        settings = Settings()
        # max_input_length is a token count; comparing it against character
        # counts is conservative (chars >= tokens for real text), which is
        # the safe direction for this check.
        chunk_size = min(settings.CHUNK_SIZE, self._provider.max_input_length)
        chunk_overlap = min(settings.CHUNK_OVERLAP, max(chunk_size - 1, 0))
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        texts = splitter.split_text(data_model.content)
        unreliable_table = bool(data_model.metadata.get("has_unreliable_tables", False))
        chunks = [
            Chunk(
                tenant_id=data_model.tenant_id,
                document_id=data_model.document_id,
                content=text,
                metadata={
                    "chunk_size": chunk_size,
                    "overlap": chunk_overlap,
                    "unreliable_table": unreliable_table,
                },
            )
            for text in texts
        ]
        logger.info(
            "Chunked document %s: %d chars -> %d chunks (chunk_size=%d, overlap=%d)",
            data_model.document_id,
            len(data_model.content),
            len(chunks),
            chunk_size,
            chunk_overlap,
        )
        return chunks


class PdfChunkingHandler(ChunkingDataHandler):
    category: ClassVar[DataCategory] = DataCategory.PDF


class TextChunkingHandler(ChunkingDataHandler):
    category: ClassVar[DataCategory] = DataCategory.TEXT


class DocxChunkingHandler(ChunkingDataHandler):
    category: ClassVar[DataCategory] = DataCategory.DOCX
