"""Chunk entities — the agent-facing searchable units and their
pre-embedding form."""

import hashlib
from typing import Any, ClassVar
from uuid import UUID

from pydantic import Field, model_validator

from app.domain.base.vector import VectorBaseDocument
from app.domain.enums import DataCategory


def generate_chunk_id(content: str, document_id: str | UUID) -> UUID:
    """Deterministic UUID from MD5(content + document_id) — the same passage
    in the same document always maps to the same point id, so re-uploads
    upsert idempotently instead of duplicating points."""
    digest = hashlib.md5(f"{content}{document_id}".encode("utf-8")).hexdigest()
    return UUID(digest)


class Chunk(VectorBaseDocument):
    """A passage split from a cleaned document. Transient: plain chunks are
    the chunking-stage output and declare no collection — only
    ``EmbeddedChunk`` subclasses are stored (A8).

    ``metadata`` carries: ``chunk_size``, ``overlap``, ``embedding_model_id``
    (written by T5) and ``unreliable_table`` (R9).
    """

    document_id: UUID
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _derive_id(cls, data: Any) -> Any:
        if isinstance(data, dict) and not data.get("id"):
            content, document_id = data.get("content"), data.get("document_id")
            if content and document_id is not None:
                data["id"] = generate_chunk_id(str(content), str(document_id))
        return data


class EmbeddedChunk(Chunk):
    """A chunk with its embedding — the only similarity-searchable entity."""

    embedding: list[float]

    use_vector_index: ClassVar[bool] = True


class PdfChunk(Chunk):
    category: ClassVar[DataCategory] = DataCategory.PDF


class TextChunk(Chunk):
    category: ClassVar[DataCategory] = DataCategory.TEXT


class DocxChunk(Chunk):
    category: ClassVar[DataCategory] = DataCategory.DOCX


class EmbeddedPdfChunk(EmbeddedChunk):
    collection_name: ClassVar[str] = "pdf_chunks"
    category: ClassVar[DataCategory] = DataCategory.PDF


class EmbeddedTextChunk(EmbeddedChunk):
    collection_name: ClassVar[str] = "text_chunks"
    category: ClassVar[DataCategory] = DataCategory.TEXT


class EmbeddedDocxChunk(EmbeddedChunk):
    collection_name: ClassVar[str] = "docx_chunks"
    category: ClassVar[DataCategory] = DataCategory.DOCX
