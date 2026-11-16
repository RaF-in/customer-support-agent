"""Cleaned-document entities — full extracted text per upload."""

from typing import Any, ClassVar
from uuid import UUID

from pydantic import Field

from app.domain.base.vector import VectorBaseDocument
from app.domain.enums import DataCategory


class CleanedDocument(VectorBaseDocument):
    """Full extracted text for one document, stored so extraction is
    inspectable and the chunk step can be debugged/re-run.

    ``platform`` records the extraction source: "digital-pdf" / "ocr-pdf" /
    "text" / "docx".
    """

    document_id: UUID
    content: str
    platform: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class CleanedPdfDocument(CleanedDocument):
    collection_name: ClassVar[str] = "cleaned_pdfs"
    category: ClassVar[DataCategory] = DataCategory.PDF


class CleanedTextDocument(CleanedDocument):
    collection_name: ClassVar[str] = "cleaned_texts"
    category: ClassVar[DataCategory] = DataCategory.TEXT


class CleanedDocxDocument(CleanedDocument):
    collection_name: ClassVar[str] = "cleaned_docxs"
    category: ClassVar[DataCategory] = DataCategory.DOCX
