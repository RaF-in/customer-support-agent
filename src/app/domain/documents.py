"""Cleaned-document entities — full extracted text per upload.

One ``CleanedDocument`` class in the single ``cleaned`` collection; the
upload format rides the point as a ``format`` payload field. The former
per-format collections (``cleaned_pdfs`` / ``cleaned_texts`` /
``cleaned_docxs``) were consolidated into one — same shape as the chunks
consolidation.
"""

from typing import Any, ClassVar
from uuid import UUID

from pydantic import Field

from app.domain.base.vector import VectorBaseDocument
from app.domain.enums import DataCategory


class CleanedDocument(VectorBaseDocument):
    """Full extracted text for one document, stored so extraction is
    inspectable and the chunk step can be debugged/re-run.

    ``platform`` records the extraction source: "digital-pdf" / "ocr-pdf" /
    "text" / "docx". ``format`` records the upload format and drives the
    chunking/embedding dispatch.
    """

    document_id: UUID
    content: str
    platform: str
    format: DataCategory
    metadata: dict[str, Any] = Field(default_factory=dict)

    collection_name: ClassVar[str] = "cleaned"
