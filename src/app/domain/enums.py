from enum import StrEnum


class DocumentStatus(StrEnum):
    """Lifecycle of an uploaded document through the ingestion pipeline."""

    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class DataCategory(StrEnum):
    """Supported ingestion formats.

    Values drive factory routing and Qdrant collection naming
    (e.g. `pdfs` -> `cleaned_pdfs` / `pdf_chunks`).
    """

    PDF = "pdfs"
    TEXT = "texts"
    DOCX = "docxs"
