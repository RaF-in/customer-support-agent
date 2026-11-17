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


class ApprovalStatus(StrEnum):
    """Lifecycle of an approval request.

    Only ``pending -> approved | rejected`` is a valid transition, and the
    runtime enforces it (T7) — not the record model. ``expired`` is
    reserved for a future expiry mechanism (ARCH Open Question).
    """

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class ActionStatus(StrEnum):
    """Outcome of one action execution — written once, immutable."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
