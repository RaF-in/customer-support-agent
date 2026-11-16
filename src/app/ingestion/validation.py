"""Upload validation — the VALIDATE stage (R10, N3).

``validate_document`` returns None when the upload is processable, else a
human-readable reason string that flows into the failed ``DocumentRecord``.
Limits come from settings (A10) — never hardcoded. Mapping reasons to HTTP
status codes (413/422) is the API layer's job (T7); this module speaks
reason strings only.
"""

import io
import logging
from typing import Optional

import pymupdf
from docx import Document as DocxDocument

from app.domain.enums import DataCategory
from app.infrastructure.settings import Settings

logger = logging.getLogger(__name__)

_BYTES_PER_MB = 1024 * 1024


def validate_document(content: bytes, category: DataCategory) -> Optional[str]:
    """Screen an upload for rejection causes: size, password protection,
    corruption, page count. Returns None when processable, else the reason."""
    settings = Settings()
    size_mb = len(content) / _BYTES_PER_MB
    if size_mb > settings.MAX_FILE_SIZE_MB:
        return (
            f"exceeds size limit: {size_mb:.1f} MB > "
            f"{settings.MAX_FILE_SIZE_MB} MB"
        )
    if category == DataCategory.PDF:
        return _validate_pdf(content, settings.MAX_PAGES)
    if category == DataCategory.DOCX:
        return _validate_docx(content)
    # Plain text has no structure to be corrupt — any bytes are valid text.
    return None


def _validate_pdf(content: bytes, max_pages: int) -> Optional[str]:
    try:
        document = pymupdf.open(stream=content, filetype="pdf")
    except Exception as error:
        logger.info(
            "Rejected unparseable PDF (%s: %s)", type(error).__name__, error
        )
        return "corrupted"
    try:
        if document.needs_pass:
            return "password-protected"
        # pymupdf silently repairs broken PDFs (e.g. truncated bytes with the
        # xref gone) instead of raising — a repair means the structure was
        # malformed, and partial text is still extractable, so reject here
        # or a truncated file would be ingested with silently missing
        # content (R10).
        if document.is_repaired:
            return "corrupted"
        if document.page_count > max_pages:
            return (
                f"exceeds size limit: {document.page_count} pages > "
                f"{max_pages} pages"
            )
    finally:
        document.close()
    return None


def _validate_docx(content: bytes) -> Optional[str]:
    try:
        DocxDocument(io.BytesIO(content))
    except Exception as error:
        logger.info(
            "Rejected unparseable DOCX (%s: %s)", type(error).__name__, error
        )
        return "corrupted"
    return None
