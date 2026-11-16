"""Per-format cleaning handlers — the EXTRACT/CLEAN stage (R2, R9).

Each handler turns raw uploaded bytes into a ``Cleaned*Document``: full
extracted text, ``platform`` (extraction source), and ``metadata``
(extraction stats + table flags). PDF pages route between embedded-text
extraction and Tesseract OCR per page (A9); tables are never rejected —
extracted best-effort and flagged (R9). Dispatcher wiring arrives in T5;
until then handlers are invoked directly.
"""

import io
import logging
from abc import ABC, abstractmethod
from typing import ClassVar
from uuid import UUID

import pymupdf
import pytesseract
from docx import Document as DocxDocument
from PIL import Image  # Pillow ships as a pytesseract dependency

from app.domain.documents import (
    CleanedDocument,
    CleanedDocxDocument,
    CleanedPdfDocument,
    CleanedTextDocument,
)
from app.domain.enums import DataCategory

logger = logging.getLogger(__name__)

# A PDF page with fewer embedded characters than this is treated as scanned
# (image-only) and routed to OCR (A9 embedded-text detection).
EMBEDDED_TEXT_MIN_CHARS = 10

# Rasterization resolution for OCR — 300 DPI is Tesseract's sweet spot.
OCR_DPI = 300


class CleaningDataHandler(ABC):
    """Strategy: extract the text of one upload format."""

    category: ClassVar[DataCategory]

    @abstractmethod
    def clean(
        self, *, content: bytes, document_id: UUID, tenant_id: str
    ) -> CleanedDocument:
        """Extract ``content`` into a ``Cleaned*Document`` for one document."""


class PdfCleaningHandler(CleaningDataHandler):
    """PDF text extraction with per-page digital/OCR routing (R2)."""

    category: ClassVar[DataCategory] = DataCategory.PDF

    def clean(
        self, *, content: bytes, document_id: UUID, tenant_id: str
    ) -> CleanedPdfDocument:
        document = pymupdf.open(stream=content, filetype="pdf")
        try:
            page_texts: list[str] = []
            ocr_pages = 0
            tables_found = 0
            for number, page in enumerate(document, start=1):
                text = page.get_text().strip()
                if len(text) >= EMBEDDED_TEXT_MIN_CHARS:
                    page_texts.append(text)
                    tables_found += self._count_tables(page, number)
                    continue
                logger.debug("Page %d lacks embedded text; routing to OCR", number)
                page_texts.append(self._ocr_page(page))
                ocr_pages += 1
        finally:
            document.close()

        platform = "ocr-pdf" if ocr_pages else "digital-pdf"
        metadata = {
            "num_pages": len(page_texts),
            "num_ocr_pages": ocr_pages,
            "ocr_used": ocr_pages > 0,
            "tables_found": tables_found,
            # Table cell text pulled from PDFs is unordered best-effort —
            # flag it so retrieval never trusts it blindly (R9).
            "has_unreliable_tables": tables_found > 0,
        }
        logger.info(
            "PDF extraction complete: %d pages (%d via OCR), %d tables, platform=%s",
            len(page_texts),
            ocr_pages,
            tables_found,
            platform,
        )
        return CleanedPdfDocument(
            document_id=document_id,
            tenant_id=tenant_id,
            content="\n\n".join(page_texts),
            platform=platform,
            metadata=metadata,
        )

    def _count_tables(self, page: "pymupdf.Page", number: int) -> int:
        """Best-effort table detection on a digital page — never fatal."""
        try:
            return len(page.find_tables().tables)
        except Exception:
            logger.debug("Table detection failed on page %d; skipping", number)
            return 0

    def _ocr_page(self, page: "pymupdf.Page") -> str:
        """Rasterize one page and OCR it with local Tesseract (A9)."""
        pixmap = page.get_pixmap(dpi=OCR_DPI)
        image = Image.open(io.BytesIO(pixmap.tobytes("png")))
        return pytesseract.image_to_string(image).strip()


class TextCleaningHandler(CleaningDataHandler):
    """Plain-text passthrough — any bytes decode as text."""

    category: ClassVar[DataCategory] = DataCategory.TEXT

    def clean(
        self, *, content: bytes, document_id: UUID, tenant_id: str
    ) -> CleanedTextDocument:
        text = content.decode("utf-8", errors="replace").strip()
        logger.info("Text extraction complete: %d chars", len(text))
        return CleanedTextDocument(
            document_id=document_id,
            tenant_id=tenant_id,
            content=text,
            platform="text",
            metadata={"num_chars": len(text)},
        )


class DocxCleaningHandler(CleaningDataHandler):
    """DOCX paragraphs plus best-effort table text (R9)."""

    category: ClassVar[DataCategory] = DataCategory.DOCX

    def clean(
        self, *, content: bytes, document_id: UUID, tenant_id: str
    ) -> CleanedDocxDocument:
        document = DocxDocument(io.BytesIO(content))
        parts = [
            paragraph.text
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        ]
        tables_found = len(document.tables)
        for table in document.tables:
            for row in table.rows:
                cells = [" ".join(cell.text.split()) for cell in row.cells]
                parts.append(" | ".join(cells))
        metadata = {
            "num_paragraphs": len(document.paragraphs),
            "tables_found": tables_found,
            # DOCX tables flatten to " | "-joined rows — layout is lost, so
            # the extraction is flagged unreliable (R9).
            "has_unreliable_tables": tables_found > 0,
        }
        logger.info(
            "DOCX extraction complete: %d paragraphs, %d tables",
            len(document.paragraphs),
            tables_found,
        )
        return CleanedDocxDocument(
            document_id=document_id,
            tenant_id=tenant_id,
            content="\n".join(parts),
            platform="docx",
            metadata=metadata,
        )
