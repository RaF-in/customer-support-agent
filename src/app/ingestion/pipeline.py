"""Asynchronous ingestion pipeline — the orchestrator (A1).

One call runs VALIDATE → EXTRACT/CLEAN → CHUNK → EMBED → LOAD to a terminal
state, driving the ``DocumentRecord`` lifecycle (``pending -> processing ->
ready|failed``; one-way transitions only) and owning the lifecycle
operations the requirements demand: duplicate flagging (A11/R8),
replace-by-name and delete (A7 — record point first, then chunk purge, no
transactions). Chunks are inserted only after every stage succeeded and the
``ready`` record is written last (A8), so non-ready content never exists in
searchable collections.

CPU-bound stages (pymupdf parsing, tesseract OCR, text splitting) run off
the event loop in worker threads (AD8); independent writes run in
parallel — but the A7/A8 ordering below is order against lifecycle
milestones, never just completion.

The initial ``pending`` record is created by the caller (the API layer,
T7); ``process`` receives it already persisted.
"""

import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import UUID

from app.application.preprocessing.dispatchers import (
    ChunkingDispatcher,
    CleaningDispatcher,
    EmbeddingDispatcher,
)
from app.domain.base.vector import VectorBaseDocument
from app.domain.chunks import (
    EmbeddedChunk,
    EmbeddedDocxChunk,
    EmbeddedPdfChunk,
    EmbeddedTextChunk,
)
from app.domain.documents import (
    CleanedDocxDocument,
    CleanedDocument,
    CleanedPdfDocument,
    CleanedTextDocument,
)
from app.domain.enums import DataCategory, DocumentStatus
from app.domain.records import DocumentRecord
from app.ingestion.validation import validate_document

logger = logging.getLogger(__name__)

_SUFFIX_TO_CATEGORY: dict[str, DataCategory] = {
    ".pdf": DataCategory.PDF,
    ".txt": DataCategory.TEXT,
    ".docx": DataCategory.DOCX,
}

# Embedded-chunk class per category — LOAD groups inserts by these.
_CHUNK_CLASS_BY_CATEGORY: dict[DataCategory, type[EmbeddedChunk]] = {
    DataCategory.PDF: EmbeddedPdfChunk,
    DataCategory.TEXT: EmbeddedTextChunk,
    DataCategory.DOCX: EmbeddedDocxChunk,
}

# Cleaned-document class per category — purged on replace/delete alongside
# the chunks (deletion removes the document and its extracted content).
_CLEANED_CLASS_BY_CATEGORY: dict[DataCategory, type[CleanedDocument]] = {
    DataCategory.PDF: CleanedPdfDocument,
    DataCategory.TEXT: CleanedTextDocument,
    DataCategory.DOCX: CleanedDocxDocument,
}

# One-way lifecycle (REQ domain rules): failed never becomes ready, and a
# terminal state never transitions again.
_ALLOWED_TRANSITIONS: dict[DocumentStatus, set[DocumentStatus]] = {
    DocumentStatus.PENDING: {DocumentStatus.PROCESSING},
    DocumentStatus.PROCESSING: {DocumentStatus.READY, DocumentStatus.FAILED},
    DocumentStatus.READY: set(),
    DocumentStatus.FAILED: set(),
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def category_from_filename(filename: str) -> Optional[DataCategory]:
    """Map a filename's suffix to its ``DataCategory`` (None = unsupported)."""
    return _SUFFIX_TO_CATEGORY.get(Path(filename).suffix.lower())


class IngestionPipeline:
    """Asynchronous document ingestion — one call per upload, terminal on
    return (N2)."""

    async def process(
        self, *, record: DocumentRecord, content: bytes
    ) -> DocumentRecord:
        """Run the full pipeline for one uploaded document.

        ``record`` must already be persisted as ``pending`` by the caller.
        Returns the terminal record (``ready`` or ``failed``), upserted at
        every transition. Raises only when even the ``failed`` status cannot
        be persisted (infrastructure outage) — never leaves a silent
        non-terminal state that could have been recorded.
        """
        try:
            return await self._process(record=record, content=content)
        except Exception as error:
            # Safety net (N2): anything unexpected still terminates.
            logger.exception("Unexpected error processing document %s", record.id)
            try:
                return await self._fail(record, f"internal error: {error}")
            except Exception:
                logger.error(
                    "Could not persist failed status for document %s "
                    "(infrastructure outage); re-raising",
                    record.id,
                )
                raise

    async def delete_document(self, *, tenant_id: str, document_id: UUID) -> bool:
        """Delete a document: record point + chunk/cleaned points (R7).
        Idempotent — an unknown id is a logged no-op. Returns True when a
        record was found and deleted."""
        for record in await self._tenant_records(tenant_id):
            if record.id == document_id:
                await self._delete_document_points(record)
                return True
        logger.info(
            "Delete: no record %s for tenant %s (no-op)", document_id, tenant_id
        )
        return False

    # ------------------------------------------------------------------
    # Stage orchestration
    # ------------------------------------------------------------------

    async def _process(
        self, *, record: DocumentRecord, content: bytes
    ) -> DocumentRecord:
        record = await self._transition(record, DocumentStatus.PROCESSING)

        # Pre-stage lifecycle (R6/A11), before any expensive work. ONE shared
        # tenant scroll feeds both the replace-purge and the duplicate flag
        # (AD6); the purge runs first so same-name + same-bytes counts as a
        # replace, not a duplicate of a record that is about to be gone.
        existing = await self._tenant_records(record.tenant_id)
        await self._purge_replaced(record, existing)
        record = self._flag_duplicate(record, existing)

        category = category_from_filename(record.filename)
        if category is None:
            return await self._fail(record, f"unsupported format: {record.filename}")

        # 1. VALIDATE (R10/N3)
        reason = validate_document(content, category)
        if reason is not None:
            return await self._fail(record, reason)

        # 2. EXTRACT/CLEAN (R2/R9) — CPU/subprocess-bound (pymupdf,
        #    tesseract): run off the event loop (AD8).
        try:
            cleaned = await asyncio.to_thread(
                CleaningDispatcher.dispatch,
                content=content,
                category=category,
                document_id=record.id,
                tenant_id=record.tenant_id,
            )
        except Exception as error:
            return await self._fail(record, f"extraction failed: {error}")

        # 3. CHUNK — CPU-bound text splitting: off the event loop (AD8).
        try:
            chunks = await asyncio.to_thread(
                ChunkingDispatcher.dispatch, data_model=cleaned
            )
        except Exception as error:
            return await self._fail(record, f"chunking failed: {error}", platform=cleaned.platform)

        # 4. EMBED (ARCH stress-test: provider down -> failed, no chunks) —
        #    awaited directly; the provider parallelizes its own batches.
        try:
            embedded = await EmbeddingDispatcher.dispatch(
                data_model=cleaned, chunks=chunks
            )
        except Exception as error:
            return await self._fail(
                record, f"embedding service unavailable: {error}", platform=cleaned.platform
            )

        # 5. LOAD (A8: ready only after everything is stored)
        return await self._load(record, cleaned, embedded)

    async def _load(
        self,
        record: DocumentRecord,
        cleaned: CleanedDocument,
        embedded: list[EmbeddedChunk],
    ) -> DocumentRecord:
        """Persist the cleaned doc and the chunks (grouped by class, with
        the groups inserted in parallel), then the ``ready`` record LAST
        (A8). Any insert failure purges what this document already wrote
        and fails — partial chunks never survive."""
        grouped: dict[type[EmbeddedChunk], list[EmbeddedChunk]] = defaultdict(list)
        for chunk in embedded:
            grouped[type(chunk)].append(chunk)

        # Chunk groups first, the cleaned document last — matching the
        # pre-async write order so the failure reason stays attributable.
        insert_classes: list[type[VectorBaseDocument]] = [
            *grouped.keys(),
            type(cleaned),
        ]
        insert_batches: list[list[VectorBaseDocument]] = [*grouped.values(), [cleaned]]
        results = await asyncio.gather(
            *(
                insert_class.bulk_insert(batch)
                for insert_class, batch in zip(insert_classes, insert_batches)
            )
        )
        if not all(results):
            written = [
                point_class
                for point_class, inserted in zip(insert_classes, results)
                if inserted
            ]
            await self._abort_load(record, written)
            chunk_inserts_ok = all(results[: len(grouped)])
            reason = (
                "vector db unavailable: cleaned document insert failed"
                if chunk_inserts_ok
                else "vector db unavailable: chunk insert failed"
            )
            return await self._fail(record, reason, platform=cleaned.platform)

        flags = dict(record.flags)
        if cleaned.metadata.get("has_unreliable_tables"):
            flags["has_unreliable_tables"] = True
        if cleaned.metadata.get("ocr_used"):
            flags["ocr_used"] = True

        record = await self._transition(
            record, DocumentStatus.READY, num_chunks=len(embedded), flags=flags
        )
        self._log_outcome(record, platform=cleaned.platform, chunk_metadata=embedded[0].metadata)
        return record

    async def _abort_load(
        self, record: DocumentRecord, written: list[type[VectorBaseDocument]]
    ) -> None:
        """Best-effort purge of partially inserted points (A8)."""
        for point_class in written:
            if not await point_class.delete_by_filter(record.tenant_id, record.id):
                logger.error(
                    "Orphan %s points may remain for document %s",
                    point_class.__name__,
                    record.id,
                )

    # ------------------------------------------------------------------
    # Lifecycle operations
    # ------------------------------------------------------------------

    async def _purge_replaced(
        self, record: DocumentRecord, existing: list[DocumentRecord]
    ) -> None:
        """Same filename in this tenant is replaced (R6): old record point
        first, then its chunk/cleaned points (A7 ordering). Runs before the
        duplicate scan so same-name + same-bytes counts as a replace, not a
        duplicate of a record that is about to be gone."""
        for other in existing:
            if other.id == record.id or other.filename != record.filename:
                continue
            logger.info(
                "Replacing document %s (%s) for tenant %s",
                other.id,
                other.filename,
                record.tenant_id,
            )
            await self._delete_document_points(other)

    def _flag_duplicate(
        self, record: DocumentRecord, existing: list[DocumentRecord]
    ) -> DocumentRecord:
        """Flag exact-duplicate content (same SHA-256 hash) among this
        tenant's other surviving records (R8/A11). Duplicates are stored and
        processed normally — only the flag differs."""
        duplicate_of: Optional[UUID] = None
        for other in existing:
            if other.id != record.id and other.content_hash == record.content_hash:
                duplicate_of = other.id
                break
        if duplicate_of is None:
            return record
        flags = dict(record.flags)
        flags["duplicate_of"] = str(duplicate_of)
        logger.info("Document %s duplicates %s", record.id, duplicate_of)
        return record.model_copy(update={"flags": flags, "updated_at": _now()})

    async def _delete_document_points(self, record: DocumentRecord) -> None:
        """A7 ordering: the record point goes first, awaited on its own (the
        document stops existing for every reader), and only then do the
        chunk/cleaned purges run — those two in parallel, since A7 is order
        against the record delete, not among the purges. A failed purge
        leaves orphan points invisible to retrieval — logged, accepted."""
        if not await DocumentRecord.delete_by_ids([record.id]):
            logger.error(
                "Failed to delete record %s; keeping its points consistent", record.id
            )
            return
        category = category_from_filename(record.filename)
        if category is None:
            return
        point_classes = (
            _CHUNK_CLASS_BY_CATEGORY[category],
            _CLEANED_CLASS_BY_CATEGORY[category],
        )
        results = await asyncio.gather(
            *(
                point_class.delete_by_filter(record.tenant_id, record.id)
                for point_class in point_classes
            )
        )
        for point_class, deleted in zip(point_classes, results):
            if not deleted:
                logger.error(
                    "Orphan %s points may remain for document %s",
                    point_class.__name__,
                    record.id,
                )

    async def _tenant_records(self, tenant_id: str) -> list[DocumentRecord]:
        """Every record of one tenant — one shared full scroll via the OVM
        (AD6), replacing the paged loop this class used to own."""
        records = await DocumentRecord.bulk_find_all(tenant_id)
        if records is None:
            raise RuntimeError("vector db unavailable: could not list records")
        return records

    # ------------------------------------------------------------------
    # Status transitions and the per-document outcome line
    # ------------------------------------------------------------------

    async def _transition(
        self, record: DocumentRecord, status: DocumentStatus, **fields: object
    ) -> DocumentRecord:
        """Move the record to ``status`` (one-way only) and upsert it. A
        status that cannot be persisted raises — a Qdrant outage must not be
        mistaken for success."""
        if status not in _ALLOWED_TRANSITIONS[record.status]:
            raise ValueError(f"Illegal status transition {record.status} -> {status}")
        updated = record.model_copy(update={"status": status, "updated_at": _now(), **fields})
        if not await DocumentRecord.bulk_insert([updated]):
            raise RuntimeError(f"vector db unavailable: could not persist status {status}")
        logger.info(
            "Document %s status: %s -> %s", record.id, record.status.value, status.value
        )
        return updated

    async def _fail(
        self, record: DocumentRecord, reason: str, platform: Optional[str] = None
    ) -> DocumentRecord:
        record = await self._transition(
            record, DocumentStatus.FAILED, failure_reason=reason
        )
        self._log_outcome(record, platform=platform)
        return record

    def _log_outcome(
        self,
        record: DocumentRecord,
        *,
        platform: Optional[str] = None,
        chunk_metadata: Optional[dict] = None,
    ) -> None:
        """Exactly one outcome line per document — the operator-scannable
        record of what happened (ARCH observability)."""
        chunk_params = "-"
        model_id = "-"
        if chunk_metadata:
            chunk_params = (
                f"size={chunk_metadata.get('chunk_size')}, "
                f"overlap={chunk_metadata.get('overlap')}"
            )
            model_id = str(chunk_metadata.get("embedding_model_id", "-"))
        logger.info(
            "Document outcome: id=%s tenant=%s filename=%s status=%s reason=%s "
            "num_chunks=%d chunk_params=(%s) embedding_model=%s platform=%s flags=%s",
            record.id,
            record.tenant_id,
            record.filename,
            record.status.value,
            record.failure_reason or "-",
            record.num_chunks,
            chunk_params,
            model_id,
            platform or "-",
            record.flags,
        )
