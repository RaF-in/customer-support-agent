"""Tenant-scoped document endpoints — upload, list, get, delete (T7).

The upload endpoint owns the HTTP-level validation gates (ARCH error
rules): 413 for oversized bodies *before* the body is read, 422 for
unsupported formats or a missing file part — both before any
``DocumentRecord`` is created. Everything that can only be judged by
processing the document is a document-level problem and comes back as a
200 with terminal status ``failed`` and a reason (never a bare 500); 500
is reserved for infrastructure outages.

``tenant_id`` comes from the URL path only (A3/A4) — no form field or
body field can ever supply it.
"""

import hashlib
import logging
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Path, Request
from fastapi.responses import Response
from pydantic import BaseModel
from starlette.datastructures import UploadFile

from app.domain.enums import DocumentStatus
from app.domain.records import DocumentRecord
from app.ingestion.pipeline import IngestionPipeline, category_from_filename
from app.infrastructure.settings import Settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tenants/{tenant_id}", tags=["documents"])

# Multipart framing overhead (boundaries, part headers) tolerated on top of
# the file-size cap in the Content-Length pre-check; the authoritative check
# runs on the received bytes.
_MULTIPART_OVERHEAD_SLACK = 64 * 1024

# Record scroll page size for tenant-wide listings.
_PAGE_SIZE = 100

_SUPPORTED_SUFFIXES = ".pdf / .txt / .docx"


class UploadResponse(BaseModel):
    """Terminal upload result — the caller never polls (N2)."""

    document_id: UUID
    status: DocumentStatus
    reason: Optional[str] = None
    flags: dict[str, Any]
    num_chunks: int


class DocumentSummary(BaseModel):
    """List item — status/flags/reason visibility without a UI (R3)."""

    document_id: UUID
    filename: str
    status: DocumentStatus
    reason: Optional[str] = None
    flags: dict[str, Any]
    created_at: datetime


class DocumentDetail(BaseModel):
    """Single-document view — the full operator-facing record (R3)."""

    document_id: UUID
    tenant_id: str
    filename: str
    content_hash: str
    status: DocumentStatus
    reason: Optional[str] = None
    flags: dict[str, Any]
    num_chunks: int
    created_at: datetime
    updated_at: datetime


def tenant_records(tenant_id: str) -> list[DocumentRecord]:
    """Scroll every record of one tenant, paging via point-id offset.

    Raises HTTPException 500 when Qdrant is unavailable (infrastructure
    outage — the one case ARCH allows a 500 for).
    """
    records: list[DocumentRecord] = []
    offset: Optional[str] = None
    while True:
        page = DocumentRecord.bulk_find(tenant_id, limit=_PAGE_SIZE, offset=offset)
        if page is None:
            raise HTTPException(
                status_code=500, detail="vector db unavailable: could not list documents"
            )
        records.extend(page)
        if len(page) < _PAGE_SIZE:
            return records
        offset = str(page[-1].id)


def _find_record(tenant_id: str, document_id: UUID) -> DocumentRecord:
    """Fetch one record scoped to its tenant. A wrong tenant and a wrong
    document id are indistinguishable — no tenant-existence oracle (R5)."""
    for record in tenant_records(tenant_id):
        if record.id == document_id:
            return record
    raise HTTPException(status_code=404, detail="document not found")


def _content_length(request: Request) -> Optional[int]:
    raw = request.headers.get("content-length")
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


@router.post("/documents", response_model=UploadResponse)
async def upload_document(
    request: Request, tenant_id: str = Path(min_length=1)
) -> UploadResponse:
    """Accept a multipart upload and run the full pipeline synchronously.

    Responds 200 with the *terminal* record — ``ready`` or ``failed`` with
    a reason (A1/N2). 413/422 are raised before any record is created.
    """
    settings = Settings()
    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024

    # Oversize gate 1 (N3): reject on Content-Length before reading the
    # body, so a huge upload is cut off before it finishes streaming.
    declared = _content_length(request)
    if declared is not None and declared > max_bytes + _MULTIPART_OVERHEAD_SLACK:
        logger.info(
            "Rejected upload (content-length %d) for tenant %s before reading body",
            declared,
            tenant_id,
        )
        raise HTTPException(
            status_code=413,
            detail=f"exceeds size limit: body larger than {settings.MAX_FILE_SIZE_MB} MB",
        )

    form = await request.form()
    upload = form.get("file")
    if not isinstance(upload, UploadFile):
        raise HTTPException(
            status_code=422, detail="no file part in the upload (field name: 'file')"
        )

    filename = upload.filename or ""
    if category_from_filename(filename) is None:
        suffix = "." + filename.rsplit(".", 1)[-1] if "." in filename else filename
        raise HTTPException(
            status_code=422,
            detail=f"unsupported format: {suffix!r} — supported: {_SUPPORTED_SUFFIXES}",
        )

    content = await upload.read()

    # Oversize gate 2 (N3): the authoritative check on the actual bytes,
    # covering chunked bodies that carry no Content-Length.
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"exceeds size limit: {len(content) / (1024 * 1024):.1f} MB > "
                f"{settings.MAX_FILE_SIZE_MB} MB"
            ),
        )

    # Upload flow (A11/A12): hash the bytes, persist the pending record,
    # process to a terminal state. Nothing is written to disk — the
    # pipeline runs on in-memory bytes only.
    content_hash = hashlib.sha256(content).hexdigest()
    record = DocumentRecord(tenant_id=tenant_id, filename=filename, content_hash=content_hash)
    if not DocumentRecord.bulk_insert([record]):
        raise HTTPException(
            status_code=500,
            detail="vector db unavailable: could not create the document record",
        )
    logger.info(
        "Upload accepted: tenant=%s filename=%s size=%d document=%s",
        tenant_id,
        filename,
        len(content),
        record.id,
    )

    try:
        terminal = IngestionPipeline().process(record=record, content=content)
    except Exception:
        # process() raises only when even `failed` could not be persisted —
        # an infrastructure outage (the one legitimate 500).
        logger.exception("Infrastructure failure while processing document %s", record.id)
        raise HTTPException(
            status_code=500,
            detail=(
                "infrastructure failure while processing — document status "
                "may be non-terminal; check via GET"
            ),
        ) from None

    return UploadResponse(
        document_id=terminal.id,
        status=terminal.status,
        reason=terminal.failure_reason,
        flags=terminal.flags,
        num_chunks=terminal.num_chunks,
    )


@router.get("/documents", response_model=list[DocumentSummary])
def list_documents(tenant_id: str = Path(min_length=1)) -> list[DocumentSummary]:
    """List this tenant's documents — statuses, flags, reasons (R3/R8)."""
    records = tenant_records(tenant_id)
    records.sort(key=lambda r: (r.created_at, str(r.id)), reverse=True)
    return [
        DocumentSummary(
            document_id=record.id,
            filename=record.filename,
            status=record.status,
            reason=record.failure_reason,
            flags=record.flags,
            created_at=record.created_at,
        )
        for record in records
    ]


@router.get("/documents/{document_id}", response_model=DocumentDetail)
def get_document(
    tenant_id: str = Path(min_length=1), document_id: UUID = Path()
) -> DocumentDetail:
    """Fetch one document's full record (R3)."""
    record = _find_record(tenant_id, document_id)
    return DocumentDetail(
        document_id=record.id,
        tenant_id=record.tenant_id,
        filename=record.filename,
        content_hash=record.content_hash,
        status=record.status,
        reason=record.failure_reason,
        flags=record.flags,
        num_chunks=record.num_chunks,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


@router.delete("/documents/{document_id}", status_code=204)
def delete_document(
    tenant_id: str = Path(min_length=1), document_id: UUID = Path()
) -> Response:
    """Delete a document immediately: record + chunks (R7). Idempotent —
    an unknown (or already deleted) id is still a 204."""
    IngestionPipeline().delete_document(tenant_id=tenant_id, document_id=document_id)
    return Response(status_code=204)
