"""Operator-facing document records — the status/lifecycle entity (R3)."""

from typing import Any, ClassVar, Optional

from pydantic import Field, model_validator

from app.domain.base.vector import VectorBaseDocument
from app.domain.enums import DocumentStatus


class DocumentRecord(VectorBaseDocument):
    """Record of one uploaded document. Payload-only point (dummy 1-dim
    vector, never similarity-searched).

    Lifecycle: ``pending -> processing -> ready`` or ``failed``; the ``ready``
    record is written only *after* its chunks are inserted (A8).
    """

    filename: str
    content_hash: str  # SHA-256 of the uploaded bytes — duplicate flagging (R8)
    status: DocumentStatus = DocumentStatus.PENDING
    failure_reason: Optional[str] = None
    flags: dict[str, Any] = Field(default_factory=dict)
    num_chunks: int = 0

    collection_name: ClassVar[str] = "documents"

    @model_validator(mode="after")
    def _failed_requires_reason(self) -> "DocumentRecord":
        if self.status == DocumentStatus.FAILED and not self.failure_reason:
            raise ValueError("failure_reason is required when status is 'failed'")
        return self
