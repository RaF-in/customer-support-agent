"""Tenant-scoped conversational endpoints (T6, CR-R1/R2/R4/R5 over HTTP).

    POST /tenants/{t}/conversations                      → 201
    POST /tenants/{t}/conversations/{cid}/messages       → 200
    GET  /tenants/{t}/conversations/{cid}/messages       → 200

Deliberately thin (the ``api/approvals.py`` shape): parsing + error mapping
only — every semantic lives behind the conversation service. ``tenant_id``
comes from the URL path alone (A3/A4; no auth in MVP). Unknown and
cross-tenant conversation ids raise one byte-identical 404 raised in one
place in the service (no oracle, CR-R12/N3). Bad messages are 422 (the
strip-aware body validator here; the service re-validates — defense in
depth). Infrastructure failures surface as the single 500 the service
raises (CR-N1). ``outcome`` rides every turn response — R5's observability
lever — with the optional ``workflow`` name behind it.
"""

import logging
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Path
from pydantic import BaseModel, field_validator

from app.application.conversations import (
    ConversationNotFoundError,
    ConversationUnavailableError,
    InvalidMessageError,
    get_transcript as read_transcript,
    send_message,
    start_conversation,
)
from app.domain.conversations import ConversationMessageRecord
from app.domain.enums import MessageKind, MessageRole, TurnOutcome
from app.infrastructure.settings import Settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tenants/{tenant_id}", tags=["conversations"])


class MessageBody(BaseModel):
    """One user message — strip-aware, length-capped (CR edge-case table).

    Whitespace-only is rejected (there is nothing to route); the cap reads
    ``CONVERSATION_MAX_MESSAGE_CHARS`` at call time, so the setting stays a
    live tunable (A13) rather than an import-time constant.
    """

    message: str

    @field_validator("message")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be empty or whitespace-only")
        cap = Settings().CONVERSATION_MAX_MESSAGE_CHARS
        if len(value) > cap:
            raise ValueError(f"message exceeds the maximum of {cap} characters")
        return value


class TurnResponse(BaseModel):
    """One completed turn: the reply plus what produced it (CR-R1/R5)."""

    conversation_id: UUID
    message_id: UUID
    reply: str
    outcome: TurnOutcome
    workflow: Optional[str] = None


class MessageOut(BaseModel):
    """One transcript row in the declared read shape (CR-R4)."""

    seq: int
    role: MessageRole
    kind: MessageKind
    content: str
    created_at: datetime
    pending_request_id: Optional[UUID] = None
    data: Optional[dict[str, Any]] = None


class TranscriptResponse(BaseModel):
    conversation_id: UUID
    messages: list[MessageOut]


def _map_turn_errors(error: Exception) -> None:
    """Raise the HTTP mapping for one service turn error (single point)."""
    if isinstance(error, InvalidMessageError):
        raise HTTPException(status_code=422, detail=str(error)) from error
    if isinstance(error, ConversationNotFoundError):
        # Unknown and cross-tenant ids land here alike — byte-identical
        # (raised as one string in the service; no oracle, CR-R12).
        raise HTTPException(status_code=404, detail=str(error)) from error
    if isinstance(error, ConversationUnavailableError):
        raise HTTPException(status_code=500, detail=str(error)) from error
    raise error  # unknown — let FastAPI's handler have it (a bug, not a state)


@router.post("/conversations", status_code=201, response_model=TurnResponse)
async def start(
    body: MessageBody, tenant_id: str = Path(min_length=1)
) -> TurnResponse:
    """Start a conversation: the header is created implicitly by this first
    message and the new conversation id rides the 201 (CR-R2)."""
    try:
        result = await start_conversation(tenant_id, body.message)
    except (InvalidMessageError, ConversationNotFoundError,
            ConversationUnavailableError) as error:
        _map_turn_errors(error)
    return TurnResponse(
        conversation_id=result.conversation_id,
        message_id=result.message_id,
        reply=result.reply,
        outcome=result.outcome,
        workflow=getattr(result, "workflow", None),
    )


@router.post(
    "/conversations/{conversation_id}/messages",
    status_code=200,
    response_model=TurnResponse,
)
async def message(
    body: MessageBody,
    tenant_id: str = Path(min_length=1),
    conversation_id: UUID = Path(),
) -> TurnResponse:
    """One turn on an existing conversation (CR-R1). Unknown or cross-tenant
    ids → the same 404, never a silent new conversation (CR-R12)."""
    try:
        result = await send_message(tenant_id, conversation_id, body.message)
    except (InvalidMessageError, ConversationNotFoundError,
            ConversationUnavailableError) as error:
        _map_turn_errors(error)
    return TurnResponse(
        conversation_id=result.conversation_id,
        message_id=result.message_id,
        reply=result.reply,
        outcome=result.outcome,
        workflow=getattr(result, "workflow", None),
    )


@router.get(
    "/conversations/{conversation_id}/messages",
    response_model=TranscriptResponse,
)
async def transcript(
    tenant_id: str = Path(min_length=1),
    conversation_id: UUID = Path(),
) -> TranscriptResponse:
    """The read surface: one conversation's messages, chronological by
    ``seq`` (CR-R4) — the async-approval delivery path too (CR-R10)."""
    try:
        rows = await read_transcript(tenant_id, conversation_id)
    except (InvalidMessageError, ConversationNotFoundError,
            ConversationUnavailableError) as error:
        _map_turn_errors(error)
    return TranscriptResponse(
        conversation_id=conversation_id,
        messages=[_row_out(row) for row in rows],
    )


def _row_out(row: ConversationMessageRecord) -> MessageOut:
    return MessageOut(
        seq=row.seq,
        role=row.role,
        kind=row.kind,
        content=row.content,
        created_at=row.created_at,
        pending_request_id=row.pending_request_id,
        data=row.data,
    )
