"""Conversation turn service — the heart of slice 1 (CR-R1..R4, CR-N1/N2/N4).

Everything the graph must not touch lives here: turn serialization via
turn-spanning Postgres advisory locks (A17), history assembly for the graph
(full transcript by default, lever-capped, A14), the consecutive-clarify
streak that computes ``must_commit`` (A7's input), one-ACID-transaction
transcript persistence (A3 — reply computed before anything is stored, then
seq bump + inserts commit together), and the approval-decision hook that
appends completed outcomes to the linked conversation (A8, CR-R10 service
half).

Failure discipline (CR-N1): this module is the single catch point. Business
outcomes are values; every infrastructure exception — pool down, persist
failure, a graph crash — maps to ONE user-meaningful error
(``ConversationUnavailableError``). The graph's own degradation ladders mean
an exception from it is a bug, but even a bug surfaces as that single error,
never a partial turn.

Tenant isolation is structural (CR-R12/N3): every accessor call carries the
tenant id, and the header check ``WHERE id AND tenant_id`` makes an unknown
and a cross-tenant conversation id raise the byte-identical
``ConversationNotFoundError``. Record access goes through the T2 accessors'
raising twins (``_get_conversation`` & co. — the ``runtime.py`` /
``ApprovalRecord._retrieve_by_ids`` precedent): a fetch that breaks raises
into the catch point, while a fetch that finds nothing returns ``None`` —
which is exactly the no-oracle signal. The turn-spanning advisory lock is
the one declared additive T2 accessor (``conversation_turn_lock``).

Logs carry ids and statuses only — never message text, replies, or workflow
payloads (CR-N4); ``turn_id`` (the envelope ``run_id``) is the correlation
key.
"""

import logging
import time
import uuid
from typing import Any, Optional, Union
from uuid import UUID

from langgraph.graph.state import CompiledStateGraph

from app.domain import conversations as records
from app.domain.conversations import MessageDraft
from app.domain.enums import MessageKind, MessageRole, TurnOutcome
from app.graphs.base.state import WorkflowResult, WorkflowStatus
from app.graphs.orchestrator.catalog import build_catalog
from app.graphs.orchestrator.graph import build_orchestrator_graph
from app.graphs.orchestrator.state import TurnResult
from app.infrastructure.settings import Settings

logger = logging.getLogger(__name__)

# Byte-identical for unknown and cross-tenant ids alike — raised in one
# place, no tenant-existence oracle (CR-R12/N3).
_NOT_FOUND_DETAIL = "conversation not found"

# The single user-meaningful infrastructure error (CR-N1): one message for
# pool outages, persist failures, and graph crashes alike.
_UNAVAILABLE_DETAIL = (
    "conversation service is temporarily unavailable — please try again"
)


class ConversationNotFoundError(LookupError):
    """404-equivalent: unknown conversation id — or another tenant's, which
    is deliberately indistinguishable (no oracle, CR-R12/N3)."""


class InvalidMessageError(ValueError):
    """422-equivalent: empty/whitespace-only or over-length message."""


class ConversationUnavailableError(RuntimeError):
    """500-equivalent: the one user-meaningful infrastructure error every
    turn failure maps to (CR-N1)."""


class ConversationTurnResult(TurnResult):
    """``TurnResult`` + the workflow name behind the turn — the ARCH HTTP
    contract's ``workflow?`` observability field. ``state.py``'s
    ``TurnResult`` (T4's file) stays untouched; this subclass satisfies the
    library contract while letting the API surface name the workflow."""

    workflow: Optional[str] = None


# Compile-once orchestrator graph cache (the ``runtime._compiled_graphs``
# pattern — one build per process; the service is the graph's only invoker).
_orchestrator_graph: Optional[CompiledStateGraph] = None


def _get_orchestrator_graph() -> CompiledStateGraph:
    global _orchestrator_graph
    if _orchestrator_graph is None:
        _orchestrator_graph = build_orchestrator_graph()
    return _orchestrator_graph


def _validate_message(message: str) -> None:
    """Strip-aware turn-input validation (CR edge-case table, A13 cap)."""
    if not isinstance(message, str) or not message.strip():
        raise InvalidMessageError("message must not be empty or whitespace-only")
    cap = Settings().CONVERSATION_MAX_MESSAGE_CHARS
    if len(message) > cap:
        raise InvalidMessageError(
            f"message is {len(message)} characters — the maximum is {cap}"
        )


def _coerce_conversation_id(conversation_id: Union[str, UUID]) -> UUID:
    """Any unparseable id reads exactly like an unknown one (no oracle)."""
    try:
        return conversation_id if isinstance(conversation_id, UUID) else UUID(
            str(conversation_id)
        )
    except ValueError:
        raise ConversationNotFoundError(_NOT_FOUND_DETAIL) from None


def assemble_history(
    transcript: list["records.ConversationMessageRecord"],
) -> list[dict[str, Any]]:
    """Render the transcript as the graph's ``history`` (chronological; every
    transcript-visible entry included — outcomes are context too, CR-R3),
    capped to the last ``CONVERSATION_HISTORY_MAX_MESSAGES`` entries when the
    lever is set (0 = unlimited, A14)."""
    history = [
        {
            "role": row.role.value,
            "kind": row.kind.value,
            "content": row.content,
        }
        for row in transcript
    ]
    limit = Settings().CONVERSATION_HISTORY_MAX_MESSAGES
    if limit > 0:
        history = history[-limit:]
    return history


def compute_must_commit(
    transcript: list["records.ConversationMessageRecord"],
) -> bool:
    """True when the last ``CLARIFY_MAX_CONSECUTIVE`` replies were all
    clarifications — the next turn must commit instead of asking again (CR
    edge case; the deterministic override itself is the conditional edge)."""
    streak = 0
    for row in reversed(transcript):
        if row.role != MessageRole.ASSISTANT or row.kind != MessageKind.TEXT:
            continue  # user messages and outcome artifacts are not replies
        outcome = (row.data or {}).get("outcome")
        if outcome == TurnOutcome.CLARIFY.value:
            streak += 1
        else:
            break
    return streak >= Settings().CLARIFY_MAX_CONSECUTIVE


def _runs_payload(
    workflow: Optional[str], runs: list[WorkflowResult]
) -> list[dict[str, Any]]:
    """JSON-safe outcome payload for the transcript (``runs: [{workflow,
    status, output?, request_id?, error?}]`` — the slice-2 chaining and
    PE-R12 follow-up substrate from day one)."""
    return [
        {"workflow": workflow, **run.model_dump(mode="json")} for run in runs
    ]


def _outcome_summary_line(
    workflow: Optional[str], runs: list[WorkflowResult]
) -> str:
    """The outcome message's displayable one-liner (terse by design — the
    conversational reply is the assistant message, this is the artifact)."""
    named = workflow or "workflow"
    parts: list[str] = []
    for run in runs:
        if run.status == WorkflowStatus.COMPLETED:
            parts.append(f"{named} completed")
        elif run.status == WorkflowStatus.PENDING_APPROVAL:
            parts.append(f"{named} pending approval (request {run.request_id})")
        elif run.status == WorkflowStatus.REJECTED:
            parts.append(f"{named} rejected — nothing was executed")
        else:
            parts.append(f"{named} failed")
    return "; ".join(parts) or f"{named} ran"


async def _run_turn(
    tenant_id: str, conversation_id: UUID, message: str
) -> ConversationTurnResult:
    """One serialized turn under the advisory lock: header check (no oracle)
    → transcript → history/streak seeding → graph → ONE-transaction persist.
    Any exception outside the business errors maps to the single
    user-meaningful error (CR-N1)."""
    turn_id = str(uuid.uuid4())
    started = time.perf_counter()
    try:
        async with records.conversation_turn_lock(conversation_id):
            header = await records._get_conversation(tenant_id, conversation_id)
            if header is None:
                raise ConversationNotFoundError(_NOT_FOUND_DETAIL)
            # Raising twin: infra failures raise (caught below); no rows
            # legitimately means an empty transcript (first turn).
            transcript = await records._get_transcript(tenant_id, conversation_id)

            final = await _get_orchestrator_graph().ainvoke(
                {
                    "tenant_id": tenant_id,
                    "run_id": turn_id,
                    "errors": [],
                    "message": message,
                    "history": assemble_history(transcript),
                    "catalog": build_catalog(),
                    "must_commit": compute_must_commit(transcript),
                }
            )
            reply = str(final.get("reply") or "")
            outcome = TurnOutcome(final.get("outcome") or TurnOutcome.DIRECT)
            runs: list[WorkflowResult] = list(final.get("runs") or [])
            workflow = (final.get("classification") or {}).get("workflow")
            pending_request_id = final.get("pending_request_id")

            drafts = [
                MessageDraft(
                    role=MessageRole.USER, kind=MessageKind.TEXT, content=message
                ),
                MessageDraft(
                    role=MessageRole.ASSISTANT,
                    kind=MessageKind.TEXT,
                    content=reply,
                    data={"outcome": outcome.value, "workflow": workflow},
                    pending_request_id=(
                        UUID(str(pending_request_id))
                        if pending_request_id
                        else None
                    ),
                ),
            ]
            if runs:
                drafts.append(
                    MessageDraft(
                        role=MessageRole.ASSISTANT,
                        kind=MessageKind.WORKFLOW_OUTCOME,
                        content=_outcome_summary_line(workflow, runs),
                        data={
                            "outcome": outcome.value,
                            "runs": _runs_payload(workflow, runs),
                        },
                    )
                )
            rows = await records._append_messages(
                tenant_id, conversation_id, drafts
            )
            assistant_row = rows[1]  # [user, assistant(, outcome)]
    except (ConversationNotFoundError, InvalidMessageError):
        raise
    except Exception as error:
        logger.exception(
            "turn crashed (tenant=%s conversation=%s turn_id=%s)",
            tenant_id,
            conversation_id,
            turn_id,
        )
        raise ConversationUnavailableError(_UNAVAILABLE_DETAIL) from error

    duration_ms = int((time.perf_counter() - started) * 1000)
    logger.info(
        "turn tenant=%s conversation=%s turn_id=%s outcome=%s workflow=%s"
        " status=%s duration_ms=%d",
        tenant_id,
        conversation_id,
        turn_id,
        outcome.value,
        workflow or "-",
        runs[-1].status.value if runs else "-",
        duration_ms,
    )
    return ConversationTurnResult(
        conversation_id=conversation_id,
        message_id=assistant_row.id,
        reply=reply,
        outcome=outcome,
        runs=runs,
        workflow=workflow,
    )


async def start_conversation(
    tenant_id: str, message: str
) -> ConversationTurnResult:
    """Create the header implicitly (CR decision 5) and run the first turn.

    422-equivalent (``InvalidMessageError``) before anything is stored;
    infrastructure failures raise the single user-meaningful error.
    """
    _validate_message(message)
    try:
        header = await records._create_conversation(tenant_id)
    except Exception as error:
        logger.exception(
            "conversation creation failed (tenant=%s)", tenant_id
        )
        raise ConversationUnavailableError(_UNAVAILABLE_DETAIL) from error
    return await _run_turn(tenant_id, header.id, message)


async def send_message(
    tenant_id: str, conversation_id: Union[str, UUID], message: str
) -> ConversationTurnResult:
    """One serialized turn on an existing conversation.

    Unknown, malformed, and cross-tenant ids raise the byte-identical
    ``ConversationNotFoundError`` (no oracle, CR-R12); bad messages the
    ``InvalidMessageError``; infrastructure the single error (CR-N1).
    """
    _validate_message(message)
    return await _run_turn(
        tenant_id, _coerce_conversation_id(conversation_id), message
    )


async def get_transcript(
    tenant_id: str, conversation_id: Union[str, UUID]
) -> list["records.ConversationMessageRecord"]:
    """The R4 read surface: one conversation's messages, chronological by
    ``seq``. Same no-oracle and single-error discipline as turns."""
    cid = _coerce_conversation_id(conversation_id)
    try:
        header = await records._get_conversation(tenant_id, cid)
        if header is None:
            raise ConversationNotFoundError(_NOT_FOUND_DETAIL)
        return await records._get_transcript(tenant_id, cid)
    except ConversationNotFoundError:
        raise
    except Exception as error:
        logger.exception(
            "transcript read failed (tenant=%s conversation=%s)",
            tenant_id,
            cid,
        )
        raise ConversationUnavailableError(_UNAVAILABLE_DETAIL) from error


async def on_approval_decided(
    tenant_id: str, request_id: Union[str, UUID], result: WorkflowResult
) -> None:
    """The CR-R10 decision hook (slice-1 branch): append the completed
    outcome to the conversation whose assistant reply carries this
    ``pending_request_id`` — append-order, i.e. after any messages the user
    sent while waiting.

    Failures are logged and swallowed, never raised: the decision response
    is already committed truth, and the approval + action records remain the
    durable audit (A8).
    """
    try:
        request = request_id if isinstance(request_id, UUID) else UUID(
            str(request_id)
        )
    except ValueError:
        logger.warning(
            "approval_outcome tenant=%s request_id=%s appended=no (unparseable"
            " id)",
            tenant_id,
            request_id,
        )
        return
    try:
        linked = await records._find_message_by_pending_request(
            tenant_id, request
        )
        if linked is None:
            # No conversational turn introduced this request (a direct
            # action-run did) — nothing to append, by design.
            logger.info(
                "approval_outcome tenant=%s request_id=%s conversation=-"
                " appended=no-link",
                tenant_id,
                request,
            )
            return
        workflow = (linked.data or {}).get("workflow")
        draft = MessageDraft(
            role=MessageRole.ASSISTANT,
            kind=MessageKind.WORKFLOW_OUTCOME,
            content=_outcome_summary_line(workflow, [result]),
            data={
                "outcome": TurnOutcome.WORKFLOW.value,
                "runs": _runs_payload(workflow, [result]),
            },
        )
        rows = await records._append_messages(
            tenant_id, linked.conversation_id, [draft]
        )
        logger.info(
            "approval_outcome tenant=%s request_id=%s conversation=%s"
            " appended=%s",
            tenant_id,
            request,
            linked.conversation_id,
            rows is not None,
        )
    except Exception:
        # A8: isolated — never change the decision response.
        logger.exception(
            "approval outcome append failed (tenant=%s request_id=%s)",
            tenant_id,
            request,
        )
