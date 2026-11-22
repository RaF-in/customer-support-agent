"""Conversation turn service — the heart of slices 1 + 2 (CR-R1..R4,
CR-N1/N2/N4, PE-R10/R12).

Everything the graph must not touch lives here: turn serialization via
turn-spanning Postgres advisory locks (A17), history assembly for the graph
(full transcript by default, lever-capped, A14), the consecutive-clarify
streak that computes ``must_commit`` (A7's input), one-ACID-transaction
transcript persistence (A3 — reply computed before anything is stored, then
seq bump + inserts commit together), and the approval-decision hook that
appends completed outcomes to the linked conversation (A8, CR-R10 service
half).

The slice-2 half (T10) persists the orchestration artifacts (PE-R12): a
multi-workflow turn appends the validated-plan snapshot (a ``plan``-kind
message) and writes the mutable ``PlanRecord`` (status, steps, todos,
pause linkage) beside the turn's transcript rows — and the approval-decision
hook grew the resume branch: a decision on a paused plan's request re-seeds
the graph's resume entry from the persisted ``PlanRecord`` (A9 — no
checkpointer; the row survives redeploys), injects the decision outcome as
the paused step's result (an approved action already executed inside
``decide_approval`` — the step is never re-run), runs the remainder under
the same turn machinery (advisory lock, one-tx append), and appends the
final answer after any newer messages (append-order).

Failure discipline (CR-N1): this module is the single catch point. Business
outcomes are values; every infrastructure exception — pool down, persist
failure, a graph crash — maps to ONE user-meaningful error
(``ConversationUnavailableError``). The graph's own degradation ladders mean
an exception from it is a bug, but even a bug surfaces as that single error,
never a partial turn.

Tenant isolation is structural (CR-R12/N3, PE-N2): every accessor call
carries the tenant id, and the header check ``WHERE id AND tenant_id``
makes an unknown and a cross-tenant conversation id raise the byte-identical
``ConversationNotFoundError``. Record access goes through the T2 accessors'
raising twins (``_get_conversation`` & co. — the ``runtime.py`` /
``domain/approvals.py`` raising-accessor precedent): a fetch that breaks
raises into the catch point, while a fetch that finds nothing returns
``None`` — which is exactly the no-oracle signal. The turn-spanning advisory
lock is the one declared additive T2 accessor (``conversation_turn_lock``).

Logs carry ids and statuses only — never message text, replies, workflow
payloads, or plan/todo content (CR-N4, PE-N4); ``turn_id`` (the envelope
``run_id``) is the correlation key.
"""

import logging
import time
import uuid
from typing import Any, Optional, Union
from uuid import UUID

from langgraph.graph.state import CompiledStateGraph

from app.domain import conversations as records
from app.domain import plans
from app.domain.conversations import MessageDraft
from app.domain.enums import (
    MessageKind,
    MessageRole,
    PlanStatus,
    TurnOutcome,
)
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


# ---------------------------------------------------------------------------
# Slice-2 helpers — plan artifacts + the mutable PlanRecord (T10, PE-R10/R12)
# ---------------------------------------------------------------------------


def _orchestrator_recursion_limit() -> int:
    """The orchestrator's own superstep budget, derived from the configured
    cycle bounds (the runtime-facade pattern — every cycle's real bound is
    its state counter; this is only the global backstop)."""
    settings = Settings()
    return (
        2 * settings.PLANNER_MAX_ATTEMPTS
        + 2 * (settings.EXECUTOR_MAX_REINVOCATIONS + 1)
        + 24
    )


def _plan_summary_line(plan_state: dict[str, Any]) -> str:
    """The plan-outcome artifact's one-liner — per-step workflows and their
    statuses (ids and statuses; the conversational reply carries the prose)."""
    parts: list[str] = []
    for step in plan_state.get("steps") or []:
        part = f"{step['workflow']} {step['status']}"
        if step["status"] == "pending" and plan_state.get("pending_request_id"):
            part += f" (request {plan_state['pending_request_id']})"
        parts.append(part)
    return "; ".join(parts) or "plan ran"


def _plan_steps_payload(plan_state: dict[str, Any]) -> list[dict[str, Any]]:
    """JSON-safe per-step outcome payload (PE-R12 — the artifact a later
    turn or an auditor reads; outputs are the workflows' own)."""
    return [
        {
            "step": step["number"],
            "workflow": step["workflow"],
            "status": step["status"],
            "output": step.get("output"),
        }
        for step in plan_state.get("steps") or []
    ]


def _plan_snapshot_payload(plan_state: dict[str, Any]) -> dict[str, Any]:
    """The immutable validated-plan snapshot (``plan``-kind message data):
    intent only — numbers, workflows, declared inputs with their
    ``$$STEP_X_OUTPUT$$`` tokens; what actually ran lives in the outcome
    artifacts and the PlanRecord."""
    return {
        "steps": [
            {
                "step": step["number"],
                "workflow": step["workflow"],
                "input": step["input"],
            }
            for step in plan_state.get("steps") or []
        ]
    }


async def _persist_plan_record(
    tenant_id: str, conversation_id: UUID, plan_state: dict[str, Any]
) -> Optional[UUID]:
    """Write the turn's final plan state as the mutable ``PlanRecord``
    (create → update: the T2 accessors' two-step path to a non-default
    status/todos/pause linkage). Failures are logged — the transcript rows
    are already the committed truth of this turn (A8-adjacent)."""
    try:
        record = await plans.create_plan(
            tenant_id,
            conversation_id,
            [plans.PlanStep.model_validate(s) for s in plan_state["steps"]],
        )
        if record is None:
            logger.warning(
                "plan record not persisted (tenant=%s conversation=%s "
                "create_failed=1)",
                tenant_id,
                conversation_id,
            )
            return None
        record.status = PlanStatus(
            plan_state.get("status") or PlanStatus.EXECUTING.value
        )
        record.steps = [plans.PlanStep.model_validate(s) for s in plan_state["steps"]]
        record.todos = [plans.Todo.model_validate(t) for t in plan_state.get("todos") or []]
        pending = plan_state.get("pending_request_id")
        record.pending_request_id = UUID(str(pending)) if pending else None
        updated = await plans.update_plan(record)
        if updated is None:
            logger.warning(
                "plan record not finalized (tenant=%s plan_id=%s update_failed=1)",
                tenant_id,
                record.id,
            )
        logger.info(
            "plan tenant=%s conversation=%s plan_id=%s status=%s steps=%d"
            " todos=%d",
            tenant_id,
            conversation_id,
            record.id,
            record.status.value,
            len(record.steps),
            len(record.todos),
        )
        return record.id
    except Exception:
        logger.exception(
            "plan record persistence crashed (tenant=%s conversation=%s)",
            tenant_id,
            conversation_id,
        )
        return None


def _resolve_paused_step(
    steps: list[dict[str, Any]],
    todos: list[dict[str, Any]],
    result: WorkflowResult,
) -> None:
    """Inject the human decision as the paused step's outcome (PE-R10 / PE
    decision 8): the approved action already executed inside
    ``decide_approval`` — the step is settled from its result, never re-run;
    a rejection or a failed action counts as the step failing. The step's
    todo closes to match (completed on success; abandoned with the
    decision's reason otherwise)."""
    paused = next((s for s in steps if s["status"] == "pending"), None)
    if paused is None:
        return
    if result.status == WorkflowStatus.COMPLETED:
        paused["status"] = "completed"
        paused["output"] = result.output
        reason = None
    else:
        paused["status"] = "failed"
        reason = (
            "approval rejected"
            if result.status == WorkflowStatus.REJECTED
            else f"workflow failed: {result.error or '(no detail)'}"
        )
    todo = next((t for t in todos if t["id"] == paused["number"]), None)
    if todo is None or todo["status"] in ("completed", "abandoned"):
        return
    if reason is None:
        todo["status"] = "completed"
    else:
        todo["status"] = "abandoned"
        todo["abandon_reason"] = reason


async def _resume_paused_plan(
    tenant_id: str,
    request: UUID,
    record: "plans.PlanRecord",
    result: WorkflowResult,
) -> None:
    """The PE-R10 resume: re-seed the graph's resume entry from the paused
    ``PlanRecord`` (A9 — Postgres is the state, no checkpointer), run the
    remainder under the same turn machinery, append the final answer
    (append-order — after any newer messages), and finalize the record.
    Failures are logged and swallowed (A8): the decision + action records
    are already the durable truth."""
    try:
        async with records.conversation_turn_lock(record.conversation_id):
            # Lock → re-check → act (the runtime's decide_approval twin):
            # a second decision, or a resume already in flight, reads here.
            fresh = await plans.get_plan(tenant_id, record.id)
            if fresh is None or fresh.status != PlanStatus.PAUSED:
                logger.info(
                    "approval_outcome tenant=%s request_id=%s conversation=%s"
                    " appended=no (plan not paused)",
                    tenant_id,
                    request,
                    record.conversation_id,
                )
                return
            linked = await records._find_message_by_pending_request(
                tenant_id, request
            )
            transcript = await records._get_transcript(
                tenant_id, fresh.conversation_id
            )
            goal = ""
            if linked is not None:
                prior = [
                    row
                    for row in transcript
                    if row.role == MessageRole.USER and row.seq < linked.seq
                ]
                goal = prior[-1].content if prior else ""
            steps = [step.model_dump() for step in fresh.steps]
            todos = [todo.model_dump() for todo in fresh.todos]
            _resolve_paused_step(steps, todos, result)
            seed = {
                "message": goal,
                "status": PlanStatus.EXECUTING.value,
                "steps": steps,
                "todos": todos,
                "pending_request_id": None,
                "reinvocations": 0,
                "takeover": False,
            }
            resume_id = str(uuid.uuid4())
            final = await _get_orchestrator_graph().ainvoke(
                {
                    "tenant_id": tenant_id,
                    "run_id": resume_id,
                    "errors": [],
                    "plan": seed,
                },
                config={"recursion_limit": _orchestrator_recursion_limit()},
            )
            reply = str(final.get("reply") or "")
            runs: list[WorkflowResult] = list(final.get("runs") or [])
            final_plan = final.get("plan") or seed
            pending = final_plan.get("pending_request_id")
            drafts = [
                MessageDraft(
                    role=MessageRole.ASSISTANT,
                    kind=MessageKind.WORKFLOW_OUTCOME,
                    content=_plan_summary_line(final_plan),
                    data={
                        "outcome": TurnOutcome.MULTI_WORKFLOW.value,
                        "runs": _plan_steps_payload(final_plan),
                    },
                ),
                MessageDraft(
                    role=MessageRole.ASSISTANT,
                    kind=MessageKind.TEXT,
                    content=reply,
                    data={"outcome": TurnOutcome.MULTI_WORKFLOW.value},
                    pending_request_id=UUID(str(pending)) if pending else None,
                ),
            ]
            appended = await records._append_messages(
                tenant_id, fresh.conversation_id, drafts
            )
            fresh.status = PlanStatus(
                final_plan.get("status") or PlanStatus.DONE.value
            )
            fresh.steps = [
                plans.PlanStep.model_validate(s)
                for s in final_plan.get("steps") or steps
            ]
            fresh.todos = [
                plans.Todo.model_validate(t)
                for t in final_plan.get("todos") or todos
            ]
            fresh.pending_request_id = UUID(str(pending)) if pending else None
            await plans.update_plan(fresh)
            logger.info(
                "approval_outcome tenant=%s request_id=%s conversation=%s"
                " appended=%s plan_status=%s",
                tenant_id,
                request,
                fresh.conversation_id,
                appended is not None,
                fresh.status.value,
            )
    except Exception:
        # A8: isolated — never change the decision response.
        logger.exception(
            "plan resume failed (tenant=%s request_id=%s)",
            tenant_id,
            request,
        )


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
                },
                config={"recursion_limit": _orchestrator_recursion_limit()},
            )
            reply = str(final.get("reply") or "")
            outcome = TurnOutcome(final.get("outcome") or TurnOutcome.DIRECT)
            runs: list[WorkflowResult] = list(final.get("runs") or [])
            workflow = (final.get("classification") or {}).get("workflow")
            pending_request_id = final.get("pending_request_id")
            plan_state = final.get("plan") or None
            slice2 = bool(
                plan_state and plan_state.get("steps")
                and outcome == TurnOutcome.MULTI_WORKFLOW
            )

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
            if slice2:
                # PE-R12 artifacts: the immutable validated-plan snapshot,
                # then the per-step outcome line (the plan is the intent,
                # the steps are what actually happened).
                drafts.append(
                    MessageDraft(
                        role=MessageRole.ASSISTANT,
                        kind=MessageKind.PLAN,
                        content=_plan_summary_line(plan_state),
                        data={
                            "outcome": outcome.value,
                            "plan": _plan_snapshot_payload(plan_state),
                        },
                    )
                )
                drafts.append(
                    MessageDraft(
                        role=MessageRole.ASSISTANT,
                        kind=MessageKind.WORKFLOW_OUTCOME,
                        content=_plan_summary_line(plan_state),
                        data={
                            "outcome": outcome.value,
                            "runs": _plan_steps_payload(plan_state),
                        },
                    )
                )
            elif runs:
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
            assistant_row = rows[1]  # [user, assistant(, plan, outcome)]
            if slice2:
                await _persist_plan_record(
                    tenant_id, conversation_id, plan_state
                )
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
    """The decision hook — both slices (CR-R10 / PE-R10).

    Slice 2 first: a paused ``PlanRecord`` linked to this request is resumed
    — the remainder runs from the paused step and the final answer is
    appended (append-order). Otherwise the slice-1 branch: append the
    completed outcome to the conversation whose assistant reply carries
    this ``pending_request_id`` — after any messages the user sent while
    waiting.

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
        paused_plan = await plans.get_plan_by_pending_request(tenant_id, request)
        if paused_plan is not None:
            await _resume_paused_plan(tenant_id, request, paused_plan, result)
            return
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
