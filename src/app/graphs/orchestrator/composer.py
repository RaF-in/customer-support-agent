"""Reply composition (CR-R9/R10/R11, A6).

The shared terminal of the slice-1 reply paths. Direct and clarify replies
are already final — they pass through untouched. A workflow result is
composed conversationally by the ``generator`` from the outcome and the
history — never the raw payload (CR-R9); a failure becomes an apology
naming what failed, so the conversation can continue (CR-R11); a pending
approval becomes a deterministic template naming the request id — no LLM
call, testable by inspection (CR-R10, A6). Ultimate fallbacks are static
strings: degradation never terminates a turn without a reply (CR-N1).
"""

import json
import logging
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from app.graphs.base.state import WorkflowResult, WorkflowStatus
from app.graphs.orchestrator.state import OrchestratorState
from app.infrastructure.llm.factory import get_chat_model

logger = logging.getLogger(__name__)

_STATIC_COMPOSED_FALLBACK = (
    "I completed the requested workflow, but I could not compose a summary "
    "just now. The outcome is recorded in our conversation."
)
_STATIC_APOLOGY_FALLBACK = (
    "I'm sorry — the requested workflow failed, so I could not complete "
    "your request. Please try again in a moment."
)

_COMPOSE_SYSTEM = (
    "You compose conversational customer-support replies from workflow "
    "outcomes. Use the workflow's result to answer or help the customer, "
    "concisely and in their language. Never dump raw JSON or the raw "
    "payload — say what it means for the customer."
)
_APOLOGY_SYSTEM = (
    "You write short, honest apologies for a customer-support assistant: "
    "name what failed, do not invent details, and invite the customer to "
    "try again. Keep it warm and brief."
)


def _pending_template(workflow: Optional[str], request_id: Optional[str]) -> str:
    """Deterministic waiting reply naming the request (CR-R10, A6)."""
    named = f"the {workflow} workflow" if workflow else "this request"
    return (
        f"I need your approval before I can continue: {named} is waiting on "
        f"approval request {request_id}. Once it is decided, the outcome "
        "will appear here in our conversation."
    )


def _rejected_template(workflow: Optional[str]) -> str:
    named = f"the {workflow} workflow" if workflow else "the request"
    return (
        f"I could not complete your request: approval for {named} was "
        "rejected, so nothing was executed."
    )


async def _generate(
    system: str, user: str, fallback: str, history: list[dict[str, Any]]
) -> str:
    """One generator call with a static fallback (never raises)."""
    model = get_chat_model("generator")
    history_text = "\n".join(
        f"[{entry.get('role', '?')}] {entry.get('content', '')}"
        for entry in history
    )
    try:
        message = await model.ainvoke(
            [
                SystemMessage(system),
                HumanMessage(f"Conversation so far:\n{history_text}\n\n{user}"),
            ]
        )
        return (
            message.content
            if isinstance(message.content, str)
            else str(message.content)
        )
    except Exception:
        logger.warning("reply composition failed; static fallback used",
                       exc_info=True)
        return fallback


async def compose_reply(state: OrchestratorState) -> dict[str, Any]:
    """The slice-1 compose node: turns this turn's outcome into the reply.

    Direct/clarify turns already carry a final ``reply`` — pass through.
    Workflow turns branch on the ``WorkflowResult``: pending approval →
    deterministic template (+ the linkage key for persistence, CR-R10);
    failed → apology naming the workflow; completed → conversational
    composition of the output.
    """
    runs: list[WorkflowResult] = state.get("runs") or []
    if not runs:
        return {}
    result = runs[-1]
    classification = state.get("classification") or {}
    workflow = classification.get("workflow")
    history = state.get("history") or []

    if result.status == WorkflowStatus.PENDING_APPROVAL:
        return {
            "reply": _pending_template(workflow, result.request_id),
            "pending_request_id": result.request_id,
        }
    if result.status == WorkflowStatus.FAILED:
        reply = await _generate(
            _APOLOGY_SYSTEM,
            f"The workflow {workflow!r} failed with this error: "
            f"{result.error or '(no detail)'}. Write the apology.",
            _STATIC_APOLOGY_FALLBACK,
            history,
        )
        return {"reply": reply}
    if result.status == WorkflowStatus.REJECTED:
        return {"reply": _rejected_template(workflow)}

    reply = await _generate(
        _COMPOSE_SYSTEM,
        f"The workflow {workflow!r} completed with this outcome:\n"
        f"{json.dumps(result.output, default=str)}\n\n"
        f"Customer's message: {state['message']}\n"
        "Write the reply.",
        _STATIC_COMPOSED_FALLBACK,
        history,
    )
    return {"reply": reply}


# Node name used by the topology (T9 adds compose_final beside this).
compose = compose_reply
