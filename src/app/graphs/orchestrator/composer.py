"""Reply composition (CR-R9/R10/R11 + PE-R11, A6).

The shared terminal of the slice-1 reply paths. Direct and clarify replies
are already final — they pass through untouched. A workflow result is
composed conversationally by the ``generator`` from the outcome and the
history — never the raw payload (CR-R9); a failure becomes an apology
naming what failed, so the conversation can continue (CR-R11); a pending
approval becomes a deterministic template naming the request id — no LLM
call, testable by inspection (CR-R10, A6). Ultimate fallbacks are static
strings: degradation never terminates a turn without a reply (CR-N1).

``compose_final`` (slice 2) is the planner–executor path's terminal: one
``generator`` call over *all* step outputs — the last step's output is
never returned verbatim alone (PE-R11, enforced mechanically by the
verbatim guard) — and, on executor takeover, an honest partial report
listing what completed, what failed or was skipped, and what remained open
(PE-R8) instead of claiming success.
"""

import json
import logging
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from app.domain.enums import TurnOutcome
from app.graphs.base.state import WorkflowResult, WorkflowStatus
from app.graphs.orchestrator.executor import TodoList, output_text
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


# ---------------------------------------------------------------------------
# compose_final (T9's declared slice) — PE-R8/PE-R11
# ---------------------------------------------------------------------------

_FINAL_SYSTEM = (
    "You compose the single final customer-facing answer for a multi-step "
    "plan. Use every step's outcome to answer the customer's original "
    "request, concisely and in their language. Synthesize across steps — "
    "never dump raw JSON and never return one step's raw output as the "
    "whole answer."
)
_TAKEOVER_SYSTEM = (
    "You write honest partial-completion reports for a customer-support "
    "assistant: name what was completed, what failed or was skipped and "
    "why, and what remained unfinished. Never claim success for unfinished "
    "work. Keep it warm and brief."
)
_STATIC_FINAL_FALLBACK = (
    "I completed the steps of your request; the details are recorded here "
    "in our conversation."
)
_STATIC_TAKEOVER_FALLBACK = (
    "I could not finish everything you asked — some steps remain "
    "incomplete. The details are recorded here in our conversation."
)


def _step_lines(steps: list[dict[str, Any]]) -> str:
    """One deterministic line per step — statuses and outputs, no prose
    invention (what the generator synthesizes from)."""
    lines: list[str] = []
    for step in steps:
        line = f"step {step['number']} ({step['workflow']}): {step['status']}"
        if step.get("output") is not None:
            line += f" — {output_text(step['output'])}"
        lines.append(line)
    return "\n".join(lines)


def _deterministic_final(steps: list[dict[str, Any]]) -> str:
    """The verbatim-guard fallback: a static report naming every step's
    outcome (PE-R11 — better an honest listing than a raw dump)."""
    return (
        "Here is where your request stands:\n"
        + "\n".join(
            f"- step {s['number']} ({s['workflow']}): {s['status']}" for s in steps
        )
    )


async def compose_final(state: OrchestratorState) -> dict[str, Any]:
    """The slice-2 terminal: synthesize the final answer over all step
    outputs (PE-R11), or the honest partial report on takeover (PE-R8)."""
    plan = state.get("plan") or {}
    steps: list[dict[str, Any]] = plan.get("steps") or []
    takeover = bool(plan.get("takeover"))
    history = state.get("history") or []
    goal = plan.get("message") or state.get("message") or ""

    if takeover:
        todos = TodoList.model_validate({"items": plan.get("todos") or []})
        user = (
            f"The customer's request: {goal}\n"
            f"Step outcomes:\n{_step_lines(steps)}\n\n"
            f"Todos that remain OPEN (this work was NOT finished):\n"
            f"{todos.render()}\n\n"
            "Write the honest partial report."
        )
        reply = await _generate(
            _TAKEOVER_SYSTEM, user, _STATIC_TAKEOVER_FALLBACK, history
        )
    else:
        user = (
            f"The customer's request: {goal}\n"
            f"Step outcomes:\n{_step_lines(steps)}\n\n"
            "Write the single final answer."
        )
        reply = await _generate(
            _FINAL_SYSTEM, user, _STATIC_FINAL_FALLBACK, history
        )

    # PE-R11, enforced: the last produced output is never the reply
    # verbatim alone — degenerate generations fall back to the honest
    # deterministic listing.
    last_output = next(
        (s["output"] for s in reversed(steps) if s.get("output") is not None),
        None,
    )
    if last_output is not None and steps:
        verbatim_forms = {output_text(last_output).strip()}
        try:
            verbatim_forms.add(json.dumps(last_output, default=str).strip())
        except TypeError:
            pass
        if reply.strip() in verbatim_forms:
            reply = _deterministic_final(steps)

    return {"reply": reply, "outcome": TurnOutcome.MULTI_WORKFLOW}
