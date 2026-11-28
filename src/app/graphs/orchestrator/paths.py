"""Slice-1 reply paths (CR-R6/R8, A6).

``direct_reply`` — a ``generator``-purpose reply from the conversation
history, with a static fallback when the generator itself fails: the user
always gets a reply and no exception escapes the graph (CR-N1/R13).
``clarify_reply`` — the classification's clarifying question becomes the
reply verbatim; no extra LLM call (A6). ``run_workflow`` — exactly one
facade call with the classification's name and input; the facade validates
and runs and maps infrastructure failures, so the node only forwards the
``WorkflowResult`` (LLM output proposes, the runtime disposes — PE-N3).
"""

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.domain.enums import TurnOutcome
from app.graphs.orchestrator.state import OrchestratorState
from app.graphs.runtime import run_workflow

from app.infrastructure.llm.factory import get_chat_model

logger = logging.getLogger(__name__)

_STATIC_DIRECT_FALLBACK = (
    "I'm here, but I could not compose a reply just now. "
    "Could you rephrase that, or try again in a moment?"
)

# History is rendered once, by the conversation service (conversation-
# compaction A2); the node only embeds it. Absence — the plan-resume path
# seeds none — reads exactly like a new conversation.
_HISTORY_START = "(conversation start)"

_DIRECT_SYSTEM = (
    "You are a customer-support assistant. Reply to the customer's latest "
    "message concisely and warmly, in the same language as the message, "
    "using the conversation so far for context."
)


async def direct_reply_node(state: OrchestratorState) -> dict[str, Any]:
    """Small talk / meta questions / out-of-contract fallback (CR-R6)."""
    model = get_chat_model("generator")
    try:
        message = await model.ainvoke(
            [
                SystemMessage(_DIRECT_SYSTEM),
                HumanMessage(
                    f"Conversation so far:\n"
                    f"{state.get('history_text') or _HISTORY_START}\n\n"
                    f"Customer message: {state['message']}"
                ),
            ]
        )
        reply = (
            message.content
            if isinstance(message.content, str)
            else str(message.content)
        )
    except Exception:
        logger.warning(
            "direct reply generation failed; static fallback used",
            exc_info=True,
        )
        reply = _STATIC_DIRECT_FALLBACK
    return {"reply": reply, "outcome": TurnOutcome.DIRECT, "runs": []}


async def clarify_reply_node(state: OrchestratorState) -> dict[str, Any]:
    """The classification's question is the reply, verbatim (CR-R8, A6)."""
    classification = state.get("classification") or {}
    question = (classification.get("clarifying_question") or "").strip()
    return {"reply": question, "outcome": TurnOutcome.CLARIFY, "runs": []}


async def run_workflow_node(state: OrchestratorState) -> dict[str, Any]:
    """One workflow run through the facade (CR-R5 path half).

    The router already validated name and input semantically; the facade
    re-validates before anything runs (it is the disposing side). Business
    outcomes — including ``failed`` — are values carried in ``runs``;
    nothing raises here on infrastructure errors (the facade caught them).
    """
    classification = state.get("classification") or {}
    workflow = classification.get("workflow")
    result = await run_workflow(
        workflow, state["tenant_id"], classification.get("workflow_input") or {}
    )
    logger.info(
        "workflow path tenant=%s turn_id=%s workflow=%s status=%s",
        state["tenant_id"],
        state["run_id"],
        workflow,
        result.status.value,
    )
    return {"runs": [result], "outcome": TurnOutcome.WORKFLOW}
