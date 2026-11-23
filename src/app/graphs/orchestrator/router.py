"""Route node — structured-output classification (A5, CR-R5/R13).

The router resolves ``get_chat_model("router")`` lazily *per call*
(graders.py style — that is what makes the ``set_chat_model`` seam control
every decision) and wraps it in ``with_structured_output`` so the decision
is a validated ``RouterDecision``, never free text. Out-of-contract output
is handled by a robustness ladder, not by raising:

1. semantic validation — the named workflow must exist in the registry and
   ``workflow_input`` must pass its input schema (the same validation the
   facade applies later; the router only *proposes* runs, the facade
   disposes — PE-N3);
2. one retry with the collected errors fed back;
3. a deterministic direct fallback — classification never raises, never
   returns an out-of-contract decision, and an exception never escapes the
   node (CR-N1).

``must_commit`` is NOT acted on here: the T4 conditional edge applies the
deterministic clarify→direct override (A7). The node only reads message,
history, and catalog.
"""

import logging
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from app.domain.enums import TurnOutcome
from app.graphs.orchestrator.catalog import build_catalog
from app.graphs.orchestrator.state import OrchestratorState, RouterDecision
from app.graphs.registry import get_workflow
from app.infrastructure.llm.factory import get_chat_model

logger = logging.getLogger(__name__)

# Initial attempt + exactly one retry (A5 — "retry ×1").
_MAX_ATTEMPTS = 2

_SYSTEM_PROMPT = """\
You are the router of a customer-support conversational assistant. Classify \
the user's latest message into exactly one outcome:
- direct: small talk, greetings, or a meta question about this assistant
- clarify: the request is genuinely ambiguous and one short question \
(returned verbatim as the reply) would resolve it
- workflow: a single workflow run answers the request (name it and its input)
- multi_workflow: the request is compound and needs a sequence of workflows

{catalog}

Return the structured decision only."""


def _history_text(history: list[dict[str, Any]]) -> str:
    """Render the transcript slice the router sees (chronological)."""
    if not history:
        return "(conversation start)"
    lines = [
        f"[{entry.get('role', '?')}] {entry.get('content', '')}"
        for entry in history
    ]
    return "\n".join(lines)


# The long-term-memory block (LM v1): the owner's fact card, service-seeded.
# Facts are customer data rendered inside a labeled block — the same trust
# treatment as transcript text, never instructions (LM ARCH security note).
_MEMORY_BLOCK_HEADER = "What you already know about this customer:"


def _memory_text(memories: Optional[list[str]]) -> str:
    """Render the memory block's body — absence reads neutrally, byte-identical
    for an empty card, a missing key, and an anonymous turn (LM-R5/R7)."""
    if not memories:
        return "(nothing known yet)"
    return "\n".join(f"- {fact}" for fact in memories)


def semantic_validation_errors(decision: RouterDecision) -> list[str]:
    """Out-of-contract checks the structured schema cannot express.

    Empty result = in contract. ``multi_workflow`` is deliberately
    unconstrained here: the slice-2 planner re-validates everything it
    proposes, and until slice 2 lands the conditional edge degrades it to
    the direct fallback (ARCH rollout note).
    """
    errors: list[str] = []
    if decision.outcome == TurnOutcome.WORKFLOW:
        if not decision.workflow:
            return ["outcome 'workflow' requires a workflow name"]
        try:
            definition = get_workflow(decision.workflow)
        except ValueError:
            return [f"unknown workflow {decision.workflow!r} — pick one from the catalog"]
        if decision.workflow_input is None:
            errors.append("outcome 'workflow' requires workflow_input")
        else:
            try:
                definition.input_schema.model_validate(decision.workflow_input)
            except ValidationError as error:
                errors.append(
                    f"workflow_input violates {decision.workflow}'s input "
                    f"schema: {error}"
                )
    elif decision.outcome == TurnOutcome.CLARIFY:
        if not (decision.clarifying_question or "").strip():
            errors.append(
                "outcome 'clarify' requires a non-empty clarifying_question"
            )
    return errors


def _fallback() -> RouterDecision:
    """The deterministic out-of-contract answer (CR-R13): a direct reply."""
    return RouterDecision(outcome=TurnOutcome.DIRECT)


async def _ask(messages: list[Any]) -> RouterDecision:
    """One structured-output classification attempt (lazy per-call model)."""
    model = get_chat_model("router").with_structured_output(RouterDecision)
    return await model.ainvoke(messages)


async def classify(
    message: str,
    history: list[dict[str, Any]],
    catalog: str,
    memories: Optional[list[str]] = None,
) -> tuple[RouterDecision, int]:
    """Classify one turn — returns ``(decision, attempts)``.

    ``attempts`` never exceeds 2 (one retry). The returned decision is
    always in contract: either a validated classification or the direct
    fallback. Never raises. ``memories`` is the service-seeded fact card
    (LM v1): durable knowledge about the customer that crosses
    conversations, rendered as its own labeled block.
    """
    base_messages = [
        SystemMessage(_SYSTEM_PROMPT.format(catalog=catalog)),
        HumanMessage(
            f"Conversation so far:\n{_history_text(history)}\n\n"
            f"{_MEMORY_BLOCK_HEADER}\n{_memory_text(memories)}\n\n"
            f"User message: {message}"
        ),
    ]
    attempts = 0
    feedback: list[str] = []
    while attempts < _MAX_ATTEMPTS:
        attempts += 1
        prompt = list(base_messages)
        if feedback:
            prompt.append(
                HumanMessage(
                    "Your previous attempt was invalid: "
                    + "; ".join(feedback)
                    + ". Return a corrected decision."
                )
            )
        try:
            decision = await _ask(prompt)
            feedback = semantic_validation_errors(decision)
            if not feedback:
                return decision, attempts
        except Exception as error:  # unparseable/garbage output — handled class
            feedback = [f"unparseable output ({type(error).__name__}: {error})"]
    return _fallback(), attempts


async def route(state: OrchestratorState) -> dict[str, Any]:
    """Graph node: classify the turn and store the decision (JSON-safe)."""
    catalog = state.get("catalog") or build_catalog()
    decision, attempts = await classify(
        state["message"],
        state.get("history") or [],
        catalog,
        memories=state.get("memories"),
    )
    logger.info(
        "route tenant=%s turn_id=%s outcome=%s workflow=%s attempt=%d",
        state["tenant_id"],
        state["run_id"],
        decision.outcome.value,
        decision.workflow,
        attempts,
    )
    return {"classification": decision.model_dump(mode="json")}
