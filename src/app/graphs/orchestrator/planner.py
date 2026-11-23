"""Planner — plan generation + validation (A10, PE-R2/R3/R4/R13).

The ``planner``-purpose LLM (temperature 0, lazily resolved per call — the
seam controls every plan) emits the JSON plan contract in the
``specs/context.txt`` style: capabilities block, JSON-list instructions,
worked example. Validation is a pure function over the emitted *text* —
malformed JSON is a handled validation class, never an escaping exception —
rejecting with five distinguishable error classes (PE-R3): malformed JSON,
unknown workflow, placeholder referencing a nonexistent-or-later step, zero
steps, and more than ``PLAN_MAX_STEPS`` steps, plus the structural
per-step contract (sequential numbering, object inputs). Invalid plans are
fed back for correction within ``PLANNER_MAX_ATTEMPTS``; exhaustion yields
an apologetic direct answer and **nothing executes** (PE-R4) — this module
never runs workflows; it only proposes steps the executor (T9) triggers
through the facade (PE-N3).

The validated plan is a JSON-safe ``list[dict]`` (``{"step", "workflow",
"input"}``) — deliberately not a domain record: the orchestrator package
persists nothing (Module Boundaries); the service layer (T10) writes the
``PlanRecord`` and the transcript snapshot.

T9 wires the per-attempt primitives (``generate_plan_text`` +
``validate_plan``) into its graph cycle with its own attempt counter;
``build_plan`` is the standalone loop form for direct callers.
"""

import json
import logging
import re
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from app.graphs.orchestrator.catalog import build_catalog
from app.graphs.registry import WORKFLOW_REGISTRY
from app.infrastructure.llm.factory import get_chat_model
from app.infrastructure.settings import Settings

logger = logging.getLogger(__name__)

# $$STEP_X_OUTPUT$$ — an input value referencing step X's output (A12's
# dependency rule: step B depends on A iff B's inputs contain this token).
_PLACEHOLDER_RE = re.compile(r"\$\$STEP_(\d+)_OUTPUT\$\$")

# The deterministic exhaustion outcome (PE-R4): apologetic, honest that
# nothing executed, and the conversation continues.
_EXHAUSTION_FALLBACK = (
    "I'm sorry — I could not put together a reliable plan for this request, "
    "so nothing was executed. Could you rephrase it, or break it into "
    "smaller steps?"
)
_EXHAUSTION_SYSTEM = (
    "You write short, honest apologies for a customer-support assistant: "
    "the assistant could not plan the compound request, nothing was "
    "executed, and the customer is invited to rephrase or split the "
    "request. Do not invent details. Keep it warm and brief."
)

_SYSTEM_TEMPLATE = """\
You are the strategic planner of a customer-support assistant. Analyze the \
user's compound goal and create a structured Execution Plan using the \
available workflows.
--- AVAILABLE CAPABILITIES ---
{catalog}
--- END CAPABILITIES ---
INSTRUCTIONS:
1. The plan MUST be a JSON list of objects, where each object is a "step":
   {{"step": <number>, "workflow": "<name from the capabilities>", \
"input": {{<that workflow's input keys>}}}}.
2. You MUST use Context Chaining. If a step requires input from a previous \
step, reference it using the syntax $$STEP_X_OUTPUT$$ (X = the earlier \
step's number). Only EARLIER steps may be referenced.
3. Be strategic. Break compound goals into distinct steps. Use the correct \
input keys of each workflow (see its input JSON schema). The same workflow \
may appear in multiple steps.
4. The plan has between 1 and {max_steps} steps, numbered sequentially \
starting at 1.

EXAMPLE GOAL: "Check where order ord-7 is, then request a full refund for \
it once it has shipped."
EXAMPLE PLAN (JSON LIST):
[
  {{"step": 1, "workflow": "live_data", "input": {{"query": "where is order \
ord-7 and has it shipped?"}}}},
  {{"step": 2, "workflow": "action_execution", "input": {{"action_name": \
"process_refund", "inputs": {{"order_id": "ord-7", "amount": \
"$$STEP_1_OUTPUT$$", "reason": "order shipped — full refund"}}}}}}
]

Return ONLY the JSON list — no prose, no code fences."""


class PlanValidationError(ValueError):
    """A plan that violates the contract (PE-R3) — ``errors`` carries one
    distinguishable message per invalidity found."""

    def __init__(self, errors: list[str]):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


def _history_text(history: Optional[list[dict[str, Any]]]) -> str:
    """Render the conversation slice the planner sees (chronological)."""
    if not history:
        return "(conversation start)"
    return "\n".join(
        f"[{entry.get('role', '?')}] {entry.get('content', '')}"
        for entry in history
    )


# The long-term-memory block (LM v1), same shape as the router's: the
# owner's fact card, service-seeded, rendered inside a labeled data block —
# customer data, never instructions. Absence reads neutrally everywhere.
_MEMORY_BLOCK_HEADER = "What you already know about this customer:"


def _memory_text(memories: Optional[list[str]]) -> str:
    """Render the memory block's body — absence is byte-identical for an
    empty card, a missing key, and an anonymous turn (LM-R5/R7)."""
    if not memories:
        return "(nothing known yet)"
    return "\n".join(f"- {fact}" for fact in memories)


def _placeholder_refs(value: Any) -> set[int]:
    """Every step number referenced by ``$$STEP_X_OUTPUT$$`` tokens in a
    (possibly nested) input value."""
    refs: set[int] = set()
    if isinstance(value, str):
        refs.update(int(m) for m in _PLACEHOLDER_RE.findall(value))
    elif isinstance(value, dict):
        for item in value.values():
            refs |= _placeholder_refs(item)
    elif isinstance(value, list):
        for item in value:
            refs |= _placeholder_refs(item)
    return refs


def validate_plan(plan_text: str) -> list[dict[str, Any]]:
    """Parse and validate one emitted plan; return the normalized JSON-safe
    steps, or raise ``PlanValidationError`` with distinguishable errors.

    Bounds (``PLAN_MAX_STEPS``) read from Settings at call time — nothing
    hardcoded (PE-R13). A well-formed plan passes every check; each
    invalidity class produces its own named error (PE-R3).
    """
    errors: list[str] = []
    try:
        raw = json.loads(plan_text)
    except json.JSONDecodeError as error:
        raise PlanValidationError([f"malformed JSON: {error.msg}"]) from None
    if not isinstance(raw, list):
        raise PlanValidationError([
            f"malformed JSON: the plan must be a JSON list of step objects, "
            f"not {type(raw).__name__}"
        ])
    if not raw:
        raise PlanValidationError([
            "plan has zero steps — a compound request needs at least one "
            "(direct or single-workflow replies are the router's business)"
        ])
    max_steps = Settings().PLAN_MAX_STEPS
    if len(raw) > max_steps:
        raise PlanValidationError([
            f"plan has {len(raw)} steps — the maximum is {max_steps}"
        ])

    steps: list[dict[str, Any]] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            errors.append(
                f"step {index} is malformed: each step must be an object"
            )
            continue
        number = item.get("step")
        if number != index or isinstance(number, bool):
            errors.append(
                f"step {index} is numbered {number!r} — steps must be "
                f"numbered sequentially from 1"
            )
        workflow = item.get("workflow")
        if not isinstance(workflow, str) or not workflow:
            errors.append(
                f"step {index} is malformed: 'workflow' must name a workflow"
            )
            continue
        if workflow not in WORKFLOW_REGISTRY:
            errors.append(
                f"step {index} names unknown workflow {workflow!r} — pick "
                f"one from the available capabilities"
            )
            continue
        step_input = item.get("input")
        if not isinstance(step_input, dict):
            errors.append(f"step {index} is malformed: 'input' must be an object")
            continue
        for ref in sorted(_placeholder_refs(step_input)):
            if 1 <= ref < index:
                continue
            if index > 1:
                hint = f"placeholders may only reference earlier steps (1..{index - 1})"
            else:
                hint = "the first step may not reference anything"
            errors.append(
                f"step {index} references $$STEP_{ref}_OUTPUT$$ — {hint}"
            )
        steps.append({"step": index, "workflow": workflow, "input": step_input})

    if errors:
        raise PlanValidationError(errors)
    return steps


async def generate_plan_text(
    message: str,
    history: Optional[list[dict[str, Any]]] = None,
    catalog: Optional[str] = None,
    feedback: Optional[list[str]] = None,
    memories: Optional[list[str]] = None,
) -> str:
    """One planner-LLM attempt: the emitted plan text (unparsed — the
    validator owns parsing, so malformed output stays a handled class).

    ``feedback`` carries the previous attempt's validation errors (PE-R4's
    correction loop); ``memories`` is the service-seeded fact card (LM v1) —
    durable customer facts that shape workflow choice and inputs.
    """
    catalog = catalog or build_catalog()
    messages: list[Any] = [
        SystemMessage(
            _SYSTEM_TEMPLATE.format(
                catalog=catalog, max_steps=Settings().PLAN_MAX_STEPS
            )
        ),
        HumanMessage(
            f"Conversation so far:\n{_history_text(history)}\n\n"
            f"{_MEMORY_BLOCK_HEADER}\n{_memory_text(memories)}\n\n"
            f"User's compound goal: {message}"
        ),
    ]
    if feedback:
        messages.append(
            HumanMessage(
                "Your previous plan was invalid: "
                + "; ".join(feedback)
                + ". Return a corrected plan."
            )
        )
    model = get_chat_model("planner")
    response = await model.ainvoke(messages)
    return (
        response.content
        if isinstance(response.content, str)
        else str(response.content)
    )


async def build_plan(
    message: str,
    history: Optional[list[dict[str, Any]]] = None,
    catalog: Optional[str] = None,
) -> tuple[Optional[list[dict[str, Any]]], int, list[str]]:
    """The bounded generate→validate loop (PE-R4).

    Returns ``(steps, attempts, errors)``: ``steps`` is the validated plan
    on success or ``None`` on exhaustion, after exactly
    ``PLANNER_MAX_ATTEMPTS`` attempts (settings-driven, PE-R13); ``errors``
    are the last attempt's distinguishable validation errors. A failing
    planner *call* counts as an invalid attempt, not an exception — the
    loop always terminates. Never runs anything (PE-R4).
    """
    attempts = 0
    errors: list[str] = []
    max_attempts = Settings().PLANNER_MAX_ATTEMPTS
    while attempts < max_attempts:
        attempts += 1
        try:
            text = await generate_plan_text(
                message, history, catalog, feedback=errors or None
            )
            steps = validate_plan(text)
            logger.info(
                "planner attempt=%d valid=%s steps=%d",
                attempts,
                True,
                len(steps),
            )
            return steps, attempts, []
        except PlanValidationError as error:
            errors = error.errors
        except Exception as error:  # planner call failed — a handled attempt
            errors = [f"planner call failed ({type(error).__name__}: {error})"]
    logger.info("planner attempts=%d valid=%s exhausted=1", attempts, False)
    return None, attempts, errors


async def exhaustion_reply(
    message: str,
    history: Optional[list[dict[str, Any]]] = None,
    memories: Optional[list[str]] = None,
) -> str:
    """The apologetic direct answer on plan exhaustion (PE-R4) — one
    ``generator`` call with a static fallback, paths.py style; nothing was
    executed and the conversation continues. ``memories`` is the
    service-seeded fact card (LM v1)."""
    model = get_chat_model("generator")
    try:
        response = await model.ainvoke(
            [
                SystemMessage(_EXHAUSTION_SYSTEM),
                HumanMessage(
                    f"Conversation so far:\n{_history_text(history)}\n\n"
                    f"{_MEMORY_BLOCK_HEADER}\n{_memory_text(memories)}\n\n"
                    f"Customer's request that could not be planned: {message}\n"
                    "Write the apology."
                ),
            ]
        )
        return (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )
    except Exception:
        logger.warning(
            "planner exhaustion reply failed; static fallback used",
            exc_info=True,
        )
        return _EXHAUSTION_FALLBACK
