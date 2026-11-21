"""Executor — todo tools (T8) + the executor/verify nodes (T9's slice).

**Tools half (A11 / PE-R5/R6):** the ``TodoList`` state and the four
executor-bound todo tools, porting the *semantics* of the supplied
reference implementation (``specs/context.txt`` — normative) into langchain
``StructuredTool`` form: statuses pending/in_progress/completed/abandoned,
observe-before-claim (a todo created since the last ``ListTodos`` cannot be
completed — the create-then-complete busywork guard), abandonment requires
a stated reason, and a list is complete only when it has no open items
**and** is non-empty.

Two deliberate deltas from the reference, both required by the REQ:

- ``is_complete`` closes the vacuous-truth trap — the reference's
  ``all([]) == True`` would call an empty list complete; PE-R5 forbids that
  ("an empty todo list is never accepted as complete").
- ``AbandonTodo``'s reason is *required* (the reference defaults it to
  ``""``) and is recorded *on the todo* (``abandon_reason``), because the
  run's audit trail — "every abandonment carries a reason in the record" —
  is the todo row, not the tool result.

Tool results are JSON strings (deterministic and framework-unambiguous):
successes carry the reference's ``{"added"|"completed"|"abandoned":
<todo>, "todos": <render>}`` shape; refusals carry ``{"error": ...}`` with
corrective guidance the executor model can act on. The tools never raise —
a refusal is a handled outcome, not an exception (the two-failure-classes
discipline).

**Node half (T9, A9/A11/A12 / PE-R5..R13):** the executor works a
validated plan as an ``agent``-purpose LLM with the todo tools plus one
``run_plan_step`` tool bound — the LLM *proposes* (creates todos in plan
order, settles steps in order, claims completion) while the tool *disposes*
(strictly: sequential order enforced, inputs are the plan's own with
``$$STEP_X_OUTPUT$$`` resolved against earlier outputs, every run through
the facade). The tool owns each step's todo transitions — in_progress at
run start, completed on success, abandoned with a specific reason on
failure/rejection — and cuts a failed step's transitive dependents before
resolution, so dependents are never resolved against failed outputs and a
normally-ended run closes with zero open todos. A ``pending_approval`` step
pauses the run: the node emits the pause state (``plan`` + todos +
``pending_request_id``) for the service to persist as a paused
``PlanRecord`` (A9 — the graph touches no Postgres). The ``verify`` node is
the orchestrator checking the todo list itself: open items re-invoke the
executor with corrective feedback ≤ ``EXECUTOR_MAX_REINVOCATIONS``, then
the orchestrator takes over (``takeover`` flag) for an honest partial
report. The executor's own inner tool-calling loop is round-bounded
(derived from ``PLAN_MAX_STEPS``); a model failure degrades to ending the
entry — never an exception (CR-N1).
"""

import json
import logging
import re
from typing import Any, Optional

from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, PrivateAttr

from app.domain.enums import PlanStatus, TodoStatus, TurnOutcome
from app.graphs.base.state import WorkflowResult, WorkflowStatus
from app.graphs.orchestrator.state import OrchestratorState
from app.graphs.runtime import run_workflow
from app.infrastructure.llm.factory import get_chat_model
from app.infrastructure.settings import Settings

logger = logging.getLogger(__name__)

# Open = still claims to be doable; closed = settled either way (reference).
_OPEN: tuple[TodoStatus, ...] = (TodoStatus.PENDING, TodoStatus.IN_PROGRESS)
_CLOSED: tuple[TodoStatus, ...] = (TodoStatus.COMPLETED, TodoStatus.ABANDONED)

# Render marks, exactly per the reference.
_MARKS = {
    TodoStatus.PENDING: "[ ]",
    TodoStatus.IN_PROGRESS: "[~]",
    TodoStatus.COMPLETED: "[x]",
    TodoStatus.ABANDONED: "[-]",
}


class Todo(BaseModel):
    """One executor todo — the shape that survives into ``PlanRecord.todos``
    (same fields as ``domain.plans.Todo``, so the service maps rows without
    a translation step)."""

    id: int
    text: str
    status: TodoStatus = TodoStatus.PENDING
    abandon_reason: Optional[str] = None  # required when abandoning


class TodoList(BaseModel):
    """The executor's working ledger — what the orchestrator (not the
    executor) inspects to verify completion (PE-R8).

    ``_created_unlisted`` holds ids created but not yet observed via
    ``ListTodos``: ``CompleteTodo`` refuses to close one of these, which
    operationalizes "never create AND complete a todo in the same breath"
    — the model must observe state before it may mark freshly-created work
    done. Not serialized (reference behavior).
    """

    items: list[Todo] = []
    _created_unlisted: set[int] = PrivateAttr(default_factory=set)

    def is_complete(self) -> bool:
        """True only when the list is non-empty AND every item is closed —
        the ``all([]) == True`` trap is closed (PE-R5)."""
        return bool(self.items) and all(t.status in _CLOSED for t in self.items)

    def has_open(self) -> bool:
        return any(t.status in _OPEN for t in self.items)

    def count_open(self) -> int:
        """Number of still-open todos — surfaced in verify/takeover reports."""
        return sum(1 for t in self.items if t.status in _OPEN)

    def get(self, todo_id: int) -> Optional[Todo]:
        return next((t for t in self.items if t.id == todo_id), None)

    def add(self, text: str) -> Todo:
        todo = Todo(id=(max((t.id for t in self.items), default=0) + 1), text=text)
        self.items.append(todo)
        self._created_unlisted.add(todo.id)
        return todo

    def mark_listed(self) -> None:
        self._created_unlisted.clear()

    def render(self) -> str:
        """The reference's deterministic rendering — ``[ ] 1. text`` lines."""
        if not self.items:
            return "(no todos yet)"
        return "\n".join(
            f"{_MARKS[t.status]} {t.id}. {t.text}" for t in self.items
        )


def _dump(payload: dict[str, Any]) -> str:
    """Tool results are JSON strings (deterministic; no default=str so an
    unserializable payload surfaces here in development, never downstream)."""
    return json.dumps(payload)


def make_todo_tools(todos: TodoList) -> list[BaseTool]:
    """Build the four todo tools bound to one ``TodoList`` (per-invocation
    closures — T9's executor node builds a fresh set per entry)."""

    async def add_todo(text: str) -> str:
        """Add one item to the plan. Break the work into small, verifiable steps.

        Args:
            text: A single, concrete task. Add several by calling this once per task.
        """
        cleaned = (text or "").strip()
        if not cleaned:
            return _dump({"error": "Todo text is empty."})
        todo = todos.add(cleaned)
        return _dump({"added": todo.model_dump(), "todos": todos.render()})

    async def complete_todo(id: int) -> str:
        """Mark a todo done — only after you have actually finished its work.

        You cannot complete a todo you created without first calling list_todos
        to review the plan (this prevents create-and-immediately-complete
        busywork).

        Args:
            id: The id of the todo to mark completed.
        """
        todo = todos.get(id)
        if todo is None:
            return _dump({"error": f"No todo with id {id}."})
        if id in todos._created_unlisted:
            return _dump(
                {
                    "error": (
                        f"Todo {id} was just created. Do its work and call "
                        "list_todos to review the plan before completing it."
                    )
                }
            )
        todo.status = TodoStatus.COMPLETED
        return _dump({"completed": todo.model_dump(), "todos": todos.render()})

    async def abandon_todo(id: int, reason: str) -> str:
        """Drop a todo that turned out to be unnecessary or impossible.

        Args:
            id: The id of the todo to abandon.
            reason: Why it is being dropped (required — kept on the record).
        """
        cleaned = (reason or "").strip()
        if not cleaned:
            return _dump(
                {"error": "Abandoning a todo requires a stated reason."}
            )
        todo = todos.get(id)
        if todo is None:
            return _dump({"error": f"No todo with id {id}."})
        todo.status = TodoStatus.ABANDONED
        todo.abandon_reason = cleaned
        # Reference behavior: abandoning a created-unlisted todo also clears
        # the unlisted flag — it is settled, not pending observation.
        todos._created_unlisted.discard(id)
        return _dump(
            {
                "abandoned": todo.model_dump(),
                "reason": cleaned,
                "todos": todos.render(),
            }
        )

    async def list_todos() -> str:
        """Review the current plan and progress. Call this before completing work."""
        todos.mark_listed()
        return _dump(
            {
                "todos": todos.render(),
                "complete": todos.is_complete(),
                "open": todos.count_open(),
            }
        )

    return [
        StructuredTool.from_function(
            coroutine=add_todo, name="add_todo", description=add_todo.__doc__ or ""
        ),
        StructuredTool.from_function(
            coroutine=complete_todo,
            name="complete_todo",
            description=complete_todo.__doc__ or "",
        ),
        StructuredTool.from_function(
            coroutine=abandon_todo,
            name="abandon_todo",
            description=abandon_todo.__doc__ or "",
        ),
        StructuredTool.from_function(
            coroutine=list_todos,
            name="list_todos",
            description=list_todos.__doc__ or "",
        ),
    ]


# ---------------------------------------------------------------------------
# Executor node (T9's declared slice of this file)
# ---------------------------------------------------------------------------

# The A12 dependency rule, executor-owned: step B depends on A iff B's
# inputs contain $$STEP_A_OUTPUT$$ (transitive closure is computed by the
# fail-stop cut below). Same token shape the planner validates.
_STEP_PLACEHOLDER_RE = re.compile(r"\$\$STEP_(\d+)_OUTPUT\$\$")

# Reason strings recorded on abandoned todos — the audit vocabulary (PE-R9
# / REQ todo rules); ids and statuses only in logs, reasons live on rows.
_REASON_DEPENDENCY = "dependency failed"

_EXECUTOR_SYSTEM_TEMPLATE = """\
You are the executor of a customer-support assistant working a validated \
plan through registered workflows.

THE PLAN (immutable — never edit it, never invent steps or inputs, never \
retry or reorder):
{plan}

Work strictly in this order:
1. FIRST prepare the todo list: call add_todo once per plan step, in plan \
order — exactly one todo per step ({step_count} steps -> {step_count} todos).
2. Call list_todos to review the plan.
3. Settle every step with run_plan_step(step) in plan order. The tool \
tracks each step's todo automatically (completed on success, abandoned \
with a reason on failure) — do not hand-complete step todos.
4. A failed step's dependents are skipped automatically; independent \
steps still run. Keep going.
5. When every step is settled, reply that you are done — the system \
verifies the todo list itself, so only claim done when it is true.
{feedback}"""


def _placeholder_refs(value: Any) -> set[int]:
    """Every step number referenced by ``$$STEP_X_OUTPUT$$`` tokens in a
    (possibly nested) step input."""
    refs: set[int] = set()
    if isinstance(value, str):
        refs.update(int(m) for m in _STEP_PLACEHOLDER_RE.findall(value))
    elif isinstance(value, dict):
        for item in value.values():
            refs |= _placeholder_refs(item)
    elif isinstance(value, list):
        for item in value:
            refs |= _placeholder_refs(item)
    return refs


def output_text(output: Any) -> str:
    """Best-effort displayable text of one step output (composer shares
    this for the PE-R11 verbatim guard and step reports)."""
    if isinstance(output, str):
        return output
    return json.dumps(output, default=str)


def _resolve_placeholders(value: Any, outputs: dict[int, Any]) -> Any:
    """Substitute ``$$STEP_X_OUTPUT$$`` tokens (PE-R7).

    A value that *is* a single token is replaced by the referenced output
    itself (type preserved — the target schema field may expect an object
    or number); tokens embedded in longer strings are substituted as text.
    """
    if isinstance(value, str):
        exact = _STEP_PLACEHOLDER_RE.fullmatch(value.strip())
        if exact:
            return outputs.get(int(exact.group(1)))
        return _STEP_PLACEHOLDER_RE.sub(
            lambda m: output_text(outputs.get(int(m.group(1)))), value
        )
    if isinstance(value, dict):
        return {key: _resolve_placeholders(item, outputs) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_placeholders(item) for item in value]
    return value


def _hydrate_steps(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Fresh mutable step dicts from the seeded/resumed plan (the state's
    dict is replaced wholesale on each node update — never mutated in place
    where the channel could observe it)."""
    return [dict(step) for step in plan.get("steps") or []]


def _executor_max_rounds() -> int:
    """Inner tool-calling rounds per executor entry, derived from the plan
    bound (never hardcoded, PE-R13): each step needs a run round plus slack
    for todo preparation, refusals, and the completion claim."""
    return Settings().PLAN_MAX_STEPS * 3 + 8


async def executor_node(state: OrchestratorState) -> dict[str, Any]:
    """Work the plan: the agent prepares the todo list and settles steps
    through ``run_plan_step``; the tool owns todo transitions, placeholder
    resolution, fail-stop, and the approval pause."""
    plan = dict(state.get("plan") or {})
    steps = _hydrate_steps(plan)
    todos = TodoList.model_validate({"items": plan.get("todos") or []})
    tenant_id = state["tenant_id"]
    runs: list[WorkflowResult] = list(state.get("runs") or [])
    new_runs: list[WorkflowResult] = []
    pause: dict[str, Any] = {}

    def settle_todo(number: int, status: TodoStatus, reason: Optional[str] = None) -> None:
        """System-side todo transition — open todos only (event-time
        reasons survive the sweep's generic ones)."""
        todo = todos.get(number)
        if todo is None or todo.status in _CLOSED:
            return
        todo.status = status
        if status == TodoStatus.ABANDONED:
            todo.abandon_reason = reason or "workflow failed"

    def cut_dependents(failed_number: int) -> list[int]:
        """A12 fail-stop: transitively skip every later step whose inputs
        reference the failed one (directly or through an already-cut step),
        abandoning its todo — dependents are cut before resolution."""
        skipped: list[int] = []
        changed = True
        while changed:
            changed = False
            for step in steps:
                if step["number"] <= failed_number or step["status"] != "pending":
                    continue
                refs = _placeholder_refs(step["input"])
                if any(
                    steps[ref - 1]["status"] in ("failed", "skipped")
                    for ref in refs
                    if 1 <= ref <= len(steps)
                ):
                    step["status"] = "skipped"
                    settle_todo(step["number"], TodoStatus.ABANDONED, _REASON_DEPENDENCY)
                    skipped.append(step["number"])
                    changed = True
        return sorted(skipped)

    async def run_plan_step(step: int) -> str:
        """Settle one plan step: resolve its $$STEP_X_OUTPUT$$ inputs
        against earlier step outputs and run its workflow through the facade.

        Args:
            step: The step number to run (steps settle strictly in plan order).
        """
        number = int(step)
        if number < 1 or number > len(steps):
            return _dump(
                {"error": f"No plan step {number} — the plan has {len(steps)} steps."}
            )
        target = steps[number - 1]
        if target["status"] != "pending":
            return _dump(
                {"step": number, "status": target["status"], "note": "already settled"}
            )
        earlier_open = [
            s["number"] for s in steps
            if s["number"] < number and s["status"] == "pending"
        ]
        if earlier_open:
            return _dump(
                {"error": f"Steps run in plan order — settle {earlier_open} first."}
            )
        if len(todos.items) < len(steps):
            # R5 in-graph: nothing executes before the todo list exists.
            return _dump(
                {
                    "error": (
                        f"Prepare the todo list first: one add_todo per plan "
                        f"step, in plan order ({len(steps)} expected, "
                        f"{len(todos.items)} present)."
                    )
                }
            )
        refs = _placeholder_refs(target["input"])
        unready = [
            ref for ref in sorted(refs)
            if ref >= number or steps[ref - 1]["status"] != "completed"
        ]
        if unready:
            # Defensive: eager cutting should have settled these already.
            target["status"] = "skipped"
            settle_todo(number, TodoStatus.ABANDONED, _REASON_DEPENDENCY)
            return _dump(
                {
                    "step": number,
                    "status": "skipped",
                    "note": f"dependency failed (step {unready} did not complete)",
                }
            )
        outputs = {
            s["number"]: s["output"]
            for s in steps
            if s["status"] == "completed" and s.get("output") is not None
        }
        resolved = _resolve_placeholders(target["input"], outputs)
        settle_todo(number, TodoStatus.IN_PROGRESS)
        result = await run_workflow(target["workflow"], tenant_id, resolved)
        new_runs.append(result)
        target["resolved_input"] = resolved
        logger.info(
            "step tenant=%s turn_id=%s step=%d workflow=%s status=%s",
            tenant_id,
            state["run_id"],
            number,
            target["workflow"],
            result.status.value,
        )
        if result.status == WorkflowStatus.COMPLETED:
            target["status"] = "completed"
            target["output"] = result.output
            settle_todo(number, TodoStatus.COMPLETED)
            return _dump(
                {
                    "step": number,
                    "workflow": target["workflow"],
                    "status": "completed",
                    "output": output_text(result.output),
                }
            )
        if result.status == WorkflowStatus.PENDING_APPROVAL:
            pause["request_id"] = result.request_id
            return _dump(
                {
                    "step": number,
                    "workflow": target["workflow"],
                    "status": "pending_approval",
                    "request_id": result.request_id,
                    "note": (
                        "The step is waiting on a human approval — the run "
                        "is paused here."
                    ),
                }
            )
        reason = (
            "approval rejected"
            if result.status == WorkflowStatus.REJECTED
            else f"workflow failed: {result.error or '(no detail)'}"
        )
        target["status"] = "failed"
        settle_todo(number, TodoStatus.ABANDONED, reason)
        skipped = cut_dependents(number)
        return _dump(
            {
                "step": number,
                "workflow": target["workflow"],
                "status": "failed",
                "error": result.error or reason,
                "skipped_dependents": skipped,
            }
        )

    tools: list[BaseTool] = make_todo_tools(todos) + [
        StructuredTool.from_function(
            coroutine=run_plan_step,
            name="run_plan_step",
            description=run_plan_step.__doc__ or "",
        )
    ]
    feedback = plan.get("executor_feedback")
    plan_view = json.dumps(
        [
            {"step": s["number"], "workflow": s["workflow"], "input": s["input"]}
            for s in steps
        ],
        default=str,
    )
    messages: list[Any] = [
        SystemMessage(
            _EXECUTOR_SYSTEM_TEMPLATE.format(
                plan=plan_view,
                step_count=len(steps),
                feedback=(
                    f"\nSUPERVISOR FEEDBACK (your previous claim was wrong):\n"
                    f"{feedback}"
                    if feedback
                    else ""
                ),
            )
        )
    ]
    try:
        bound = get_chat_model("agent").bind_tools(tools)
    except Exception:
        logger.warning(
            "executor model unavailable; entry ends for verify to handle",
            exc_info=True,
        )
        bound = None

    rounds = 0
    max_rounds = _executor_max_rounds()
    while bound is not None and rounds < max_rounds and not pause:
        try:
            response = await bound.ainvoke(messages)
        except Exception:
            # LLM degradation (CR-N1 ladder): end this entry — verify's
            # bounded re-invocation / takeover machinery takes over.
            logger.warning(
                "executor model call failed; entry ends for verify to handle",
                exc_info=True,
            )
            break
        rounds += 1
        messages.append(response)
        if not response.tool_calls:
            break  # the executor's completion claim — verify checks it
        for call in response.tool_calls:
            tool = next((t for t in tools if t.name == call["name"]), None)
            if tool is None:
                result = _dump({"error": f"unknown tool {call['name']!r}"})
            else:
                result = str(await tool.ainvoke(call.get("args") or {}))
            messages.append(
                ToolMessage(content=result, tool_call_id=call["id"] or "call")
            )
            if pause:
                break

    # System closure sweep: on a normally-ended entry every terminal step's
    # todo is closed (event-time reasons already recorded; these are
    # belt-and-braces) — "a run never ends with open todos" (PE-R9).
    for step in steps:
        if step["status"] == "completed":
            settle_todo(step["number"], TodoStatus.COMPLETED)
        elif step["status"] == "failed":
            settle_todo(step["number"], TodoStatus.ABANDONED, "workflow failed")
        elif step["status"] == "skipped":
            settle_todo(step["number"], TodoStatus.ABANDONED, _REASON_DEPENDENCY)

    plan.update(
        status=PlanStatus.PAUSED.value if pause else PlanStatus.EXECUTING.value,
        steps=steps,
        todos=[todo.model_dump(mode="json") for todo in todos.items],
        executor_feedback=None,
    )
    all_runs = runs + new_runs
    logger.info(
        "executor tenant=%s turn_id=%s rounds=%d steps=%d open=%d paused=%s",
        tenant_id,
        state["run_id"],
        rounds,
        len(steps),
        todos.count_open(),
        bool(pause),
    )
    if pause:
        plan["pending_request_id"] = pause["request_id"]
        return {
            "plan": plan,
            "runs": all_runs,
            "outcome": TurnOutcome.MULTI_WORKFLOW,
            "pending_request_id": pause["request_id"],
        }
    return {"plan": plan, "runs": all_runs}


async def verify_node(state: OrchestratorState) -> dict[str, Any]:
    """The orchestrator verifies the todo list itself (PE-R8/A11) — never
    the executor's claim. All closed → ``done``; open items → bounded
    re-invocation with corrective feedback; exhausted → takeover (plan
    ``failed``) for an honest partial report in ``compose_final``."""
    plan = dict(state.get("plan") or {})
    todos = TodoList.model_validate({"items": plan.get("todos") or []})
    if todos.is_complete():
        plan["status"] = PlanStatus.DONE.value
        return {"plan": plan}
    max_reinvocations = Settings().EXECUTOR_MAX_REINVOCATIONS
    if plan.get("reinvocations", 0) < max_reinvocations:
        plan["reinvocations"] = plan.get("reinvocations", 0) + 1
        if not todos.items:
            plan["executor_feedback"] = (
                "The todo list is EMPTY — an empty list is never complete. "
                "Build it first: one add_todo per plan step, in plan order."
            )
        else:
            plan["executor_feedback"] = (
                "You claimed completion but these todos remain open:\n"
                f"{todos.render()}\n"
                "Continue the plan: settle the remaining steps with "
                "run_plan_step, in plan order."
            )
        logger.info(
            "verify tenant=%s turn_id=%s open=%d reinvocation=%d/%d",
            state["tenant_id"],
            state["run_id"],
            todos.count_open(),
            plan["reinvocations"],
            max_reinvocations,
        )
        return {"plan": plan}
    plan["takeover"] = True
    plan["status"] = PlanStatus.FAILED.value
    logger.info(
        "verify tenant=%s turn_id=%s open=%d takeover=1",
        state["tenant_id"],
        state["run_id"],
        todos.count_open(),
    )
    return {"plan": plan}
