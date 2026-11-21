"""Executor — todo tools (T8) + the executor node (T9's declared slice).

**Tools half (this file, A11 / PE-R5/R6):** the ``TodoList`` state and the
four executor-bound todo tools, porting the *semantics* of the supplied
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

The tools are per-invocation instances bound (by closure) to one
``TodoList`` — the list rides the orchestrator graph state, and T9's
executor node builds a fresh tool set per entry, the ``live_data`` per-run
tool-resolution pattern. Hydration is plain pydantic
(``TodoList.model_validate({"items": ...})``); the private
created-unlisted guardrail set is deliberately not serialized (reference
behavior — it is per-session observation state, not plan state).
"""

import json
import logging
from typing import Any, Optional

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, PrivateAttr

from app.domain.enums import TodoStatus

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
