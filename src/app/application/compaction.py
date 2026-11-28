"""Conversation compaction — history as one bounded, layered string (v1).

The history handed to the orchestrator's prompts is assembled here, once,
in plain code: an exact **ledger** of every action the assistant requested,
executed, refused or left pending (built from the structured ``data`` the
transcript rows already store — copied, never retold by a model), an
optional incremental LLM-written **summary** of the older free text, and
the **recent turns word for word**, under a token **hard cap** that drops
whole oldest turns and never cuts a turn in half.

The read half (this file's ``build_history``) is pure: no model call, no
query, never raises. It reads the levers from Settings and nothing else.
The conversation service is its only caller; graphs never import it.

Rendered contract — fixed section labels in this order, each omitted when
empty; the verbatim label appears only when the ledger or the summary does:

    Actions taken so far (exact records):
    - <action>: <status> | input: <exact input> | result: <bounded> | ...
    Earlier conversation (summary of older turns):
    <the stored summary>
    Recent turns (verbatim):
    [user] ...
    [assistant] ...

With no actions and no summary the text is exactly the pre-feature
rendering: bare ``[role] content`` lines, or ``(conversation start)``.
Logs carry counts only — never ledger lines, summary or message content.
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

from app.domain.conversations import ConversationMessageRecord
from app.domain.enums import MessageRole
from app.infrastructure.settings import Settings
from app.infrastructure.tokens import count_tokens

logger = logging.getLogger(__name__)

LEDGER_LABEL = "Actions taken so far (exact records):"
SUMMARY_LABEL = "Earlier conversation (summary of older turns):"
VERBATIM_LABEL = "Recent turns (verbatim):"
CONVERSATION_START = "(conversation start)"

# The one workflow that changes anything. Lookups (self_rag, live_data) are
# ordinary conversation material and never appear in the ledger. The name is
# spelled here rather than read from the registry: this module never imports
# a graph module (Module Boundaries).
ACTION_WORKFLOW = "action_execution"

# Stored status -> the ledger's status vocabulary.
_STATUS_WORDS = {
    "completed": "completed",
    "pending_approval": "pending approval",
    "rejected": "rejected",
    "failed": "failed",
    "skipped": "skipped",
    "pending": "not run",
}
_UNKNOWN_STATUS = "not run"
_NOT_RECORDED = "(not recorded)"


@dataclass(frozen=True)
class HistoryView:
    """What ``build_history`` produced, with the counts the turn log line
    carries (ints and flags only — never text)."""

    text: str
    tokens: int
    ledger_lines: int
    summary_used: bool
    capped: bool
    # The last transcript seq the *next* summary would cover: the row just
    # before the turn that starts the verbatim tail. None when nothing has
    # aged out of the tail.
    summary_boundary: Optional[int]


@dataclass
class _Entry:
    """One action as the records state it (transient)."""

    name: str
    status: str = _UNKNOWN_STATUS
    input_value: Any = None  # None = not on the record (rows predate the feature)
    result: Optional[str] = None
    error: Optional[str] = None
    request_id: Optional[str] = None
    step: Optional[int] = None  # plan step number; None for single-workflow actions
    seq: int = 0  # the row of first appearance


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _lines(rows: list[ConversationMessageRecord]) -> str:
    """The pre-feature rendering: one ``[role] content`` line per row."""
    return "\n".join(f"[{row.role.value}] {row.content}" for row in rows)


def _flatten(turns: list[list[ConversationMessageRecord]]) -> list[ConversationMessageRecord]:
    return [row for turn in turns for row in turn]


def _render(ledger: list[str], summary: str, rows: list[ConversationMessageRecord]) -> str:
    """The three fixed sections in order; empty sections are omitted."""
    blocks: list[str] = []
    if ledger:
        blocks.append(f"{LEDGER_LABEL}\n" + "\n".join(ledger))
    if summary:
        blocks.append(f"{SUMMARY_LABEL}\n{summary}")
    verbatim = _lines(rows)
    if blocks:
        blocks.append(
            f"{VERBATIM_LABEL}\n{verbatim}" if verbatim else VERBATIM_LABEL
        )
    else:
        blocks.append(verbatim)
    text = "\n\n".join(blocks)
    return text if text else CONVERSATION_START


# ---------------------------------------------------------------------------
# Turns — a user row plus every row up to the next user row
# ---------------------------------------------------------------------------


def _split_turns(rows: list[ConversationMessageRecord]) -> list[list[ConversationMessageRecord]]:
    """Group rows into turns. Rows appended outside a turn (approval
    outcomes, resumed-plan answers) land in the turn during which they were
    appended; rows before the first ``user`` row belong to the first turn."""
    turns: list[list[ConversationMessageRecord]] = []
    seen_user = False
    for row in rows:
        if row.role == MessageRole.USER:
            if seen_user:
                turns.append([])
            elif not turns:
                turns.append([])
            seen_user = True
        elif not turns:
            turns.append([])  # leading rows
        turns[-1].append(row)
    return turns


def _tail_start(turns: list[list[ConversationMessageRecord]], soft: int, keep: int) -> int:
    """Index of the first turn of the verbatim tail: the most recent ``keep``
    turns, shrunk by whole turns until their own tokens fit ``soft`` — never
    below the most recent turn. The count excludes ledger and summary."""
    start = max(0, len(turns) - keep)
    per_turn = [count_tokens(_lines(turn)) for turn in turns]
    if sum(per_turn[start:]) <= soft:
        return start  # per-turn sums over-count; nothing to shrink
    last = len(turns) - 1
    while start < last and count_tokens(_lines(_flatten(turns[start:]))) > soft:
        start += 1
    return start


def _fit_hard_cap(
    ledger: list[str], summary: str, turns: list[list[ConversationMessageRecord]], hard: int
) -> tuple[str, int]:
    """Drop whole oldest turns until the final string fits ``hard`` tokens
    (ledger and summary included); the most recent turn is never dropped."""
    def text_from(start: int) -> str:
        return _render(ledger, summary, _flatten(turns[start:]))

    last = len(turns) - 1
    head = count_tokens(_render(ledger, summary, []))
    per_turn = [count_tokens(_lines(turn)) for turn in turns]
    # First pass from per-turn counts (one encode per turn), then exact
    # counts to correct the estimate's boundary error — normally 0–2 encodes.
    start = last
    used = 0
    while start > 0 and head + used + per_turn[start - 1] <= hard:
        start -= 1
        used += per_turn[start]
    while start < last and count_tokens(text_from(start)) > hard:
        start += 1
    while start > 0 and count_tokens(text_from(start - 1)) <= hard:
        start -= 1
    text = text_from(start)
    return text, count_tokens(text)


# ---------------------------------------------------------------------------
# The ledger — code copies the records, no model is involved
# ---------------------------------------------------------------------------


def _dump(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


def _as_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    return value if isinstance(value, str) else _dump(value)


def _truncate(text: str, limit: int) -> str:
    """Cut ``text`` to at most ``limit`` characters with an explicit marker."""
    marker_room = 40  # room for " … [truncated, N more chars]"
    if len(text) <= limit:
        return text
    keep = max(1, limit - marker_room)
    omitted = len(text) - keep
    return f"{text[:keep]} … [truncated, {omitted} more chars]"


def _action_name(workflow: Any, input_value: Any) -> str:
    """``cancel_order`` rather than ``action_execution`` when the input
    names the action."""
    if (
        workflow == ACTION_WORKFLOW
        and isinstance(input_value, dict)
        and isinstance(input_value.get("action_name"), str)
        and input_value["action_name"]
    ):
        return input_value["action_name"]
    return str(workflow)


def _step_number(step: Any) -> Optional[int]:
    return step if isinstance(step, int) and not isinstance(step, bool) else None


def _is_plan_row(data: dict[str, Any]) -> bool:
    """A plan snapshot row carries ``data.plan.steps``."""
    plan = data.get("plan")
    return isinstance(plan, dict) and isinstance(plan.get("steps"), list)


def _ledger(ledger: list[str], transcript: list[ConversationMessageRecord], line_max: int) -> None:
    """Fill ``ledger`` with one line per action, ordered by first appearance,
    each showing the latest recorded status."""
    entries: dict[tuple, _Entry] = {}
    plan_rows: list[int] = []  # seqs of plan snapshot rows, in order
    plan_for_request: dict[str, Optional[int]] = {}

    def single(run: dict[str, Any], data: dict[str, Any], row: ConversationMessageRecord, index: int) -> None:
        request_id = run.get("request_id")
        input_value = data.get("input")
        key: tuple = ("request", str(request_id)) if request_id else ("run", row.seq, index)
        entry = entries.get(key)
        if entry is None:
            # A later row carrying the same request id (the approval
            # decision) replaces the status; the input stays the first one.
            entry = _Entry(
                name=_action_name(run.get("workflow"), input_value),
                input_value=input_value,
                seq=row.seq,
            )
            entries[key] = entry
        elif entry.input_value is None and input_value is not None:
            entry.input_value = input_value
        entry.status = _STATUS_WORDS.get(str(run.get("status")), _UNKNOWN_STATUS)
        entry.result = _as_text(run.get("output"))
        entry.error = _as_text(run.get("error"))
        if request_id:
            entry.request_id = str(request_id)

    def plan(row: ConversationMessageRecord, data: dict[str, Any]) -> None:
        steps = (data.get("plan") or {}).get("steps")
        if not isinstance(steps, list):
            return
        plan_rows.append(row.seq)
        for step in steps:
            if not isinstance(step, dict) or step.get("workflow") != ACTION_WORKFLOW:
                continue
            number = _step_number(step.get("step"))
            if number is None:
                continue
            key = (row.seq, number)
            if key not in entries:
                entries[key] = _Entry(
                    name=_action_name(step.get("workflow"), step.get("input")),
                    input_value=step.get("input"),
                    step=number,
                    seq=row.seq,
                )

    def plan_outcome(row: ConversationMessageRecord, data: dict[str, Any], runs: list[Any]) -> None:
        # Which plan these steps belong to: the one whose pause the row
        # resumes (resumed_request_id), else the most recent plan row.
        resumed = data.get("resumed_request_id")
        plan_seq = plan_for_request.get(str(resumed)) if resumed else None
        if plan_seq is None:
            plan_seq = plan_rows[-1] if plan_rows else None
        pending_id = data.get("pending_request_id")
        if pending_id:
            plan_for_request[str(pending_id)] = plan_seq
        awaiting_named = False
        for step in runs:
            if not isinstance(step, dict) or step.get("workflow") != ACTION_WORKFLOW:
                continue
            number = _step_number(step.get("step"))
            if number is None:
                continue
            key = (plan_seq if plan_seq is not None else row.seq, number)
            entry = entries.get(key)
            if entry is None:
                entry = _Entry(
                    name=_action_name(step.get("workflow"), None),
                    step=number,
                    seq=row.seq,
                )
                entries[key] = entry
            raw = str(step.get("status"))
            if raw == "pending" and pending_id and not awaiting_named:
                # The row's pending_request_id marks the one step that is
                # waiting on a human; every later pending step was not reached.
                entry.status = "pending approval"
                entry.request_id = str(pending_id)
                awaiting_named = True
            else:
                entry.status = _STATUS_WORDS.get(raw, _UNKNOWN_STATUS)
            entry.result = _as_text(step.get("output"))
            entry.error = _as_text(step.get("error"))

    for row in transcript:
        data = row.data
        if not isinstance(data, dict):
            continue
        if _is_plan_row(data):
            try:
                plan(row, data)
            except Exception:  # one unusable entry never voids the rest
                logger.warning("compaction: a plan row could not be read (seq=%d)", row.seq)
            continue
        runs = data.get("runs")
        if not isinstance(runs, list):
            continue
        step_runs = [run for run in runs if isinstance(run, dict) and "step" in run]
        if step_runs:
            try:
                plan_outcome(row, data, step_runs)
            except Exception:
                logger.warning("compaction: a plan outcome row could not be read (seq=%d)", row.seq)
            continue
        for index, run in enumerate(runs):
            if not isinstance(run, dict) or run.get("workflow") != ACTION_WORKFLOW:
                continue
            try:
                single(run, data, row, index)
            except Exception:
                logger.warning("compaction: an outcome row could not be read (seq=%d)", row.seq)

    for entry in entries.values():
        parts = [f"{entry.name}: {entry.status}"]
        parts.append(
            "input: " + (_dump(entry.input_value) if entry.input_value is not None else _NOT_RECORDED)
        )
        if entry.error:
            parts.append("error: " + _truncate(entry.error, line_max))
        elif entry.result:
            parts.append("result: " + _truncate(entry.result, line_max))
        if entry.request_id:
            parts.append(f"request: {entry.request_id}")
        if entry.step is not None:
            parts.append(f"plan step: {entry.step}")
        parts.append(f"seq: {entry.seq}")
        ledger.append("- " + " | ".join(parts))


# ---------------------------------------------------------------------------
# Public API — the read half
# ---------------------------------------------------------------------------


def build_history(
    transcript: list[ConversationMessageRecord],
    summary: Optional[str] = None,
    summary_through_seq: Optional[int] = None,
) -> HistoryView:
    """The one history string every prompt embeds. Pure; never raises.

    ``transcript`` is the whole conversation; ``summary`` /
    ``summary_through_seq`` are the header's two columns (None = nothing
    summarized yet).
    """
    settings = Settings()
    if not settings.COMPACTION_ENABLED:
        # Rollback: today's rendering of every row, whatever the header holds.
        text = _lines(transcript) or CONVERSATION_START
        return HistoryView(text, count_tokens(text), 0, False, False, None)

    ledger: list[str] = []
    _ledger(ledger, transcript, settings.COMPACTION_LEDGER_LINE_MAX_CHARS)
    summary = (summary or "").strip()
    verbatim = [
        row
        for row in transcript
        if summary_through_seq is None or row.seq > summary_through_seq
    ]

    # Where the next summary would end: the row before the verbatim tail.
    turns = _split_turns(verbatim)
    tail = _tail_start(turns, settings.COMPACTION_SOFT_TOKENS, settings.COMPACTION_KEEP_TURNS)
    if tail == 0:
        boundary = summary_through_seq
    else:
        boundary = turns[tail - 1][-1].seq

    text, tokens = _fit_hard_cap(ledger, summary, turns, settings.COMPACTION_HARD_TOKENS)
    capped = tokens > settings.COMPACTION_HARD_TOKENS
    if capped:
        # Only possible when the most recent turn alone is over the cap.
        logger.warning(
            "compaction: the most recent turn alone exceeds the hard cap "
            "(tokens=%d cap=%d turns_kept=1 ledger_lines=%d summary=%d)",
            tokens,
            settings.COMPACTION_HARD_TOKENS,
            len(ledger),
            int(bool(summary)),
        )
    return HistoryView(
        text=text,
        tokens=tokens,
        ledger_lines=len(ledger),
        summary_used=bool(summary),
        capped=capped,
        summary_boundary=boundary,
    )
