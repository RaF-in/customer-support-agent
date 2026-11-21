from enum import StrEnum


class DocumentStatus(StrEnum):
    """Lifecycle of an uploaded document through the ingestion pipeline."""

    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class DataCategory(StrEnum):
    """Supported ingestion formats.

    Values drive factory routing and Qdrant collection naming
    (e.g. `pdfs` -> `cleaned_pdfs` / `pdf_chunks`).
    """

    PDF = "pdfs"
    TEXT = "texts"
    DOCX = "docxs"


class ApprovalStatus(StrEnum):
    """Lifecycle of an approval request.

    Only ``pending -> approved | rejected`` is a valid transition, and the
    runtime enforces it (T7) — not the record model. ``expired`` is
    reserved for a future expiry mechanism (ARCH Open Question).
    """

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class ActionStatus(StrEnum):
    """Outcome of one action execution — written once, immutable."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"


class TurnOutcome(StrEnum):
    """Router classification of one conversational turn (CR-R5, PE-R1).

    ``multi_workflow`` routes the turn into the slice-2 planner–executor
    path; the other three outcomes are served by the slice-1 reply paths.
    """

    DIRECT = "direct"
    CLARIFY = "clarify"
    WORKFLOW = "workflow"
    MULTI_WORKFLOW = "multi_workflow"


class MessageRole(StrEnum):
    """Author of one transcript row — the user or the assistant."""

    USER = "user"
    ASSISTANT = "assistant"


class MessageKind(StrEnum):
    """Content class of one transcript row (CR-R3, PE-R12).

    User messages are always ``text``; workflow outcomes and validated
    plans are transcript artifacts too — everything transcript-visible
    lives in the one messages table.
    """

    TEXT = "text"
    WORKFLOW_OUTCOME = "workflow_outcome"
    PLAN = "plan"


class PlanStatus(StrEnum):
    """Lifecycle of one orchestrated plan's mutable working record.

    ``validated -> executing -> (paused <-> executing) -> done``, terminal
    via failure-exhaustion or executor takeover. Rejection of a step's
    approval counts as that step failing (PE decision 8).
    """

    VALIDATED = "validated"
    EXECUTING = "executing"
    PAUSED = "paused"
    DONE = "done"
    FAILED = "failed"


class TodoStatus(StrEnum):
    """Status of one executor todo — semantics per ``specs/context.txt``."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ABANDONED = "abandoned"
