"""Orchestrator graph state + the turn contracts (conversation-orchestrator).

``OrchestratorState`` extends the shared envelope: the conversation service
injects the envelope keys per turn — ``tenant_id`` (only source of identity,
same structural rule as the runtime), a fresh ``run_id`` as the turn's
correlation id, empty ``errors``. Extension keys carry the turn's input
(``message`` / ``history`` / ``catalog`` / ``must_commit``, service-seeded),
the route classification, the reply paths' results, and the slice-2 resume
key (``plan``) — declared now, wired by T9/T10.

``RouterDecision`` is the structured-output target of the route node (A5);
semantic validation (registry membership + input schema) lives in the
router, not in the schema. ``TurnResult`` is the conversation service's
per-turn return (library contract the HTTP layer maps in T6).
"""

from typing import Any, Optional, TypedDict
from uuid import UUID

from pydantic import BaseModel, Field

from app.domain.enums import TurnOutcome
from app.graphs.base.state import WorkflowResult, WorkflowState


class OrchestratorState(WorkflowState, total=False):
    """Envelope extension for the orchestrator graph.

    All extension keys are optional (``total=False``): nodes merge partial
    updates, and the service injects the envelope keys per turn. The graph
    is orchestration-only — it never touches Qdrant, Postgres, or ``api/``;
    persistence flows back through state to the service.
    """

    # Turn input — seeded by the conversation service
    message: str
    history: list[dict[str, Any]]  # prior transcript entries, chronological
    catalog: str  # deterministic menu text (catalog.build_catalog())
    must_commit: bool  # clarify cap reached — the T4 conditional edge forces a commit
    # Long-term memory (LM v1): the owner's active fact-card contents in
    # recency order, plain strings — the graph learns memory exactly the way
    # it learns history, as a service-seeded key (graphs import nothing new).
    # Seeded only when non-empty; absence (anonymous turn, empty card, or the
    # resume path) must render neutrally in every consuming prompt (LM-R5/R7).
    memories: list[str]
    # route node output — RouterDecision.model_dump()
    classification: dict[str, Any]
    # reply-path output (T4)
    runs: list[WorkflowResult]  # facade results of the turn's workflow runs
    reply: str
    pending_request_id: Optional[str]  # carried for persistence (CR-R10)
    outcome: TurnOutcome
    # slice-2 resume keys (T9/T10) — seeded from a persisted PlanRecord
    plan: dict[str, Any]


class RouterDecision(BaseModel):
    """Structured-output classification of one turn (A5).

    Field constraints beyond the types (workflow must name a registry
    entry, ``workflow_input`` must pass its schema, ``clarify`` needs a
    question) are enforced by the router's semantic validation — the
    structured-output schema stays provider-friendly.
    """

    outcome: TurnOutcome = Field(
        description="direct | clarify | workflow | multi_workflow"
    )
    workflow: Optional[str] = Field(
        default=None,
        description="WORKFLOW_REGISTRY name — required when outcome is workflow",
    )
    workflow_input: Optional[dict[str, Any]] = Field(
        default=None,
        description="input payload for the named workflow, matching its input JSON schema",
    )
    clarifying_question: Optional[str] = Field(
        default=None,
        description="the one clarifying question — required when outcome is clarify",
    )


class TurnResult(BaseModel):
    """One completed turn — what the conversation service returns.

    ``outcome`` is the R5 observability lever (the router classification
    that produced this reply); ``runs`` are the turn's workflow results.
    """

    conversation_id: UUID
    message_id: UUID
    reply: str
    outcome: TurnOutcome
    runs: list[WorkflowResult] = Field(default_factory=list)
