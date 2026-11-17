"""Self-RAG graph state + input contract (T4/R4).

``SelfRagState`` extends the shared envelope with the knowledge graph's own
channels: the question under treatment (rewritten between attempts), the
retrieved passage context, the current answer draft, the four grader
decisions, and the two loop counters whose bounds are checked on the
conditional edges (A12). Terminal nodes write the runtime's outcome keys
(``status`` / ``output``) — internal channels never leak into the result.
"""

from typing import Any

from pydantic import BaseModel, Field

from app.graphs.base.state import WorkflowState, WorkflowStatus


class SelfRagInput(BaseModel):
    """Input contract for ``self_rag`` runs — one question, nothing else.

    ``tenant_id`` is deliberately absent: the runtime injects it from the
    caller's argument and overwrites any same-named key (R3).
    """

    question: str = Field(min_length=1, description="The question to answer")


class SelfRagState(WorkflowState, total=False):
    """Envelope extension for the self-RAG graph.

    All extension keys are optional (``total=False``): nodes merge partial
    updates, and the runtime injects the envelope keys at invocation time.
    """

    question: str
    # decide_retrieval
    needs_retrieval: bool
    # retrieve — passages as returned by the retrieval service
    context: list[dict[str, Any]]
    # generation nodes — latest draft wins
    generation: str
    # grader decisions (latest grading of each loop)
    relevant: bool
    supported: bool
    useful: bool
    # loop counters (A12 — bounds checked on the conditional edges)
    revision_count: int
    rewrite_count: int
    # outcome keys — written by terminal nodes only (runtime contract)
    status: WorkflowStatus
    output: Any
