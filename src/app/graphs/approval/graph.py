"""Approval workflow wiring — the human gate (T7/R5/R9/R3).

    START → create_request → resolve_approver → await_decision ⏸
          → record_decision → END

``create_request`` persists the ``ApprovalRecord`` (``pending``,
``thread_id = request_id = record id``); ``await_decision`` pauses the run
via ``interrupt()`` — the checkpoint at that pause is what a fresh process
resumes; ``record_decision`` stamps the audit fields (status, decided_by,
decision_note, decided_at) onto the record and ends the run.

Two graphs, not one (A3): this graph never imports the action graph — on
``approved`` it simply ends, and the *runtime* re-invokes the
action-execution graph (approval_id set → re-validate, gate skipped →
execute → verify → record). All chaining lives in the runtime.

Failure classes (R9): a failed record persistence raises (infrastructure —
the runtime facade maps it to ``failed``); every business outcome —
including a rejection — is a terminal state, never an exception.

Deploy rule (README): LangGraph checkpoints are node-topology-sensitive —
drain or expire pending approvals before structural changes to this graph.
"""

import logging
from datetime import datetime
from typing import Any, Optional

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from app.domain.approvals import ApprovalRecord
from app.domain.base.vector import utc_now
from app.domain.enums import ApprovalStatus
from app.graphs.approval.state import ApprovalState

logger = logging.getLogger(__name__)

_VALID_DECISIONS = ("approved", "rejected")


async def create_request_node(state: ApprovalState) -> dict[str, Any]:
    """Persist the pending ``ApprovalRecord`` (id = request_id = thread_id).

    The gate already minted ``request_id``, so the record is written under
    that id — one identifier names the request everywhere (ARCH data
    model). A repeated id (which the gate's fresh-uuid4 minting makes
    impossible in practice) would upsert-overwrite benignly.
    """
    record = ApprovalRecord(
        id=state["request_id"],  # pydantic coerces the str uuid
        tenant_id=state["tenant_id"],
        action_name=state["action_name"],
        action_input=dict(state.get("action_input", {})),
        reason=state["reason"],
        risk_level=state["risk_level"],
        approver_role=state["approver_role"],
        status=ApprovalStatus.PENDING,
        thread_id=state["request_id"],
    )
    persisted = await ApprovalRecord.bulk_insert([record])
    if not persisted:
        raise RuntimeError(
            "approval request could not be persisted (vector db unavailable)"
        )
    logger.info(
        "approval request created (tenant=%s request_id=%s action=%s risk=%s)",
        state["tenant_id"],
        record.id,
        record.action_name,
        record.risk_level,
    )
    # The original creation time rides the checkpointed state so the
    # decided record keeps its true created_at after the pause.
    return {"created_at": record.created_at.isoformat()}


async def resolve_approver_node(state: ApprovalState) -> dict[str, Any]:
    """Resolve who is expected to decide, then hand off to the pause.

    MVP: no authentication exists (ARCH Open Question), so the expected
    approver resolves verbatim from the policy verdict's ``approver_role``
    and ``decided_by`` stays declared-audit-only. This node is the future
    role-resolution point (role registries, escalation, two-person rules) —
    the topology slot exists so that lands here without graph rewiring.
    """
    logger.info(
        "approval awaiting decision (tenant=%s request_id=%s approver_role=%s)",
        state["tenant_id"],
        state["request_id"],
        state["approver_role"],
    )
    return {}


async def await_decision_node(state: ApprovalState) -> dict[str, Any]:
    """Pause for the human decision; on resume, carry it into state.

    ``interrupt()`` raises the pause on first execution and returns the
    resume value (``{decision, decided_by, note}``) when the runtime
    continues the thread. A malformed resume value is a caller bug — it
    raises here (the resume facade maps it to ``failed``), never a silent
    default.
    """
    resume = interrupt(
        {
            "request_id": state["request_id"],
            "action_name": state["action_name"],
            "reason": state["reason"],
            "risk_level": state["risk_level"],
            "approver_role": state["approver_role"],
        }
    )
    if (
        not isinstance(resume, dict)
        or resume.get("decision") not in _VALID_DECISIONS
    ):
        raise ValueError(
            f"invalid resume value {resume!r}; expected "
            "{'decision': 'approved' | 'rejected', 'decided_by': str, 'note': str}"
        )
    return {
        "decision": resume["decision"],
        "decided_by": str(resume.get("decided_by", "")),
        "decision_note": resume.get("note"),
    }


async def record_decision_node(state: ApprovalState) -> dict[str, Any]:
    """Stamp the audit fields onto the record; end approved or rejected.

    The record is overwritten by id (Qdrant upsert): same point, now
    carrying status + ``decided_by`` + ``decision_note`` + ``decided_at``
    and its original ``created_at`` from the checkpoint. Only
    ``pending → decided`` ever reaches here — ``decide_approval`` holds the
    per-request lock and re-checks status *before* resuming; transition
    enforcement lives in the runtime, not in the record model.
    """
    approved = state["decision"] == "approved"
    created = state.get("created_at")
    record = ApprovalRecord(
        id=state["request_id"],
        tenant_id=state["tenant_id"],
        action_name=state["action_name"],
        action_input=dict(state.get("action_input", {})),
        reason=state["reason"],
        risk_level=state["risk_level"],
        approver_role=state["approver_role"],
        status=ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED,
        decision_note=state.get("decision_note"),
        decided_by=state.get("decided_by"),
        decided_at=utc_now(),
        thread_id=state["request_id"],
        created_at=datetime.fromisoformat(created) if created else utc_now(),
    )
    persisted = await ApprovalRecord.bulk_insert([record])
    if not persisted:
        raise RuntimeError(
            "approval decision could not be persisted (vector db unavailable)"
        )
    logger.info(
        "approval decided (tenant=%s request_id=%s decision=%s decided_by=%s)",
        state["tenant_id"],
        record.id,
        record.status.value,
        record.decided_by,
    )
    return {
        "output": {
            "request_id": state["request_id"],
            "decision": state["decision"],
            "decided_by": state.get("decided_by"),
        }
    }


def build_approval_graph(
    checkpointer: Optional[BaseCheckpointSaver] = None,
) -> CompiledStateGraph:
    """Wire and compile the approval graph.

    The checkpointer attaches at compile — passed by the runtime facade,
    which owns checkpointer wiring (ARCH Module Boundaries). Compiling
    without one produces a graph that cannot pause (``interrupt()`` writes
    no checkpoint to resume from), which is why this graph is never
    registry-run: only the facade launches it, always with the
    process-wide entered saver.
    """
    builder = StateGraph(ApprovalState)
    builder.add_node("create_request", create_request_node)
    builder.add_node("resolve_approver", resolve_approver_node)
    builder.add_node("await_decision", await_decision_node)
    builder.add_node("record_decision", record_decision_node)
    builder.add_edge(START, "create_request")
    builder.add_edge("create_request", "resolve_approver")
    builder.add_edge("resolve_approver", "await_decision")
    builder.add_edge("await_decision", "record_decision")
    builder.add_edge("record_decision", END)
    return builder.compile(checkpointer=checkpointer)
