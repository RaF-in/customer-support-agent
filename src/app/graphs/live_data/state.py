"""Live-data graph state + input contract (T5/R6).

``LiveDataState`` extends the shared envelope with the canonical agent-loop
channels: ``query`` (the operational request, seeded into the transcript on
the first agent turn), the running ``messages`` transcript (``add_messages``
reducer), and the step counter whose bound (``LIVE_DATA_MAX_STEPS``) stops
the agent ↔ ToolNode loop (A12).

The tenant's MCP tools ride in state for the life of one run: resolved by
the agent node's first entry, consumed by the agent (``bind_tools``) and
the prebuilt ``ToolNode``. They are deliberately transient — this graph has
no checkpointer, and if one is ever attached the tool objects must move out
of state (they are not serializable); noted here so that change is not
discovered late.
"""

from typing import Annotated, Any

from langchain_core.messages import AnyMessage
from langchain_core.tools import BaseTool
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from app.graphs.base.state import WorkflowState, WorkflowStatus


class LiveDataInput(BaseModel):
    """Input contract for ``live_data`` runs — one operational query.

    ``tenant_id`` is deliberately absent: the runtime injects it from the
    caller's argument and overwrites any same-named key (R3).
    """

    query: str = Field(
        min_length=1, description="The operational question to answer with MCP tools"
    )


class LiveDataState(WorkflowState, total=False):
    """Envelope extension for the live-data graph.

    All extension keys are optional (``total=False``): nodes merge partial
    updates, and the runtime injects the envelope keys at invocation time.
    """

    query: str
    messages: Annotated[list[AnyMessage], add_messages]
    # Tenant-filtered MCP tools, resolved on the agent's first entry
    # (transient — see the module docstring).
    tools: list[BaseTool]
    # Loop counter (A12): model invocations so far; bound =
    # LIVE_DATA_MAX_STEPS, checked on agent re-entry.
    step_count: int
    # Outcome keys — written by the agent node on terminal turns only
    # (runtime contract).
    status: WorkflowStatus
    output: Any
