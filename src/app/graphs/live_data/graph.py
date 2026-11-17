"""Live-data graph wiring — the canonical prebuilt agent loop (T5/R6/R3).

    START → agent ──tool_calls──→ tools (prebuilt ToolNode) → agent → …
             │ no tool calls
             └────────────────────────────→ END

``tools_condition`` routes agent → tools whenever the model emits tool
calls, END otherwise — the prebuilt router, no custom dispatch (R6). The
agent node carries the only custom logic:

- **Tool resolution (R3):** on first entry it loads this tenant's MCP tools
  through ``infrastructure.mcp`` — tenant identity from the runtime-injected
  envelope and nowhere else. A deny-by-default refusal or an unreachable
  server raises here, so the run fails *before any model call* (no partial
  agent loop). Re-entries reuse the stashed tools: one tool-load per run,
  not one per step.
- **Step budget (A12):** ``step_count`` increments on every model
  invocation, and an agent re-entry at the bound stops the loop with an
  honest bounded-terminal state instead of another call. The process-wide
  ``recursion_limit`` (runtime, T3) is the backstop, never the mechanism.

Settings are read per call, never captured at import or compile time, so
environment overrides take effect on the next run (nothing hardcoded).
"""

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.graphs.base.state import WorkflowStatus
from app.graphs.live_data.state import LiveDataState
from app.infrastructure.llm.factory import get_chat_model
from app.infrastructure.mcp import load_mcp_tools
from app.infrastructure.settings import Settings

logger = logging.getLogger(__name__)


def _content(message: AIMessage) -> str:
    """Best-effort text of a model message (content is usually ``str``)."""
    return message.content if isinstance(message.content, str) else str(message.content)


async def agent_node(state: LiveDataState) -> dict[str, Any]:
    """Invoke the tool-bound agent model, or stop honestly at the budget.

    First entry seeds the transcript with the operational query, resolves
    the tenant's MCP tools, and binds them to the purpose-keyed agent model
    (A6 — all LLM access flows through the factory). A turn whose response
    carries no tool calls is the terminal answer turn: its text becomes the
    run's ``output``.
    """
    tools: list[BaseTool] = state.get("tools") or []
    if not tools:
        tools = await load_mcp_tools(state["tenant_id"])

    settings = Settings()
    steps = state.get("step_count", 0)
    if steps >= settings.LIVE_DATA_MAX_STEPS:
        reason = (
            f"live-data step budget exhausted after {steps} agent ↔ tool"
            f" rounds (LIVE_DATA_MAX_STEPS={settings.LIVE_DATA_MAX_STEPS})"
        )
        logger.info(
            "live_data stops at the step bound (tenant=%s run_id=%s)",
            state["tenant_id"],
            state["run_id"],
        )
        return {
            "tools": tools,
            "messages": [AIMessage(content=reason)],
            "status": WorkflowStatus.COMPLETED,
            "output": {"answer": None, "reason": reason, "steps": steps},
        }

    messages = state.get("messages") or [HumanMessage(content=state["query"])]
    model = get_chat_model("agent").bind_tools(tools)
    response = await model.ainvoke(messages)
    update: dict[str, Any] = {
        "tools": tools,
        "messages": [response],
        "step_count": steps + 1,
    }
    if not response.tool_calls:
        update["output"] = {"answer": _content(response), "reason": None}
    return update


async def tools_node(state: LiveDataState) -> dict[str, Any]:
    """Execute the requested tool calls via the prebuilt ``ToolNode`` (R6).

    A fresh ``ToolNode`` over the run's tenant-filtered tools per turn —
    tool execution, parallelism, and error handling stay the prebuilt's;
    this wrapper only supplies the per-tenant tool set (the one thing the
    compile-once graph cannot know at build time).
    """
    return await ToolNode(state["tools"]).ainvoke(state)


def build_live_data_graph() -> CompiledStateGraph:
    """Wire and compile the live-data agent loop (registry factory)."""
    builder = StateGraph(LiveDataState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tools_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", END: END},
    )
    builder.add_edge("tools", "agent")
    return builder.compile()
