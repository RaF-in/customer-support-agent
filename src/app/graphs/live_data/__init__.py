"""Live-data fetch workflow (T5/R6) — self-registers as ``live_data``.

One registration entry is this workflow's entire integration with the rest
of the system: the runtime facade discovers it by name (A2), and no other
graph, the registry, or the runtime knows anything about it (R2).
"""

from app.graphs.live_data.graph import build_live_data_graph
from app.graphs.live_data.state import LiveDataInput, LiveDataState
from app.graphs.registry import WorkflowDefinition, register

register(
    WorkflowDefinition(
        name="live_data",
        description=(
            "Live operational-data fetch — tool-calling agent over"
            " tenant-allowlisted MCP tools (agent + prebuilt ToolNode +"
            " tools_condition), bounded by LIVE_DATA_MAX_STEPS"
        ),
        factory=build_live_data_graph,
        input_schema=LiveDataInput,
    )
)

__all__ = ["LiveDataInput", "LiveDataState", "build_live_data_graph"]
