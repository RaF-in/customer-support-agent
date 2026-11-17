"""Self-RAG knowledge-retrieval workflow (T4/R4) — self-registers as ``self_rag``.

One registration entry is this workflow's entire integration with the rest
of the system: the runtime facade discovers it by name (A2), and no other
graph, the registry, or the runtime knows anything about it (R2).
"""

from app.graphs.knowledge.graph import build_self_rag_graph
from app.graphs.knowledge.state import SelfRagInput, SelfRagState
from app.graphs.registry import WorkflowDefinition, register

register(
    WorkflowDefinition(
        name="self_rag",
        description=(
            "Self-RAG knowledge retrieval — decide-retrieval, relevance/"
            "groundedness/usefulness grading, bounded revision and question-"
            "rewrite loops (reference architecture)"
        ),
        factory=build_self_rag_graph,
        input_schema=SelfRagInput,
    )
)

__all__ = ["SelfRagInput", "SelfRagState", "build_self_rag_graph"]
