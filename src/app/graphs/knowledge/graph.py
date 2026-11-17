"""Self-RAG graph wiring — the exact topology of ``self_rag.png`` (T4/R4).

Node names normalized per the ARCH:

    __start__ → decide_retrieval ──no──→ generate_direct → END
                    │yes
                    ▼
                 retrieve ←────────────────────┐
                    ▼                          │ (rewrite_question loop,
                 is_relevant ──irrelevant──→ no_answer_found   ≤ SELF_RAG_MAX_REWRITES)
                    ▼ relevant
        generate_from_context → is_sup ──unsupported──→ revise_answer ─┐
                                 ▲          (≤ SELF_RAG_MAX_REVISIONS,  │
                                 └──────────── then no_answer_found)    │
                               supported                                   │
                                 ▼                                        │
                              is_use ──not useful──→ rewrite_question ───┘
                                 ▼ useful
                                 END

Loop bounding (A12): both cycles are guarded on their grading node's
conditional edge — a further iteration only starts while the counter is
still under the settings bound, so ``revise_answer`` executes at most
``SELF_RAG_MAX_REVISIONS`` times per attempt and ``rewrite_question`` at
most ``SELF_RAG_MAX_REWRITES`` times per run; when a budget is spent the
run diverts to ``no_answer_found`` instead. The process-wide
``recursion_limit`` (runtime, T3) is the backstop, never the mechanism.
Bounds are read from settings at routing time — never hardcoded.
"""

import logging

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.graphs.knowledge import nodes
from app.graphs.knowledge.state import SelfRagState
from app.infrastructure.settings import Settings

logger = logging.getLogger(__name__)

# Decision keys used by the routers below (values written by grader nodes).
_AFTER_DECIDE = "decide_retrieval"
_RETRIEVE = "retrieve"
_GENERATE_DIRECT = "generate_direct"


def route_after_decide(state: SelfRagState) -> str:
    """yes/no on retrieval (first fork of the reference architecture)."""
    return _RETRIEVE if state.get("needs_retrieval") else _GENERATE_DIRECT


def route_after_relevance(state: SelfRagState) -> str:
    """Irrelevant (or empty) context → the no-answer terminal."""
    if state.get("relevant"):
        return "generate_from_context"
    return "no_answer_found"


def route_after_groundedness(state: SelfRagState) -> str:
    """Unsupported → revise while the revision budget lasts, else stop."""
    if state.get("supported"):
        return "is_use"
    if state.get("revision_count", 0) < Settings().SELF_RAG_MAX_REVISIONS:
        return "revise_answer"
    return "no_answer_found"


def route_after_usefulness(state: SelfRagState) -> str:
    """Not useful → rewrite while the rewrite budget lasts, else stop."""
    if state.get("useful"):
        return "end"
    if state.get("rewrite_count", 0) < Settings().SELF_RAG_MAX_REWRITES:
        return "rewrite_question"
    return "no_answer_found"


def build_self_rag_graph() -> CompiledStateGraph:
    """Wire and compile the Self-RAG knowledge graph (registry factory)."""
    builder = StateGraph(SelfRagState)

    builder.add_node("decide_retrieval", nodes.decide_retrieval_node)
    builder.add_node("retrieve", nodes.retrieve_node)
    builder.add_node("generate_direct", nodes.generate_direct_node)
    builder.add_node("is_relevant", nodes.is_relevant_node)
    builder.add_node("generate_from_context", nodes.generate_from_context_node)
    builder.add_node("is_sup", nodes.is_sup_node)
    builder.add_node("revise_answer", nodes.revise_answer_node)
    builder.add_node("is_use", nodes.is_use_node)
    builder.add_node("rewrite_question", nodes.rewrite_question_node)
    builder.add_node("no_answer_found", nodes.no_answer_found_node)

    builder.add_edge(START, "decide_retrieval")
    builder.add_conditional_edges(
        "decide_retrieval",
        route_after_decide,
        {_RETRIEVE: "retrieve", _GENERATE_DIRECT: "generate_direct"},
    )
    builder.add_edge("retrieve", "is_relevant")
    builder.add_conditional_edges(
        "is_relevant",
        route_after_relevance,
        {"generate_from_context": "generate_from_context", "no_answer_found": "no_answer_found"},
    )
    builder.add_edge("generate_from_context", "is_sup")
    builder.add_conditional_edges(
        "is_sup",
        route_after_groundedness,
        {
            "revise_answer": "revise_answer",
            "is_use": "is_use",
            "no_answer_found": "no_answer_found",
        },
    )
    builder.add_edge("revise_answer", "is_sup")
    builder.add_conditional_edges(
        "is_use",
        route_after_usefulness,
        {
            "end": END,
            "rewrite_question": "rewrite_question",
            "no_answer_found": "no_answer_found",
        },
    )
    builder.add_edge("rewrite_question", "retrieve")
    builder.add_edge("generate_direct", END)
    builder.add_edge("no_answer_found", END)

    return builder.compile()
