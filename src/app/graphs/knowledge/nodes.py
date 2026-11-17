"""Node functions for the Self-RAG knowledge graph (T4/R4).

Orchestration only — no business rules, no HTTP (ARCH Module Boundaries).
Every node is a small async function over ``SelfRagState`` returning a
partial update. LLM nodes resolve their model through the purpose-keyed
factory at call time; the retrieve node is the only one touching data, and
it takes the tenant from the runtime-injected envelope and nothing else
(structural isolation, R3).

Failure classes (R9): "no passages retrieved" is a *valid* business outcome
that feeds the no-answer / rewrite paths; an infrastructure failure (Qdrant
or embedding unavailable) raises out of ``retrieve_passages`` and is caught
by the runtime facade as ``failed``.

Settings are read per call, never captured at import or compile time, so
environment overrides take effect on the next run (nothing hardcoded).
"""

import asyncio
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.application.retrieval import retrieve_passages
from app.graphs.base.state import WorkflowStatus
from app.graphs.knowledge import graders
from app.graphs.knowledge.state import SelfRagState
from app.infrastructure.llm.factory import get_chat_model
from app.infrastructure.settings import Settings

logger = logging.getLogger(__name__)


def _answer_output(answer: str, path: str) -> dict[str, Any]:
    """The output payload carried on answer paths (runtime outcome key)."""
    return {"answer": answer, "path": path, "reason": None}


async def decide_retrieval_node(state: SelfRagState) -> dict[str, Any]:
    """Grade whether the question needs knowledge-base retrieval at all."""
    needs_retrieval = await graders.grade_retrieval_need(state["question"])
    return {"needs_retrieval": needs_retrieval}


async def retrieve_node(state: SelfRagState) -> dict[str, Any]:
    """Retrieve this tenant's best passages for the current question.

    The tenant comes from the envelope only; the sync retrieval service
    (A9) runs in a worker thread. An empty list is valid — it feeds the
    no-answer path; infrastructure errors raise (→ runtime → ``failed``).
    """
    top_k = Settings().SELF_RAG_TOP_K
    passages = await asyncio.to_thread(
        retrieve_passages, state["tenant_id"], state["question"], top_k
    )
    return {"context": passages}


async def generate_direct_node(state: SelfRagState) -> dict[str, Any]:
    """Answer from the model's own knowledge (no retrieval)."""
    model = get_chat_model("generator")
    message = await model.ainvoke(
        [
            SystemMessage(
                "You are a customer-support answer writer. Answer the"
                " customer's question directly and concisely in the same"
                " language as the question."
            ),
            HumanMessage(f"Question: {state['question']}"),
        ]
    )
    answer = message.content if isinstance(message.content, str) else str(message.content)
    return {"generation": answer, "output": _answer_output(answer, "direct")}


async def is_relevant_node(state: SelfRagState) -> dict[str, Any]:
    """Grade whether any retrieved passage is relevant to the question."""
    relevant = await graders.grade_relevance(state["question"], state.get("context", []))
    return {"relevant": relevant}


async def generate_from_context_node(state: SelfRagState) -> dict[str, Any]:
    """Draft an answer grounded in the retrieved passages."""
    model = get_chat_model("generator")
    passages = graders._format_passages(state.get("context", []))
    message = await model.ainvoke(
        [
            SystemMessage(
                "You are a customer-support answer writer. Answer the"
                " customer's question using only the retrieved passages. If"
                " they do not contain the answer, say so plainly instead of"
                " inventing facts."
            ),
            HumanMessage(
                f"Question: {state['question']}\n\nRetrieved passages:\n{passages}"
            ),
        ]
    )
    answer = message.content if isinstance(message.content, str) else str(message.content)
    return {"generation": answer, "output": _answer_output(answer, "context")}


async def is_sup_node(state: SelfRagState) -> dict[str, Any]:
    """Grade whether the draft answer is supported by the passages."""
    supported = await graders.grade_groundedness(
        state["question"], state.get("generation", ""), state.get("context", [])
    )
    return {"supported": supported}


async def revise_answer_node(state: SelfRagState) -> dict[str, Any]:
    """Revise the unsupported draft toward grounding; count the revision."""
    model = get_chat_model("generator")
    passages = graders._format_passages(state.get("context", []))
    message = await model.ainvoke(
        [
            SystemMessage(
                "You revise draft answers for groundedness. Revise the draft"
                " so that every claim is supported by the retrieved passages,"
                " removing or correcting anything unsupported."
            ),
            HumanMessage(
                f"Question: {state['question']}\n\n"
                f"Retrieved passages:\n{passages}\n\n"
                f"Draft answer: {state.get('generation', '')}"
            ),
        ]
    )
    answer = message.content if isinstance(message.content, str) else str(message.content)
    return {
        "generation": answer,
        "revision_count": state.get("revision_count", 0) + 1,
        "output": _answer_output(answer, "context"),
    }


async def is_use_node(state: SelfRagState) -> dict[str, Any]:
    """Grade whether the grounded draft actually addresses the question."""
    useful = await graders.grade_usefulness(state["question"], state.get("generation", ""))
    return {"useful": useful}


async def rewrite_question_node(state: SelfRagState) -> dict[str, Any]:
    """Rewrite the question for a fresh retrieval attempt; count the rewrite.

    A rewritten attempt starts clean: the revision budget resets (each
    attempt gets its own ``SELF_RAG_MAX_REVISIONS``) and the previous
    attempt's context and draft are discarded.
    """
    model = get_chat_model("generator")
    message = await model.ainvoke(
        [
            SystemMessage(
                "You rewrite questions to improve retrieval. Rewrite the"
                " question into a stand-alone, search-friendly form that"
                " preserves the customer's intent; output only the rewritten"
                " question."
            ),
            HumanMessage(f"Question: {state['question']}"),
        ]
    )
    rewritten = message.content if isinstance(message.content, str) else str(message.content)
    return {
        "question": rewritten.strip() or state["question"],
        "rewrite_count": state.get("rewrite_count", 0) + 1,
        "revision_count": 0,
        "context": [],
        "generation": "",
    }


def _no_answer_reason(state: SelfRagState) -> str:
    """Infer why the no-answer terminal was reached, from the grader flags.

    Checked in loop order: irrelevance beats revision exhaustion beats
    rewrite exhaustion — matching which condition diverted the run here.
    """
    settings = Settings()
    if not state.get("relevant", False):
        return "no relevant passages were retrieved for this question"
    if not state.get("supported", False):
        return (
            "the answer could not be grounded in the retrieved passages"
            f" within {settings.SELF_RAG_MAX_REVISIONS} revisions"
        )
    return (
        "no useful answer was produced"
        f" within {settings.SELF_RAG_MAX_REWRITES} question rewrites"
    )


async def no_answer_found_node(state: SelfRagState) -> dict[str, Any]:
    """Terminal: the run ends honestly without an answer, with its reason."""
    reason = _no_answer_reason(state)
    logger.info(
        "self_rag ends without an answer (tenant=%s run_id=%s): %s",
        state["tenant_id"],
        state["run_id"],
        reason,
    )
    return {
        "status": WorkflowStatus.COMPLETED,
        "output": {"answer": None, "path": "none", "reason": reason},
    }
