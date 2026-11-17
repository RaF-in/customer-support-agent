"""Structured-output graders for the Self-RAG reflection loops (A6/R4).

Every grader wraps the purpose-keyed ``grader`` chat model in
``with_structured_output`` so each decision is a validated boolean — never
free text to parse. The structured model is built lazily *per call*: the
factory singleton is resolved at grading time, which is what makes the T1
``set_chat_model`` seam able to control every decision in run-once
verification. All LLM access flows through the factory — graph code never
imports a provider class (ARCH Patterns).
"""

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.infrastructure.llm.factory import get_chat_model

logger = logging.getLogger(__name__)


class RetrievalDecision(BaseModel):
    """``decide_retrieval`` output."""

    retrieve: bool = Field(
        description=(
            "True when answering the question requires retrieved knowledge-base"
            " passages rather than the model's own knowledge"
        )
    )


class RelevanceGrade(BaseModel):
    """``is_relevant`` output."""

    relevant: bool = Field(
        description="True when at least one retrieved passage is relevant to the question"
    )


class GroundednessGrade(BaseModel):
    """``is_sup`` output."""

    supported: bool = Field(
        description="True when every claim in the answer is supported by the retrieved passages"
    )


class UsefulnessGrade(BaseModel):
    """``is_use`` output."""

    useful: bool = Field(
        description="True when the answer actually addresses the user's question"
    )


def _format_passages(passages: list[dict[str, Any]]) -> str:
    """Render retrieved passages as a numbered block for grading prompts."""
    if not passages:
        return "(no passages were retrieved)"
    lines = [
        f"[{i}] {passage.get('content', '')}"
        for i, passage in enumerate(passages, start=1)
    ]
    return "\n".join(lines)


async def grade_retrieval_need(question: str) -> bool:
    """``decide_retrieval``: does this question need knowledge-base passages?"""
    model = get_chat_model("grader").with_structured_output(RetrievalDecision)
    decision = await model.ainvoke(
        [
            SystemMessage(
                "You decide whether answering a customer question requires"
                " retrieving passages from the product knowledge base. Pure"
                " small talk, greetings, or general questions the model can"
                " answer without company-specific facts do not need retrieval;"
                " anything about products, policies, billing, or procedures does."
            ),
            HumanMessage(f"Question: {question}"),
        ]
    )
    return bool(decision.retrieve)


async def grade_relevance(question: str, passages: list[dict[str, Any]]) -> bool:
    """``is_relevant``: is at least one retrieved passage relevant?"""
    model = get_chat_model("grader").with_structured_output(RelevanceGrade)
    grade = await model.ainvoke(
        [
            SystemMessage(
                "You grade retrieval usefulness: check whether at least one"
                " retrieved passage contains information relevant to answering"
                " the question. Grade strictly on topical relevance."
            ),
            HumanMessage(
                f"Question: {question}\n\nRetrieved passages:\n"
                f"{_format_passages(passages)}"
            ),
        ]
    )
    return bool(grade.relevant)


async def grade_groundedness(
    question: str, answer: str, passages: list[dict[str, Any]]
) -> bool:
    """``is_sup``: is the answer grounded in (supported by) the passages?"""
    model = get_chat_model("grader").with_structured_output(GroundednessGrade)
    grade = await model.ainvoke(
        [
            SystemMessage(
                "You grade groundedness: check whether the draft answer is"
                " supported by the retrieved passages — every factual claim"
                " must be traceable to them. Unsupportable claims mean not"
                " grounded."
            ),
            HumanMessage(
                f"Question: {question}\n\n"
                f"Retrieved passages:\n{_format_passages(passages)}\n\n"
                f"Draft answer: {answer}"
            ),
        ]
    )
    return bool(grade.supported)


async def grade_usefulness(question: str, answer: str) -> bool:
    """``is_use``: does the answer actually address the question?"""
    model = get_chat_model("grader").with_structured_output(UsefulnessGrade)
    grade = await model.ainvoke(
        [
            SystemMessage(
                "You grade usefulness: check whether the draft answer genuinely"
                " addresses the user's question. A grounded but evasive,"
                " incomplete, or off-topic answer is not useful."
            ),
            HumanMessage(f"Question: {question}\n\nDraft answer: {answer}"),
        ]
    )
    return bool(grade.useful)
