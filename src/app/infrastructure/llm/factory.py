"""Purpose-keyed chat-model factory + process-wide singletons (A6).

``get_chat_model(purpose)`` returns the shared chat model for a *purpose* —
``grader`` (temperature 0: deterministic structured grading), ``generator``
(0.3: answer generation and revision), ``agent`` (0.2: tool-calling) —
built by the provider named by ``LLM_PROVIDER`` and named by ``CHAT_MODEL``.
Unknown purposes and unknown providers fail fast with a clear message,
exactly like the embedding factory. Graphs never import a provider class
directly — all LLM access flows through here.

``set_chat_model(purpose, model)`` is the override seam: it replaces the
cached model for a purpose so run-once checklist verification can drive
every LLM-dependent path with langchain-core fake chat models injected at
this one point (developer decision, 2026-11-17). It exists for
verification and tests — never call it from production code paths.
"""

import logging
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from app.infrastructure.settings import Settings

logger = logging.getLogger(__name__)

# Purpose -> sampling temperature. The closed vocabulary of purposes graphs
# may request (A6: graders deterministic, generator more creative, agent
# balanced).
CHAT_MODEL_PURPOSES: dict[str, float] = {
    "grader": 0.0,
    "generator": 0.3,
    "agent": 0.2,
}

_CHAT_PROVIDERS: dict[str, type[BaseChatModel]] = {
    "openai": ChatOpenAI,
}

# Process-wide singleton per purpose — also the seam target set_chat_model
# writes into.
_chat_models: dict[str, BaseChatModel] = {}


def get_chat_model(purpose: str, settings: Optional[Settings] = None) -> BaseChatModel:
    """Return the shared chat model for ``purpose`` (cached per purpose).

    Raises ValueError immediately on an unknown purpose or an unknown
    provider so a bad config never survives past first use.
    """
    try:
        temperature = CHAT_MODEL_PURPOSES[purpose]
    except KeyError:
        known = ", ".join(sorted(CHAT_MODEL_PURPOSES))
        raise ValueError(
            f"Unknown chat-model purpose {purpose!r}. Known purposes: {known}."
        ) from None
    cached = _chat_models.get(purpose)
    if cached is not None:
        return cached
    settings = settings or Settings()
    try:
        provider_cls = _CHAT_PROVIDERS[settings.LLM_PROVIDER]
    except KeyError:
        known = ", ".join(sorted(_CHAT_PROVIDERS))
        raise ValueError(
            f"Unknown chat provider {settings.LLM_PROVIDER!r}. "
            f"Known providers: {known}."
        ) from None
    model = provider_cls(
        model=settings.CHAT_MODEL,
        temperature=temperature,
        api_key=settings.OPENAI_API_KEY,
    )
    _chat_models[purpose] = model
    logger.info(
        "Chat model ready for purpose %r (provider=%s, model=%s, temperature=%s)",
        purpose,
        settings.LLM_PROVIDER,
        settings.CHAT_MODEL,
        temperature,
    )
    return model


def set_chat_model(purpose: str, model: BaseChatModel) -> None:
    """Override the cached chat model for ``purpose`` (verification seam).

    Subsequent ``get_chat_model(purpose)`` calls return ``model`` — this is
    how checklist scripts inject langchain-core fake chat models without
    touching graph code.
    """
    if purpose not in CHAT_MODEL_PURPOSES:
        known = ", ".join(sorted(CHAT_MODEL_PURPOSES))
        raise ValueError(
            f"Unknown chat-model purpose {purpose!r}. Known purposes: {known}."
        )
    _chat_models[purpose] = model
