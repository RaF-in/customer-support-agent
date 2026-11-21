"""Purpose-keyed chat-model factory + process-wide singletons (A6).

``get_chat_model(purpose)`` returns the shared chat model for a *purpose* —
``grader`` (temperature 0: deterministic structured grading), ``generator``
(0.3: answer generation and revision), ``agent`` (0.2: tool-calling),
``router`` (0.0: deterministic turn classification), ``planner`` (0.0:
deterministic plan generation) —
built by the provider named by ``LLM_PROVIDER`` and named by ``CHAT_MODEL``,
authenticating with that provider's own key (``_CHAT_API_KEY_ATTRS`` —
never a single hardcoded env var). Unknown purposes and unknown providers
fail fast with a clear message, exactly like the embedding factory. Graphs
never import a provider class directly — all LLM access flows through here.

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
# balanced; router/planner deterministic — conversation-orchestrator A5/A10).
CHAT_MODEL_PURPOSES: dict[str, float] = {
    "grader": 0.0,
    "generator": 0.3,
    "agent": 0.2,
    "router": 0.0,
    "planner": 0.0,
}

_CHAT_PROVIDERS: dict[str, type[BaseChatModel]] = {
    "openai": ChatOpenAI,
}

# Provider -> the Settings attribute (same name as its env var) carrying
# that provider's API key. Registering a new provider is three lines: the
# class in ``_CHAT_PROVIDERS``, its key attribute here, and an optional
# Settings field (e.g. ``ANTHROPIC_API_KEY``) — no other call site changes.
# ``openai``'s key stays a *required* Settings field because the embedding
# factory reads it too.
_CHAT_API_KEY_ATTRS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
}


def _provider_api_key(settings: Settings, provider: str) -> str:
    """Resolve ``provider``'s API key from Settings (fail-fast, no default).

    A provider missing from the key map, a key attribute Settings does not
    declare, or an empty value all raise immediately with instructions — a
    missing credential must surface here at construction time, never as a
    confusing 401 from the provider at call time.
    """
    try:
        attr = _CHAT_API_KEY_ATTRS[provider]
    except KeyError:
        raise ValueError(
            f"Provider {provider!r} has no API-key mapping. Add its key"
            " attribute to _CHAT_API_KEY_ATTRS in app/infrastructure/llm/"
            "factory.py."
        ) from None
    value = getattr(settings, attr, None)
    if not value:
        raise ValueError(
            f"Provider {provider!r} requires a non-empty {attr}. Set the"
            " environment variable (or .env entry); if Settings does not"
            " declare that field yet, add it first."
        )
    return value

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
        api_key=_provider_api_key(settings, settings.LLM_PROVIDER),
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
