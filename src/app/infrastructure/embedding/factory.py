"""Config-driven embedding provider selection + process-wide singleton.

``get_embedding_provider()`` builds the provider named by the
``EMBEDDING_PROVIDER`` setting — unknown names fail fast at startup with a
clear message (A5). ``get_embedding_model()`` is the cached singleton the
rest of the app consumes (handbook ``EmbeddingModelSingleton`` pattern).
"""

import logging
from typing import Optional

from app.infrastructure.embedding.base import EmbeddingProvider
from app.infrastructure.embedding.openai_provider import OpenAIEmbeddingProvider
from app.infrastructure.settings import Settings

logger = logging.getLogger(__name__)

_EMBEDDING_PROVIDERS: dict[str, type[EmbeddingProvider]] = {
    "openai": OpenAIEmbeddingProvider,
}


def get_embedding_provider(settings: Optional[Settings] = None) -> EmbeddingProvider:
    """Build the provider selected by ``EMBEDDING_PROVIDER``.

    Raises ValueError immediately on an unknown provider name so a bad
    config never survives past startup.
    """
    settings = settings or Settings()
    name = settings.EMBEDDING_PROVIDER
    try:
        provider_cls = _EMBEDDING_PROVIDERS[name]
    except KeyError:
        known = ", ".join(sorted(_EMBEDDING_PROVIDERS))
        raise ValueError(
            f"Unknown embedding provider {name!r}. Known providers: {known}."
        ) from None
    provider = provider_cls(settings)
    logger.info(
        "Embedding provider %r selected (model=%s, dim=%d)",
        name, provider.model_id, provider.dim,
    )
    return provider


_embedding_model: Optional[EmbeddingProvider] = None


def get_embedding_model() -> EmbeddingProvider:
    """Process-wide singleton — every caller shares one provider instance."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = get_embedding_provider()
    return _embedding_model
