"""OpenAI ``text-embedding-3-small`` — the MVP EmbeddingProvider (A5)."""

import logging
from typing import Optional

from openai import OpenAI

from app.infrastructure.embedding.base import Embedding, EmbeddingProvider
from app.infrastructure.settings import Settings

logger = logging.getLogger(__name__)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Batched embeddings via the OpenAI API.

    The OpenAI client is built eagerly in ``__init__`` from typed settings,
    so a missing ``OPENAI_API_KEY`` surfaces as a clear pydantic-settings
    error at construction — not as an opaque 401 deep inside ``embed()``.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or Settings()
        self._client = OpenAI(api_key=self._settings.OPENAI_API_KEY)

    @property
    def model_id(self) -> str:
        return "text-embedding-3-small"

    @property
    def dim(self) -> int:
        # Output dimensionality of text-embedding-3-small. This binds every
        # chunk collection's vector size (T2 note) — keep it accurate.
        return 1536

    @property
    def max_input_length(self) -> int:
        # Per-input token limit for this model (OpenAI docs); consumed by the
        # chunking handlers (T5) to split oversized chunks.
        return 8191

    def embed(self, texts: list[str]) -> list[Embedding]:
        """Embed texts in ``EMBEDDING_BATCH_SIZE`` batches. Raises on API
        failure — the pipeline (T6) catches and marks the document failed."""
        if not texts:
            return []
        batch_size = self._settings.EMBEDDING_BATCH_SIZE
        total_batches = (len(texts) + batch_size - 1) // batch_size
        vectors: list[Embedding] = []
        for number, start in enumerate(range(0, len(texts), batch_size), start=1):
            batch = texts[start : start + batch_size]
            response = self._client.embeddings.create(
                model=self.model_id, input=batch
            )
            vectors.extend(item.embedding for item in response.data)
            logger.info(
                "Embedding batch %d/%d: %d texts -> %d dims (model=%s)",
                number,
                total_batches,
                len(batch),
                self.dim,
                self.model_id,
            )
        return vectors
