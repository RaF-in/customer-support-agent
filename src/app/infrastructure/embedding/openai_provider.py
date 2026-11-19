"""OpenAI ``text-embedding-3-small`` — the MVP EmbeddingProvider (A5)."""

import asyncio
import logging
from typing import Optional

from openai import AsyncOpenAI

from app.infrastructure.embedding.base import Embedding, EmbeddingProvider
from app.infrastructure.settings import Settings

logger = logging.getLogger(__name__)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Batched embeddings via the OpenAI API, independent batches gathered
    under a per-call semaphore (``EMBEDDING_MAX_CONCURRENCY``).

    The OpenAI client is built eagerly in ``__init__`` from typed settings,
    so a missing ``OPENAI_API_KEY`` surfaces as a clear pydantic-settings
    error at construction — not as an opaque 401 deep inside ``embed()``.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or Settings()
        self._client = AsyncOpenAI(api_key=self._settings.OPENAI_API_KEY)

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

    async def embed(self, texts: list[str]) -> list[Embedding]:
        """Embed texts in ``EMBEDDING_BATCH_SIZE`` batches, gathered under a
        per-call semaphore. Vectors return positionally ordered — gather
        yields results in argument order, never completion order. Raises on
        API failure — the pipeline catches and marks the document failed."""
        if not texts:
            return []
        batch_size = self._settings.EMBEDDING_BATCH_SIZE
        batches = [
            texts[start : start + batch_size]
            for start in range(0, len(texts), batch_size)
        ]
        total_batches = len(batches)
        # Per-call semaphore from a fresh Settings() read (AD4) — the cap may
        # change between embed calls; no shared mutable state.
        semaphore = asyncio.Semaphore(Settings().EMBEDDING_MAX_CONCURRENCY)

        async def _embed_batch(number: int, batch: list[str]) -> list[Embedding]:
            async with semaphore:
                response = await self._client.embeddings.create(
                    model=self.model_id, input=batch
                )
            logger.info(
                "Embedding batch %d/%d: %d texts -> %d dims (model=%s)",
                number,
                total_batches,
                len(batch),
                self.dim,
                self.model_id,
            )
            return [item.embedding for item in response.data]

        # results[i] is batch i's vectors (gather preserves argument order);
        # the ordered flatten keeps the output aligned with ``texts``.
        batch_vectors = await asyncio.gather(
            *(
                _embed_batch(number, batch)
                for number, batch in enumerate(batches, start=1)
            )
        )
        return [vector for vectors in batch_vectors for vector in vectors]
