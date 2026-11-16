"""Pluggable embedding interface — one Protocol every provider implements.

Callers adapt to the provider through the read-only properties instead of
knowing the implementation: ``dim`` binds Qdrant collection vector sizes,
``max_input_length`` lets chunking split oversized inputs (T5), and
``model_id`` is written into chunk metadata as the durable model record.
"""

from typing import Protocol, runtime_checkable

Embedding = list[float]


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Turns a batch of texts into fixed-size vectors."""

    def embed(self, texts: list[str]) -> list[Embedding]:
        """Embed ``texts``; returns exactly one vector per input, in order."""
        ...

    @property
    def dim(self) -> int:
        """Vector dimensionality (e.g. 1536 for ``text-embedding-3-small``)."""
        ...

    @property
    def model_id(self) -> str:
        """Provider-native model identifier (stored in chunk metadata)."""
        ...

    @property
    def max_input_length(self) -> int:
        """Maximum accepted input length in tokens; longer inputs are split."""
        ...
