"""Dispatchers and factories — the routing layer between the pipeline and
the per-format handlers (A6: Factory + Strategy).

Each pipeline stage has one dispatcher routing on the ``DataCategory``:
cleaning takes the category from the upload's filename suffix, chunking and
embedding from the cleaned document's ``format`` field. Unknown formats
raise — never a silent wrong route. Adding a format = new handlers + one
factory branch; zero pipeline or API changes.
"""

import logging
from abc import ABC, abstractmethod
from typing import ClassVar
from uuid import UUID

from app.application.preprocessing.chunking_data_handlers import (
    ChunkingDataHandler,
    DocxChunkingHandler,
    PdfChunkingHandler,
    TextChunkingHandler,
)
from app.application.preprocessing.cleaning_data_handlers import (
    CleaningDataHandler,
    DocxCleaningHandler,
    PdfCleaningHandler,
    TextCleaningHandler,
)
from app.application.preprocessing.embedding_data_handlers import (
    DocxEmbeddingHandler,
    EmbeddingDataHandler,
    PdfEmbeddingHandler,
    TextEmbeddingHandler,
)
from app.domain.chunks import Chunk, EmbeddedChunk
from app.domain.documents import CleanedDocument
from app.domain.enums import DataCategory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Factories — one create_handler per format per stage
# ---------------------------------------------------------------------------


class CleaningDataHandlerFactory(ABC):
    """Creates the cleaning handler for one format."""

    @staticmethod
    @abstractmethod
    def create_handler() -> CleaningDataHandler:
        ...


class PdfCleaningFactory(CleaningDataHandlerFactory):
    @staticmethod
    def create_handler() -> PdfCleaningHandler:
        return PdfCleaningHandler()


class TextCleaningFactory(CleaningDataHandlerFactory):
    @staticmethod
    def create_handler() -> TextCleaningHandler:
        return TextCleaningHandler()


class DocxCleaningFactory(CleaningDataHandlerFactory):
    @staticmethod
    def create_handler() -> DocxCleaningHandler:
        return DocxCleaningHandler()


class ChunkingDataHandlerFactory(ABC):
    """Creates the chunking handler for one format."""

    @staticmethod
    @abstractmethod
    def create_handler() -> ChunkingDataHandler:
        ...


class PdfChunkingFactory(ChunkingDataHandlerFactory):
    @staticmethod
    def create_handler() -> PdfChunkingHandler:
        return PdfChunkingHandler()


class TextChunkingFactory(ChunkingDataHandlerFactory):
    @staticmethod
    def create_handler() -> TextChunkingHandler:
        return TextChunkingHandler()


class DocxChunkingFactory(ChunkingDataHandlerFactory):
    @staticmethod
    def create_handler() -> DocxChunkingHandler:
        return DocxChunkingHandler()


class EmbeddingDataHandlerFactory(ABC):
    """Creates the embedding handler for one format."""

    @staticmethod
    @abstractmethod
    def create_handler() -> EmbeddingDataHandler:
        ...


class PdfEmbeddingFactory(EmbeddingDataHandlerFactory):
    @staticmethod
    def create_handler() -> PdfEmbeddingHandler:
        return PdfEmbeddingHandler()


class TextEmbeddingFactory(EmbeddingDataHandlerFactory):
    @staticmethod
    def create_handler() -> TextEmbeddingHandler:
        return TextEmbeddingHandler()


class DocxEmbeddingFactory(EmbeddingDataHandlerFactory):
    @staticmethod
    def create_handler() -> DocxEmbeddingHandler:
        return DocxEmbeddingHandler()


# ---------------------------------------------------------------------------
# Dispatchers — route each stage to the right factory
# ---------------------------------------------------------------------------


class CleaningDispatcher:
    """Routes the EXTRACT/CLEAN stage by ``DataCategory``."""

    _factories: ClassVar[dict[DataCategory, type[CleaningDataHandlerFactory]]] = {
        DataCategory.PDF: PdfCleaningFactory,
        DataCategory.TEXT: TextCleaningFactory,
        DataCategory.DOCX: DocxCleaningFactory,
    }

    @classmethod
    def dispatch(
        cls,
        *,
        content: bytes,
        category: DataCategory,
        document_id: UUID,
        tenant_id: str,
    ) -> CleanedDocument:
        factory = cls._factories.get(category)  # type: ignore[arg-type]
        if factory is None:
            supported = sorted(c.value for c in cls._factories)
            raise ValueError(
                f"Unsupported data category for cleaning: {category!r}. "
                f"Supported: {supported}."
            )
        handler = factory.create_handler()
        logger.debug(
            "Cleaning dispatch: %s -> %s", category, type(handler).__name__
        )
        return handler.clean(
            content=content, document_id=document_id, tenant_id=tenant_id
        )


class ChunkingDispatcher:
    """Routes the CHUNK stage on the cleaned document's ``format``."""

    _factories: ClassVar[
        dict[DataCategory, type[ChunkingDataHandlerFactory]]
    ] = {
        DataCategory.PDF: PdfChunkingFactory,
        DataCategory.TEXT: TextChunkingFactory,
        DataCategory.DOCX: DocxChunkingFactory,
    }

    @classmethod
    def dispatch(cls, *, data_model: CleanedDocument) -> list[Chunk]:
        factory = cls._factories.get(data_model.format)
        if factory is None:
            supported = sorted(c.value for c in cls._factories)
            raise ValueError(
                f"Unsupported format for chunking: {data_model.format!r}. "
                f"Supported: {supported}."
            )
        handler = factory.create_handler()
        logger.debug(
            "Chunking dispatch: %s -> %s",
            data_model.format,
            type(handler).__name__,
        )
        return handler.chunk(data_model=data_model)


class EmbeddingDispatcher:
    """Routes the EMBED stage on the cleaned document's ``format``."""

    _factories: ClassVar[
        dict[DataCategory, type[EmbeddingDataHandlerFactory]]
    ] = {
        DataCategory.PDF: PdfEmbeddingFactory,
        DataCategory.TEXT: TextEmbeddingFactory,
        DataCategory.DOCX: DocxEmbeddingFactory,
    }

    @classmethod
    async def dispatch(
        cls, *, data_model: CleanedDocument, chunks: list[Chunk]
    ) -> list[EmbeddedChunk]:
        factory = cls._factories.get(data_model.format)
        if factory is None:
            supported = sorted(c.value for c in cls._factories)
            raise ValueError(
                f"Unsupported format for embedding: {data_model.format!r}. "
                f"Supported: {supported}."
            )
        handler = factory.create_handler()
        logger.debug(
            "Embedding dispatch: %s -> %s",
            data_model.format,
            type(handler).__name__,
        )
        return await handler.embed(data_model=data_model, chunks=chunks)
