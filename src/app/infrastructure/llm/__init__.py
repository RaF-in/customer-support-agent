"""Chat-model access for the graphs subsystem — purpose-keyed factory."""

from app.infrastructure.llm.factory import get_chat_model, set_chat_model

__all__ = ["get_chat_model", "set_chat_model"]
