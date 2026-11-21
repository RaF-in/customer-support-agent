from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application configuration, loaded from environment variables and `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Required
    OPENAI_API_KEY: str

    # Infrastructure
    QDRANT_URL: str = "http://localhost:6333"
    EMBEDDING_PROVIDER: str = "openai"

    # Tunable limits (A10 — explicit and configurable)
    MAX_FILE_SIZE_MB: int = Field(default=200, gt=0)
    MAX_PAGES: int = Field(default=1000, gt=0)
    CHUNK_SIZE: int = Field(default=500, gt=0)
    CHUNK_OVERLAP: int = Field(default=50, ge=0)
    EMBEDDING_BATCH_SIZE: int = Field(default=100, gt=0)

    # Concurrency caps for remote I/O (async data layer): per-call asyncio
    # semaphores are sized from these — max gathered Qdrant requests per
    # operation, and max concurrent OpenAI embedding requests.
    QDRANT_MAX_CONCURRENCY: int = Field(default=8, ge=1)
    EMBEDDING_MAX_CONCURRENCY: int = Field(default=4, ge=1)

    # LLM / chat models (graphs subsystem — purpose-keyed factory)
    LLM_PROVIDER: str = "openai"
    CHAT_MODEL: str = "gpt-4o-mini"

    # Self-RAG bounds (A12 — every loop is settings-driven, never hardcoded;
    # defaults reproduce the ARCH latency budget: 2 calls on the direct path,
    # ≤ ~25 = 1 decide + 3 attempts × 8 grading/generation calls)
    SELF_RAG_MAX_REWRITES: int = Field(default=2, ge=0)  # 0 disables the loop
    SELF_RAG_MAX_REVISIONS: int = Field(default=2, ge=0)  # 0 disables the loop
    SELF_RAG_TOP_K: int = Field(default=4, ge=1)
    LIVE_DATA_MAX_STEPS: int = Field(default=8, ge=1)  # agent ↔ ToolNode rounds

    # Conversational orchestrator bounds (conversation-orchestrator ARCH A13
    # — same discipline: every cycle is settings-driven, never hardcoded).
    # The history lever reads 0 = unlimited (full transcript to the router,
    # A14); POSTGRES_POOL_MAX_SIZE is the concurrent-turns lever — each
    # active turn holds one pool connection for its advisory lock.
    CONVERSATION_MAX_MESSAGE_CHARS: int = Field(default=8000, gt=0)
    CONVERSATION_HISTORY_MAX_MESSAGES: int = Field(default=0, ge=0)  # 0 = unlimited
    CLARIFY_MAX_CONSECUTIVE: int = Field(default=2, ge=1)
    PLANNER_MAX_ATTEMPTS: int = Field(default=3, ge=1)
    EXECUTOR_MAX_REINVOCATIONS: int = Field(default=3, ge=0)
    PLAN_MAX_STEPS: int = Field(default=8, ge=1)
    POSTGRES_POOL_MAX_SIZE: int = Field(default=10, ge=1)

    # MCP tool servers (JSON, langchain-mcp-adapters client format) and the
    # per-tenant tool allowlist (JSON: tenant -> [tool names]). Deny-by-
    # default: a tenant absent from the allowlist gets NO tools.
    MCP_SERVERS: dict[str, Any] = Field(default_factory=dict)
    MCP_TENANT_ALLOWLIST: dict[str, list[str]] = Field(default_factory=dict)

    # LangGraph checkpointer (approval-graph persistence, A7): sqlite needs
    # no extra infra (single-writer dev limit); postgres is the default
    # compose service (conversation transcripts) and doubles as the
    # multi-worker checkpointer backend.
    CHECKPOINT_BACKEND: str = "sqlite"
    SQLITE_CHECKPOINT_PATH: str = "checkpoints.sqlite"
    POSTGRES_DSN: str = "postgresql://app:app@localhost:5432/app"
