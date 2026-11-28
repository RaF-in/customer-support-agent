from typing import Any

from pydantic import Field, model_validator
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
    # POSTGRES_POOL_MAX_SIZE is the concurrent-turns lever — each active
    # turn holds one pool connection for its advisory lock. (History is
    # bounded by the COMPACTION_* token budgets below, not by a row count.)
    CONVERSATION_MAX_MESSAGE_CHARS: int = Field(default=8000, gt=0)
    CLARIFY_MAX_CONSECUTIVE: int = Field(default=2, ge=1)
    PLANNER_MAX_ATTEMPTS: int = Field(default=3, ge=1)
    EXECUTOR_MAX_REINVOCATIONS: int = Field(default=3, ge=0)
    PLAN_MAX_STEPS: int = Field(default=8, ge=1)
    POSTGRES_POOL_MAX_SIZE: int = Field(default=10, ge=1)

    # Long-term memory (LM ARCH A9): the master switch — the one-env-change
    # rollback lever, off stops both the turn-start card read and the
    # post-turn extraction — and the card cap (the prompt-budget knob; the
    # card is pure recency in v1, so the cap is also the retention bound).
    MEMORY_ENABLED: bool = True
    MEMORY_MAX_FACTS: int = Field(default=50, ge=1)

    # Conversation compaction (conversation-compaction ARCH A14): history
    # handed to prompts is an exact action ledger + an incremental summary
    # + recent turns verbatim, bounded in tokens. COMPACTION_ENABLED is the
    # rollback lever — false renders the pre-feature history and schedules
    # no summarization. The soft budget triggers summarization and caps the
    # verbatim tail; the hard cap is what the read path never exceeds except
    # for a single oversized turn (soft <= hard, checked below).
    COMPACTION_ENABLED: bool = True
    COMPACTION_SOFT_TOKENS: int = Field(default=4000, ge=1)
    COMPACTION_HARD_TOKENS: int = Field(default=8000, ge=1)
    COMPACTION_KEEP_TURNS: int = Field(default=12, ge=1)
    COMPACTION_MAX_SUMMARY_CHARS: int = Field(default=4000, ge=1)
    COMPACTION_LEDGER_LINE_MAX_CHARS: int = Field(default=300, ge=1)

    # MCP tool servers (JSON, langchain-mcp-adapters client format) and the
    # per-tenant tool allowlist (JSON: tenant -> [tool names]). Deny-by-
    # default: a tenant absent from the allowlist gets NO tools.
    MCP_SERVERS: dict[str, Any] = Field(default_factory=dict)
    MCP_TENANT_ALLOWLIST: dict[str, list[str]] = Field(default_factory=dict)

    # Postgres — the row-state substrate: conversation transcripts, plan
    # working records, and (since the checkpointer was deleted) approval
    # request rows and action-result audit. A default compose dependency.
    #
    # Two roles, because tenant isolation is a row-level-security policy and
    # admins ignore those: POSTGRES_DSN is the pool's everyday role (subject
    # to the policies), POSTGRES_ADMIN_DSN the owner that creates the schema
    # at boot. open_postgres refuses a POSTGRES_DSN role that could bypass
    # the policies. All three must match docker-compose.yml.
    POSTGRES_DSN: str = "postgresql://app_rw:app_rw@localhost:5432/app"
    POSTGRES_ADMIN_DSN: str = "postgresql://app:app@localhost:5432/app"
    POSTGRES_APP_ROLE: str = Field(default="app_rw", min_length=1)

    @model_validator(mode="after")
    def _compaction_budgets_are_ordered(self) -> "Settings":
        """The soft budget must fit inside the hard cap it is bounded by."""
        if self.COMPACTION_SOFT_TOKENS > self.COMPACTION_HARD_TOKENS:
            raise ValueError(
                "COMPACTION_SOFT_TOKENS "
                f"({self.COMPACTION_SOFT_TOKENS}) must be less than or equal "
                "to COMPACTION_HARD_TOKENS "
                f"({self.COMPACTION_HARD_TOKENS})"
            )
        return self
