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
