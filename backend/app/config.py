"""Application configuration via environment variables."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """RAGForge settings. All values overridable via environment / .env file."""

    # App
    app_name: str = "RAGForge API"
    backend_url: str = "http://localhost:8000"
    frontend_url: str = "http://localhost:3000"
    debug: bool = True

    # Paths (relative to the backend/ directory by default)
    data_dir: Path = Path(__file__).resolve().parent.parent / "data"

    # Derived paths — computed from data_dir so DATA_DIR env override works.
    @property
    def db_path(self) -> Path:
        return self.data_dir / "ragforge.db"

    @property
    def documents_dir(self) -> Path:
        return self.data_dir / "documents"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None

    # LLM provider: "openai" | "anthropic" | "gemini" | "mock"
    llm_provider: str = ""
    llm_model: str = ""
    llm_api_key: str = ""
    llm_base_url: str = ""  # optional, for OpenAI-compatible local servers

    # Development mode: if true and no LLM is configured, a clearly-labelled
    # deterministic mock analyzer is used instead of failing hard.
    allow_mock_llm: bool = True

    # Embeddings: "sentence-transformers" | "openai"
    embedding_provider: str = "sentence-transformers"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_api_key: str = ""

    # Retrieval defaults
    default_top_k: int = 5

    model_config = {
        "env_file": str(Path(__file__).resolve().parent.parent / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache
def get_settings() -> Settings:
    return Settings()
