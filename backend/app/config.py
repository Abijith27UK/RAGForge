"""Application configuration via environment variables."""
import logging
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


def normalize_loopback_url(url: str) -> str:
    """Rewrite a ``localhost`` host to ``127.0.0.1``; leave everything else alone.

    Why this exists (V22, measured): on Windows the name ``localhost`` resolves to
    ``::1`` BEFORE ``127.0.0.1``, while a locally started Qdrant listens on IPv4
    only. Every HTTP call therefore paid a fixed ~2.0 s connect timeout on
    ``::1`` before falling back, for EVERY request including ``/healthz``:
    measured 2 035-2 100 ms per call versus 3.3 ms to ``127.0.0.1``. That made a
    single retrieval take ~2.1 s of which ~2.05 s was a failed connect, and it
    dwarfed the embedding work (~23 ms).

    Only the literal host ``localhost`` is rewritten, and only for the URL we
    use to reach a loopback service; a remote or IPv6-only endpoint is untouched.
    """
    if not url:
        return url
    parts = urlsplit(url.strip())
    if parts.hostname != "localhost":
        return url.strip().rstrip("/")
    netloc = "127.0.0.1"
    if parts.port:
        netloc = f"127.0.0.1:{parts.port}"
    if parts.username or parts.password:
        # Credentials in the URL are not expected for Qdrant; do not rewrite a
        # URL we do not fully understand.
        return url.strip().rstrip("/")
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment)).rstrip("/")


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

    @property
    def uploads_dir(self) -> Path:
        """Root for user-uploaded originals (V4). Never sent externally."""
        return self.data_dir / "uploads"

    def kb_upload_dir(self, kb_id: str) -> Path:
        return self.uploads_dir / kb_id

    # Qdrant
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_api_key: str | None = None

    @field_validator("qdrant_url", mode="after")
    @classmethod
    def _normalize_qdrant_url(cls, value: str) -> str:
        resolved = normalize_loopback_url(value)
        if resolved != value.strip().rstrip("/"):
            logger.info(
                "QDRANT_URL host 'localhost' rewritten to 127.0.0.1 (IPv4) to avoid "
                "a fixed ~2 s IPv6 connect timeout on every request"
            )
        return resolved

    # LLM provider: "openai" | "anthropic" | "gemini" | "mock"
    llm_provider: str = ""
    llm_model: str = ""
    llm_api_key: str = ""
    llm_base_url: str = ""  # optional, for OpenAI-compatible local servers

    # V19: explicit call budget. The openai SDK otherwise applies its own
    # defaults (a 600 s timeout and 2 automatic retries), which can stall one
    # MCP tool call for half an hour when an endpoint hangs. Both values are
    # operator-visible configuration, not hidden client defaults.
    llm_timeout_s: float = Field(default=120.0, gt=0, le=3600)
    llm_max_retries: int = Field(default=2, ge=0, le=10)

    # Development mode: if true and no LLM is configured, a clearly-labelled
    # deterministic mock analyzer is used instead of failing hard.
    allow_mock_llm: bool = True

    # Embeddings: "sentence-transformers" | "openai"
    embedding_provider: str = "sentence-transformers"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_api_key: str = ""

    # Retrieval defaults
    default_top_k: int = 5

    # --- V20: grounding-gate relevance floor ---
    # Below this top-chunk similarity nothing is treated as topically related to
    # the question, so the answer abstains instead of grounding an off-domain
    # question on generic word overlap. Default 0.35 was measured on the protected
    # Automobile corpus with the default embedding model (see
    # app/services/answering/gate.py, MIN_RELEVANCE_TO_ANSWER for the numbers).
    # It is configuration, not a hidden constant, because a cosine scale is
    # embedding-model dependent: recalibrate when EMBEDDING_MODEL changes.
    answer_min_relevance: float = Field(default=0.35, ge=0.0, le=1.0)

    # --- V14: held-out validation artifacts ---
    # Overridable so a test (or an experiment on a copy of the corpus) can point
    # the held-out tooling at its own artifacts instead of the shipped ones. A
    # test that could only use the real files would either mutate the research
    # instrument or not exist at all.
    heldout_candidate_path: Path | None = None
    heldout_benchmark_path: Path | None = None

    # --- V22: remote MCP transport (Streamable HTTP) ---
    # The stdio server (V17) stays the local default. These settings only apply
    # when the remote transport is started explicitly:
    #     backend> .venv\\Scripts\\python.exe -m app.mcp.remote
    mcp_remote_host: str = "127.0.0.1"
    mcp_remote_port: int = 8010
    #: RFC 8707 resource indicator / token audience. Defaults to
    #: ``{issuer}/mcp``; set it explicitly once the server has a public URL.
    mcp_resource_server_url: str = ""
    #: Public origin of this MCP server (issuer for OAuth metadata). Defaults to
    #: ``http://127.0.0.1:{mcp_remote_port}``, which is development-only.
    mcp_issuer_url: str = ""
    #: Enable the embedded OAuth 2.1 authorization server (required for ChatGPT).
    mcp_allow_oauth: bool = False
    #: Consent page is auto-approved without a human. Local development only.
    mcp_oauth_auto_approve: bool = False
    #: Identity every OAuth-issued token carries. Never taken from the client.
    mcp_oauth_subject: str = "local"
    #: ``token:subject:scope1 scope2`` entries separated by ``;`` or newlines.
    #: For LOCAL clients only - ChatGPT cannot present a static API key.
    mcp_access_tokens: str = ""
    #: Comma-separated Host/Origin allow-lists. Empty = loopback only.
    mcp_allowed_hosts: str = ""
    mcp_allowed_origins: str = ""
    mcp_max_request_bytes: int = 1024 * 1024
    mcp_rate_limit_rpm: int = 120
    mcp_timeout_s: float = 60.0

    # --- V4: user document ingestion ---
    max_upload_bytes: int = 100 * 1024 * 1024  # 100 MB per uploaded file
    max_upload_files_per_request: int = 200

    # Privacy: user-provided files may be private course material. Parsing,
    # chunking and embedding are LOCAL by default. Set this to True only after
    # explicitly deciding that uploaded content may leave the machine.
    allow_external_llm_for_user_documents: bool = False

    model_config = {
        "env_file": str(Path(__file__).resolve().parent.parent / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache
def get_settings() -> Settings:
    return Settings()
