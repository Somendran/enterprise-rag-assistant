"""
config.py
─────────
Centralised configuration using pydantic-settings.
All values are read from environment variables (or .env file).
Importing `settings` anywhere in the app gives you the same singleton.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    # ── Google Gemini ────────────────────────────────────────────────────────
    google_api_key: str

    # Model identifiers
    embedding_model: str = "models/gemini-embedding-001"
    llm_model: str = "gemini-2.5-flash"

    # ── Storage paths ────────────────────────────────────────────────────────
    # Where uploaded PDFs are saved on disk
    upload_dir: str = "data/uploads"

    # Where the FAISS index is persisted between restarts
    faiss_index_path: str = "data/faiss_index"

    # ── Retrieval ────────────────────────────────────────────────────────────
    # How many chunks to fetch per query
    retrieval_top_k: int = 3
    # Candidate chunks to collect before rerank-lite trims to retrieval_top_k.
    retrieval_candidate_k: int = 10
    # If confidence is below this threshold, trigger one deterministic fallback pass.
    retrieval_low_confidence_threshold: float = 0.30
    # Floor threshold below which the API should refuse to answer confidently.
    answer_low_confidence_threshold: float = 0.22

    # ── Embedding ingestion throughput ───────────────────────────────────────
    # Number of chunks per embedding call during upload ingestion.
    embedding_batch_size: int = 16
    # Conservative cap to stay below free-tier request limits.
    embedding_requests_per_minute: int = 90

    # ── Text splitting ───────────────────────────────────────────────────────
    chunk_size: int = 700
    chunk_overlap: int = 100

    # ── Retrieval behavior toggles ───────────────────────────────────────────
    enable_retrieval_fallback: bool = True
    enable_retrieval_diagnostics: bool = True

    # Tells pydantic-settings to load a .env file automatically
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Returns a cached Settings instance.
    Using lru_cache ensures the .env file is only parsed once,
    which is important for performance in async FastAPI apps.
    """
    return Settings()


# Convenience alias — just `from app.config import settings` anywhere
settings = get_settings()
