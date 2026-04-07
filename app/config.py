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
    embedding_model: str = "models/gemini-embedding-004"
    llm_model: str = "gemini-2.0-flash"

    # ── Storage paths ────────────────────────────────────────────────────────
    # Where uploaded PDFs are saved on disk
    upload_dir: str = "data/uploads"

    # Where the FAISS index is persisted between restarts
    faiss_index_path: str = "data/faiss_index"

    # ── Retrieval ────────────────────────────────────────────────────────────
    # How many chunks to fetch per query
    retrieval_top_k: int = 5

    # ── Text splitting ───────────────────────────────────────────────────────
    chunk_size: int = 700
    chunk_overlap: int = 100

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
