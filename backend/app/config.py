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
    # Keep Gemini fallback disabled by default for strict local operation.
    enable_gemini_fallback: bool = False
    # App-side limiter for generation calls to reduce Gemini 429s.
    llm_requests_per_minute: int = 30
    # Local Ollama generation defaults.
    local_llm_endpoint: str = "http://localhost:11434/api/generate"
    local_llm_model: str = "gemma4:e4b"
    # Validate model availability via Ollama /api/tags before generation.
    local_llm_validate_model: bool = True
    local_llm_temperature: float = 0.25
    local_llm_num_predict: int = 256
    # Second-attempt budget used when Ollama stops with done_reason=length.
    local_llm_retry_num_predict: int = 512
    # Ollama request controls.
    local_llm_stream: bool = True
    local_llm_connect_timeout_seconds: int = 10
    local_llm_read_timeout_seconds: int = 150
    # -1 lets Ollama auto-detect and use available GPU resources.
    local_llm_num_gpu: int = -1

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
    # Keep context compact for local generation latency.
    max_context_characters: int = 2200

    # ── Embedding ingestion throughput ───────────────────────────────────────
    # Number of chunks per embedding call during upload ingestion.
    # Tuned for local sentence-transformers throughput on CPU.
    embedding_batch_size: int = 64
    # Optional embedding parallelism. Keep at 1 unless backend is confirmed
    # thread-safe for concurrent embed_documents calls.
    embedding_parallel_workers: int = 1

    # ── Text splitting ───────────────────────────────────────────────────────
    chunk_size: int = 700
    chunk_overlap: int = 100

    # ── Retrieval behavior toggles ───────────────────────────────────────────
    enable_retrieval_fallback: bool = True
    enable_retrieval_diagnostics: bool = True

    # ── Query caching ─────────────────────────────────────────────────────────
    enable_query_cache: bool = True
    query_cache_ttl_seconds: int = 600

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
