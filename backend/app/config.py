"""Centralized configuration using pydantic-settings."""

from functools import lru_cache
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "development"

    # API / browser access controls. Leave app_api_key empty for local-only dev;
    # set it in production to require X-API-Key or Authorization: Bearer <key>.
    app_api_key: str = ""
    allowed_cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    enable_user_auth: bool = True
    auth_token_ttl_hours: int = 24
    public_demo_mode: bool = False
    demo_max_upload_mb: int = 5
    demo_max_pages: int = 20
    demo_max_files_per_request: int = 2
    demo_max_docs_per_session: int = 3
    demo_uploads_per_hour: int = 3
    demo_queries_per_hour: int = 30
    demo_uploads_per_hour_ip: int = 10
    demo_queries_per_hour_ip: int = 120
    demo_doc_ttl_hours: int = 24

    # Model identifiers
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    embedding_device: str = "cpu"
    # Local Ollama generation defaults.
    local_llm_endpoint: str = "http://localhost:11434/api/generate"
    local_llm_model: str = "gemma4:e2b"
    # Validate model availability via Ollama /api/tags before generation.
    local_llm_validate_model: bool = True
    local_llm_temperature: float = 0.25
    # Primary generation budget exposed for tuning in .env.
    llm_max_tokens: int = 256
    local_llm_num_predict: int = 256
    # Second-attempt budget used when Ollama stops with done_reason=length.
    local_llm_retry_num_predict: int = 512
    # Hard cap for local generation calls per request path.
    local_llm_max_attempts: int = 2
    # Ollama request controls.
    local_llm_stream: bool = True
    local_llm_connect_timeout_seconds: int = 10
    local_llm_read_timeout_seconds: int = 150
    # -1 lets Ollama auto-detect and use available GPU resources.
    local_llm_num_gpu: int = -1

    # OpenAI generation.
    use_openai: bool = True
    openai_api_key: str = ""
    openai_model: str = "gpt-5.4-mini"
    openai_max_tokens: int = 320
    openai_temperature: float = 0.2
    openai_timeout_seconds: int = 30
    openai_network_retry_attempts: int = 1
    openai_fallback_to_local: bool = False

    # Storage paths
    # Where uploaded PDFs are saved on disk
    upload_dir: str = "data/uploads"
    max_upload_size_mb: int = 25

    # Where the FAISS index is persisted between restarts
    faiss_index_path: str = "data/faiss_index"
    # SQLite metadata store for document registry, feedback, and admin data.
    metadata_db_path: str = "data/app_metadata.sqlite3"

    # Retrieval
    # Final number of chunks passed to the LLM after reranking.
    retrieval_top_n: int = 6
    # Wider profile for summary/synthesis questions.
    summary_retrieval_top_n: int = 8
    summary_max_context_characters: int = 9000
    complex_query_rerank_always: bool = True
    # How many candidates to gather before neural reranking.
    retrieval_initial_top_k: int = 40
    # Backward-compat alias used in some call paths.
    retrieval_top_k: int = 6
    # Candidate chunks to collect before rerank-lite trims to retrieval_top_k.
    retrieval_candidate_k: int = 10
    bm25_top_k: int = 30
    bm25_weight: float = 0.25
    # Adaptive fast mode for simple questions.
    fast_mode_enabled: bool = True
    simple_query_short_token_limit: int = 8
    simple_query_direct_token_limit: int = 12
    fast_mode_initial_top_k: int = 18
    fast_mode_top_n: int = 3
    fast_mode_max_context_characters: int = 1200
    fast_mode_llm_max_tokens: int = 224
    fast_mode_trim_prompt: bool = True
    # If confidence is below this threshold, trigger one deterministic fallback pass.
    retrieval_low_confidence_threshold: float = 0.30
    # Floor threshold below which the API should refuse to answer confidently.
    answer_low_confidence_threshold: float = 0.22
    answer_hard_refusal_threshold: float = 0.12
    low_confidence_min_chunk_score: float = 0.18
    # Keep context compact for local generation latency.
    max_context_characters: int = 6000
    # Structured ingestion + vision enrichment toggles.
    enable_docling: bool = True
    enable_ocr: bool = False
    docling_batch_size: int = 10
    ingestion_low_text_page_chars: int = 80
    ingestion_ocr_page_ratio_threshold: float = 0.60
    enable_vision_enrichment: bool = True
    max_vision_calls_per_doc: int = 10
    openai_vision_model: str = "gpt-4.1-mini"
    # Ollama chat endpoint for multimodal models when OpenAI is disabled.
    local_vision_endpoint: str = "http://localhost:11434/api/chat"
    local_vision_model: str = ""
    # Per-chunk metadata enrichment controls.
    enable_metadata_enrichment: bool = True
    enable_summary: bool = True
    summary_max_tokens: int = 20
    summary_max_chunks: int = 30
    summary_min_chars: int = 300
    # Lightweight post-generation verification controls.
    enable_verification: bool = True
    verification_similarity_threshold: float = 0.75
    verification_min_answer_chars: int = 80
    verification_warning_support_threshold: float = 0.50

    # Embedding ingestion throughput
    # Number of chunks per embedding call during upload ingestion.
    # Tune batch size for the selected sentence-transformers device.
    embedding_batch_size: int = 64
    # Optional embedding parallelism. Keep at 1 unless backend is confirmed
    # thread-safe for concurrent embed_documents calls.
    embedding_parallel_workers: int = 1

    # Text splitting
    chunk_size: int = 700
    chunk_overlap: int = 100

    # Retrieval behavior toggles
    enable_retrieval_fallback: bool = True
    enable_retrieval_diagnostics: bool = True
    enable_neural_reranker: bool = True
    reranker_model_name: str = "BAAI/bge-reranker-base"
    reranker_batch_size: int = 16
    reranker_use_fp16: bool = False
    # Limit docs passed to cross-encoder and optionally skip when top score dominates.
    rerank_top_k: int = 10
    reranker_min_candidates: int = 6
    reranker_skip_if_score_gap: bool = True
    reranker_score_gap_threshold: float = 0.18
    reranker_skip_if_high_confidence: bool = True
    reranker_high_confidence_threshold: float = 0.80

    # Adaptive context sizing from score distribution.
    context_dominant_gap_threshold: float = 0.20
    vector_weight: float = 0.60
    lexical_weight: float = 0.15

    # Query caching
    enable_query_cache: bool = True
    query_cache_ttl_seconds: int = 600

    @property
    def cors_origins(self) -> list[str]:
        origins = [
            item.strip()
            for item in self.allowed_cors_origins.split(",")
            if item.strip()
        ]
        return origins or ["http://localhost:5173"]

    @property
    def max_upload_size_bytes(self) -> int:
        return max(1, int(self.max_upload_size_mb)) * 1024 * 1024

    @model_validator(mode="after")
    def require_api_key_in_production(self) -> "Settings":
        if self.app_env.strip().lower() == "production" and not self.app_api_key.strip():
            raise ValueError("APP_API_KEY is required when APP_ENV=production.")
        return self

    # Tells pydantic-settings to load a .env file automatically
    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parents[1] / ".env"),
        env_file_encoding="utf-8",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Returns a cached Settings instance.
    Using lru_cache ensures the .env file is only parsed once,
    which is important for performance in async FastAPI apps.
    """
    return Settings()


# Convenience alias: just `from app.config import settings` anywhere.
settings = get_settings()
