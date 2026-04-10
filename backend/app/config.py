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
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "gemini-2.5-flash"
    # Keep Gemini fallback disabled by default for strict local operation.
    enable_gemini_fallback: bool = False
    # App-side limiter for generation calls to reduce Gemini 429s.
    llm_requests_per_minute: int = 30
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

    # ── OpenAI generation (optional primary path) ───────────────────────────
    use_openai: bool = False
    openai_api_key: str = ""
    openai_model: str = "gpt-4.1-mini"
    openai_max_tokens: int = 300
    openai_temperature: float = 0.2
    openai_timeout_seconds: int = 30
    openai_network_retry_attempts: int = 1

    # ── Storage paths ────────────────────────────────────────────────────────
    # Where uploaded PDFs are saved on disk
    upload_dir: str = "data/uploads"

    # Where the FAISS index is persisted between restarts
    faiss_index_path: str = "data/faiss_index"

    # ── Retrieval ────────────────────────────────────────────────────────────
    # Final number of chunks passed to the LLM after reranking.
    retrieval_top_n: int = 5
    # How many candidates to gather before neural reranking.
    retrieval_initial_top_k: int = 30
    # Backward-compat alias used in some call paths.
    retrieval_top_k: int = 5
    # Candidate chunks to collect before rerank-lite trims to retrieval_top_k.
    retrieval_candidate_k: int = 10
    bm25_top_k: int = 20
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
    max_context_characters: int = 2200
    # Structured ingestion + vision enrichment toggles.
    enable_docling: bool = True
    enable_ocr: bool = False
    docling_batch_size: int = 10
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
