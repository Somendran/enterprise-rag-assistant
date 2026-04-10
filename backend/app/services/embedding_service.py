"""
embedding_service.py
────────────────────
Responsibility: Provide a local Hugging Face embeddings client.

This implementation uses sentence-transformers/all-MiniLM-L6-v2 locally,
so embedding generation does not rely on external API calls.
"""

from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class LocalHuggingFaceEmbeddings(Embeddings):
    """
    Wrap HuggingFaceEmbeddings with a stable local model.

    The class preserves the Embeddings interface expected by FAISS and
    the rest of the RAG pipeline.
    """

    def __init__(self, model_name: str, batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = max(1, batch_size)
        self._client = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={
                "batch_size": self.batch_size,
                "normalize_embeddings": True,
            },
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        try:
            vectors = self._client.embed_documents(texts)
            if len(vectors) != len(texts):
                raise RuntimeError(
                    f"Embedding size mismatch: expected {len(texts)}, got {len(vectors)}"
                )
            return vectors
        except Exception as exc:
            logger.error("Local embedding generation failed: %s", exc)
            raise RuntimeError(
                f"Embedding generation failed for model '{self.model_name}': {exc}"
            )

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def is_local_embedding_backend(embeddings: Embeddings) -> bool:
    """Return True when the active embeddings instance runs locally."""
    return isinstance(embeddings, LocalHuggingFaceEmbeddings)


def embedding_backend_name(embeddings: Embeddings) -> str:
    """Best-effort human-readable backend name for diagnostics."""
    if is_local_embedding_backend(embeddings):
        return "local_sentence_transformers"
    return type(embeddings).__name__

def get_embedding_model() -> Embeddings:
    return _get_cached_embedding_model()


@lru_cache(maxsize=1)
def _get_cached_embedding_model() -> Embeddings:
    configured_model = (settings.embedding_model or "").strip()
    model_name = (
        configured_model
        if configured_model.startswith("sentence-transformers/")
        else DEFAULT_LOCAL_EMBEDDING_MODEL
    )

    if model_name != configured_model:
        logger.info(
            "Configured embedding model '%s' is not a local sentence-transformers model. "
            "Using '%s' instead.",
            configured_model,
            model_name,
        )

    logger.info(
        "Initialising local Hugging Face embeddings with model: %s",
        model_name,
    )
    return LocalHuggingFaceEmbeddings(
        model_name=model_name,
        batch_size=settings.embedding_batch_size,
    )
