"""
embedding_service.py

Responsibility: provide the local HuggingFace sentence-transformers client
used for FAISS indexing and querying.
"""

from functools import lru_cache

from langchain_core.embeddings import Embeddings

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _resolve_embedding_device() -> str:
    requested = (settings.embedding_device or "cpu").strip().lower()
    if requested in {"", "cpu"}:
        return "cpu"

    if requested == "gpu":
        requested = "cuda"

    if requested == "auto":
        return "cuda" if _cuda_is_available() else "cpu"

    if requested == "cuda" or requested.startswith("cuda:"):
        if _cuda_is_available():
            return requested
        logger.warning(
            "EMBEDDING_DEVICE='%s' requested but CUDA is unavailable to PyTorch. "
            "Falling back to CPU.",
            requested,
        )
        return "cpu"

    logger.warning(
        "Unsupported EMBEDDING_DEVICE='%s'. Supported values are cpu, auto, cuda, "
        "or cuda:<index>. Falling back to CPU.",
        requested,
    )
    return "cpu"


def _cuda_is_available() -> bool:
    try:
        import torch
    except ImportError:
        logger.warning(
            "EMBEDDING_DEVICE requests CUDA, but PyTorch is not installed. "
            "Falling back to CPU."
        )
        return False

    return bool(torch.cuda.is_available())


class LocalHuggingFaceEmbeddings(Embeddings):
    """
    Wrap HuggingFaceEmbeddings with a stable local model.

    The class preserves the Embeddings interface expected by FAISS and
    the rest of the RAG pipeline.
    """

    def __init__(self, model_name: str, batch_size: int = 32, device: str = "cpu"):
        self.model_name = model_name
        self.batch_size = max(1, batch_size)
        self.device = device

        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as exc:
            raise RuntimeError(
                "The HuggingFace embedding backend requires 'langchain-huggingface'. "
                "Install backend/requirements.txt before starting the backend."
            ) from exc

        self._client = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": self.device},
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
            ) from exc

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def is_local_embedding_backend(embeddings: Embeddings) -> bool:
    """Return True when the active embeddings instance runs in-process."""
    return isinstance(embeddings, LocalHuggingFaceEmbeddings)


def embedding_backend_name(embeddings: Embeddings) -> str:
    """Best-effort human-readable backend name for diagnostics."""
    if isinstance(embeddings, LocalHuggingFaceEmbeddings):
        return f"local_sentence_transformers[{getattr(embeddings, 'device', 'unknown')}]"
    return type(embeddings).__name__


def resolve_embedding_model_name() -> str:
    """Return the sentence-transformers model this runtime can actually load."""
    configured_model = (settings.embedding_model or "").strip()
    if configured_model.startswith("sentence-transformers/"):
        return configured_model
    return DEFAULT_LOCAL_EMBEDDING_MODEL


def get_embedding_model() -> Embeddings:
    return _get_cached_embedding_model()


@lru_cache(maxsize=1)
def _get_cached_embedding_model() -> Embeddings:
    configured_model = (settings.embedding_model or "").strip()
    model_name = resolve_embedding_model_name()

    if model_name != configured_model:
        logger.info(
            "Configured embedding model '%s' is not a local sentence-transformers model. "
            "Using '%s' instead.",
            configured_model,
            model_name,
        )

    logger.info("Initialising local Hugging Face embeddings with model: %s", model_name)
    device = _resolve_embedding_device()
    logger.info("Embedding device resolved to: %s", device)
    return LocalHuggingFaceEmbeddings(
        model_name=model_name,
        batch_size=settings.embedding_batch_size,
        device=device,
    )
