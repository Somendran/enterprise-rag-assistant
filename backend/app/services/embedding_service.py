"""Embedding backends for FAISS indexing and querying."""

from functools import lru_cache
import hashlib
import math
import re

from langchain_core.embeddings import Embeddings

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _torch_cuda_available() -> bool:
    try:
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _resolve_embedding_device() -> str:
    requested = (settings.embedding_device or "cpu").strip().lower()
    if requested not in {"auto", "cpu", "cuda"}:
        logger.warning("Unsupported EMBEDDING_DEVICE='%s'. Falling back to CPU.", requested)
        requested = "cpu"

    if requested == "cpu":
        return "cpu"

    cuda_available = _torch_cuda_available()
    if requested == "cuda":
        if cuda_available:
            return "cuda"
        logger.warning("EMBEDDING_DEVICE=cuda requested but CUDA is unavailable. Falling back to CPU.")
        return "cpu"

    return "cuda" if cuda_available else "cpu"


class LocalHuggingFaceEmbeddings(Embeddings):
    """Wrap HuggingFaceEmbeddings with the Embeddings interface."""

    def __init__(self, model_name: str, batch_size: int = 32, device: str = "cpu"):
        self.model_name = model_name
        self.batch_size = max(1, batch_size)
        self.device = device

        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as exc:
            raise RuntimeError(
                "The HuggingFace embedding backend requires 'langchain-huggingface'. "
                "Use EMBEDDING_BACKEND=hash for the lightweight Docker image."
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
            )

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class HashEmbeddings(Embeddings):
    """Deterministic lightweight embeddings with no ML runtime dependency.

    This backend is intended for Docker demos and CI where small images matter
    more than semantic embedding quality. Use local_hf for better retrieval.
    """

    def __init__(self, dimensions: int = 384):
        self.dimensions = max(64, int(dimensions))

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"[a-zA-Z0-9]+", (text or "").lower())
        if not tokens:
            tokens = [(text or "empty").strip().lower()]

        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm <= 0:
            return vector
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)


def is_local_embedding_backend(embeddings: Embeddings) -> bool:
    """Return True when the active embeddings instance runs in-process."""
    return isinstance(embeddings, (LocalHuggingFaceEmbeddings, HashEmbeddings))


def embedding_backend_name(embeddings: Embeddings) -> str:
    """Best-effort human-readable backend name for diagnostics."""
    if isinstance(embeddings, HashEmbeddings):
        return f"hash_embeddings[{embeddings.dimensions}]"
    if isinstance(embeddings, LocalHuggingFaceEmbeddings):
        return f"local_sentence_transformers[{getattr(embeddings, 'device', 'unknown')}]"
    return type(embeddings).__name__


def get_embedding_model() -> Embeddings:
    return _get_cached_embedding_model()


@lru_cache(maxsize=1)
def _get_cached_embedding_model() -> Embeddings:
    backend = (settings.embedding_backend or "local_hf").strip().lower()
    if backend in {"hash", "hash_embeddings", "lightweight"}:
        dimensions = max(64, int(settings.hash_embedding_dimensions))
        logger.info("Initialising lightweight hash embeddings with dimensions=%d", dimensions)
        return HashEmbeddings(dimensions=dimensions)

    if backend not in {"local_hf", "huggingface", "sentence_transformers"}:
        logger.warning("Unsupported EMBEDDING_BACKEND='%s'. Falling back to local_hf.", backend)

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

    logger.info("Initialising local Hugging Face embeddings with model: %s", model_name)
    device = _resolve_embedding_device()
    logger.info("Embedding device resolved to: %s", device)
    return LocalHuggingFaceEmbeddings(
        model_name=model_name,
        batch_size=settings.embedding_batch_size,
        device=device,
    )
