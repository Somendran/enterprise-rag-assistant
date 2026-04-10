"""
vector_store.py
───────────────
Responsibility: Manage a FAISS vector store — add documents and persist the index.

Architecture note (pgvector upgrade path):
───────────────────────────────────────────
The public interface of this module is intentionally minimal:
    - get_or_create_store()   → returns a VectorStore
    - add_documents()         → embeds + stores chunks
    - load_store()            → reloads a persisted index

To switch to pgvector, you only need to replace the internals of these
three functions with PGVector calls. None of the callers (upload.py,
retriever.py) will need to change.

Thread-safety note:
───────────────────
FAISS is not thread-safe for concurrent writes.  For a production deployment
with multiple workers, use pgvector instead (it is Postgres-backed and safe).
For single-worker / development use, FAISS is fine.
"""

import time
import shutil
import json
import hashlib
import re
import threading
import math
from pathlib import Path
from typing import List, Optional, Any
from concurrent.futures import ThreadPoolExecutor

from langchain_core.embeddings import Embeddings
from langchain.schema import Document
from langchain_community.vectorstores import FAISS

from app.services.embedding_service import (
    get_embedding_model,
    is_local_embedding_backend,
    embedding_backend_name,
)
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Module-level reference so the store is only loaded from disk once per
# process.  This acts as a simple in-process cache.
_store: Optional[FAISS] = None
_META_FILENAME = "index_meta.json"
_DOC_REGISTRY_FILENAME = "document_registry.json"
_write_lock = threading.Lock()


def _get_index_dimension(store: FAISS) -> Optional[int]:
    """Best-effort retrieval of the underlying FAISS vector dimension."""
    idx = getattr(store, "index", None)
    if idx is None:
        return None

    # Common FAISS index wrappers expose dimension differently.
    for attr in ("d", "d_out"):
        value = getattr(idx, attr, None)
        if isinstance(value, int):
            return int(value)

    # PreTransform / wrapper indexes may expose an inner index.
    inner = getattr(idx, "index", None)
    if inner is not None:
        for attr in ("d", "d_out"):
            value = getattr(inner, attr, None)
            if isinstance(value, int):
                return int(value)

    try:
        return int(idx.d)
    except Exception:
        return None


def _index_path() -> str:
    """
    Return a model-scoped absolute path for FAISS index persistence.

    Different embedding models produce different vector dimensions. By storing
    each index in its own model-specific directory, we avoid cross-model merges
    and eliminate a whole class of dimension mismatch failures.
    """
    base_path = Path(settings.faiss_index_path).resolve()
    model_name = settings.embedding_model.strip().lower()

    # Keep directory names readable and filesystem-safe, plus a short hash.
    safe_name = re.sub(r"[^a-z0-9._-]+", "-", model_name).strip("-")[:48] or "unknown-model"
    model_hash = hashlib.sha1(model_name.encode("utf-8")).hexdigest()[:10]
    scoped_path = base_path / f"{safe_name}-{model_hash}"
    return str(scoped_path)


def _meta_path(index_path: str) -> Path:
    """Return metadata file path stored alongside the FAISS index."""
    return Path(index_path) / _META_FILENAME


def _doc_registry_path(index_path: str) -> Path:
    """Return document registry path used for duplicate detection."""
    return Path(index_path) / _DOC_REGISTRY_FILENAME


def _read_index_meta(index_path: str) -> Optional[dict[str, Any]]:
    """Read persisted index metadata if present."""
    meta_file = _meta_path(index_path)
    if not meta_file.exists():
        return None

    try:
        return json.loads(meta_file.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to read FAISS index metadata: %s", exc)
        return None


def _read_doc_registry(index_path: str) -> dict[str, Any]:
    """Read persisted document registry if available."""
    registry_file = _doc_registry_path(index_path)
    if not registry_file.exists():
        return {}
    try:
        payload = json.loads(registry_file.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    except Exception as exc:
        logger.warning("Failed to read document registry: %s", exc)
    return {}


def _write_doc_registry(index_path: str, registry: dict[str, Any]) -> None:
    """Persist document registry for duplicate checks across restarts."""
    registry_file = _doc_registry_path(index_path)
    registry_file.parent.mkdir(parents=True, exist_ok=True)
    registry_file.write_text(json.dumps(registry, indent=2), encoding="utf-8")


def _write_index_meta(index_path: str, dimension: Optional[int]) -> None:
    """Persist minimal index metadata for compatibility checks on reload."""
    meta_file = _meta_path(index_path)
    meta_file.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "embedding_model": settings.embedding_model,
        "dimension": dimension,
    }
    meta_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _resolve_current_embedding_dimension(index_path: str, embeddings: Embeddings) -> Optional[int]:
    """
    Resolve current embedding dimension.

    Prefer metadata value for the active model to avoid unnecessary embedding calls.
    Fall back to a single probe embedding when metadata is missing/stale.
    """
    meta = _read_index_meta(index_path)
    if meta and meta.get("embedding_model") == settings.embedding_model:
        dim = meta.get("dimension")
        if isinstance(dim, int) and dim > 0:
            return dim

    try:
        probe = embeddings.embed_query("dimension probe")
        return len(probe)
    except Exception as exc:
        logger.warning("Could not resolve current embedding dimension: %s", exc)
        return None


def _reset_persisted_index(index_path: str) -> None:
    """Delete any persisted FAISS data so a new index can be created safely."""
    path = Path(index_path)
    if not path.exists():
        return

    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


def is_document_indexed(file_hash: str) -> bool:
    """Return True when the same file hash has already been ingested."""
    index_path = _index_path()
    registry = _read_doc_registry(index_path)
    return file_hash in registry


def register_indexed_document(
    file_hash: str,
    filename: str,
    chunk_count: int,
    document_id: str,
) -> None:
    """Persist hash and document metadata for duplicate prevention."""
    index_path = _index_path()
    registry = _read_doc_registry(index_path)
    registry[file_hash] = {
        "filename": filename,
        "chunk_count": chunk_count,
        "document_id": document_id,
        "embedding_model": settings.embedding_model,
        "indexed_at": int(time.time()),
    }
    _write_doc_registry(index_path, registry)


def list_indexed_documents() -> list[dict[str, Any]]:
    """Return persisted indexed document metadata for UI display."""
    index_path = _index_path()
    registry = _read_doc_registry(index_path)

    items: list[dict[str, Any]] = []
    for file_hash, meta in registry.items():
        if not isinstance(meta, dict):
            continue
        items.append(
            {
                "file_hash": file_hash,
                "filename": str(meta.get("filename", "unknown")),
                "chunk_count": int(meta.get("chunk_count", 0) or 0),
                "indexed_at": int(meta.get("indexed_at", 0) or 0),
            }
        )

    items.sort(key=lambda x: (x["filename"].lower(), -x["indexed_at"]))
    return items


def get_or_create_store() -> FAISS:
    """
    Return the active FAISS vector store.

    - If an index already exists on disk, load and return it.
    - Otherwise return a fresh (empty) placeholder that will be populated
      the first time add_documents() is called.

    The store is cached in the module-level `_store` variable so subsequent
    calls within the same process don't re-read from disk.
    """
    global _store

    if _store is not None:
        return _store

    index_path = _index_path()
    embeddings = get_embedding_model()

    if Path(index_path).exists():
        logger.info(f"Loading existing FAISS index from: {index_path}")
        _store = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True,  # safe for local use
        )

        existing_dim = _get_index_dimension(_store)
        current_dim = _resolve_current_embedding_dimension(index_path, embeddings)

        if (
            existing_dim is not None
            and current_dim is not None
            and existing_dim != current_dim
        ):
            logger.warning("FAISS index dimension mismatch detected. Resetting index.")
            _store = None
            _reset_persisted_index(index_path)
    else:
        logger.info("No existing FAISS index found. A new one will be created on first upload.")
        _store = None  # remains None until the first add_documents() call

    return _store


def add_documents(chunks: List[Document]) -> FAISS:
    """
    Embed and store a list of Document chunks in the FAISS index.

    If no index exists yet, one is created from scratch.
    If an index already exists, the new chunks are merged into it.
    The updated index is persisted to disk after every call.

    Args:
        chunks: Text chunks with metadata to index.

    Returns:
        The updated FAISS store.
    """
    global _store

    with _write_lock:
        add_documents_start = time.perf_counter()
        embeddings = get_embedding_model()
        index_path = _index_path()
        batch_size = max(1, settings.embedding_batch_size)
        parallel_workers = max(1, settings.embedding_parallel_workers)
        backend_name = embedding_backend_name(embeddings)
        is_local_backend = is_local_embedding_backend(embeddings)

        # Ensure the parent directory exists
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)

        texts = [doc.page_content for doc in chunks]
        metadatas = [doc.metadata for doc in chunks]
        total_chunks = len(texts)
        total_batches = math.ceil(total_chunks / batch_size) if total_chunks else 0

        logger.info(
            "Embedding start | backend=%s total_chunks=%d batch_size=%d total_batches=%d parallel_workers=%d",
            backend_name,
            total_chunks,
            batch_size,
            total_batches,
            parallel_workers,
        )

        vectors: list[list[float]] = []
        batch_inputs = [texts[start : start + batch_size] for start in range(0, total_chunks, batch_size)]
        embedding_start = time.perf_counter()

        # Safe default: keep deterministic sequential execution for local backends.
        # Optional parallel mode is allowed only for non-local backends to avoid
        # thread-safety issues in local model runtimes.
        if parallel_workers > 1 and not is_local_backend:
            ordered_vectors: list[Optional[list[list[float]]]] = [None] * len(batch_inputs)

            def _embed_batch(i: int, batch_texts: list[str]) -> tuple[int, list[list[float]], float]:
                t0 = time.perf_counter()
                result = embeddings.embed_documents(batch_texts)
                elapsed_s = time.perf_counter() - t0
                return i, result, elapsed_s

            with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
                futures = [
                    executor.submit(_embed_batch, i, batch_texts)
                    for i, batch_texts in enumerate(batch_inputs)
                ]
                for future in futures:
                    i, result, elapsed_s = future.result()
                    ordered_vectors[i] = result
                    logger.info(
                        "Embedding batch %d/%d completed in %.3fs (items=%d)",
                        i + 1,
                        total_batches,
                        elapsed_s,
                        len(batch_inputs[i]),
                    )

            for i, maybe_vectors in enumerate(ordered_vectors):
                if maybe_vectors is None:
                    raise RuntimeError(f"Missing embedding results for batch {i + 1}/{total_batches}")
                vectors.extend(maybe_vectors)
        else:
            if parallel_workers > 1 and is_local_backend:
                logger.info(
                    "Parallel embedding requested (%d workers) but backend=%s is kept sequential for safety.",
                    parallel_workers,
                    backend_name,
                )

            for i, batch_texts in enumerate(batch_inputs, start=1):
                batch_start = time.perf_counter()
                batch_vectors = embeddings.embed_documents(batch_texts)
                vectors.extend(batch_vectors)
                batch_elapsed = time.perf_counter() - batch_start
                logger.info(
                    "Embedding batch %d/%d completed in %.3fs (items=%d)",
                    i,
                    total_batches,
                    batch_elapsed,
                    len(batch_texts),
                )

        embedding_elapsed = time.perf_counter() - embedding_start
        logger.info(
            "Embedding complete | backend=%s total_chunks=%d total_batches=%d total_time_s=%.3f avg_batch_time_s=%.3f",
            backend_name,
            total_chunks,
            total_batches,
            embedding_elapsed,
            (embedding_elapsed / total_batches) if total_batches else 0.0,
        )

        faiss_start = time.perf_counter()
        text_embeddings = list(zip(texts, vectors))
        new_dim = len(vectors[0]) if vectors else None

        if _store is None:
            logger.info(f"Creating new FAISS index with {len(chunks)} chunk(s).")
            _store = FAISS.from_embeddings(text_embeddings, embeddings, metadatas=metadatas)
        else:
            existing_dim = _get_index_dimension(_store)
            logger.info("FAISS merge pre-check dimensions | existing=%s new=%s", existing_dim, new_dim)

            if existing_dim is not None and new_dim is not None and existing_dim != new_dim:
                logger.warning("FAISS index dimension mismatch detected. Resetting index.")
                _store = None
                _reset_persisted_index(index_path)
                _store = FAISS.from_embeddings(text_embeddings, embeddings, metadatas=metadatas)
            else:
                logger.info(f"Merging {len(chunks)} new chunk(s) into existing FAISS index.")
                new_store = FAISS.from_embeddings(text_embeddings, embeddings, metadatas=metadatas)
                try:
                    _store.merge_from(new_store)
                except Exception as exc:
                    # Final safeguard: if FAISS still reports incompatible indexes,
                    # reset and rebuild with the active embedding dimension.
                    if "other->d == d" in str(exc) or "check_compatible_for_merge" in str(exc):
                        logger.warning(
                            "FAISS index dimension mismatch detected during merge. "
                            "Resetting index. Error: %s",
                            exc,
                        )
                        _store = None
                        _reset_persisted_index(index_path)
                        _store = FAISS.from_embeddings(text_embeddings, embeddings, metadatas=metadatas)
                    else:
                        raise

        # Persist after every successful write
        _store.save_local(index_path)
        _write_index_meta(index_path, _get_index_dimension(_store))
        faiss_elapsed = time.perf_counter() - faiss_start
        total_elapsed = time.perf_counter() - add_documents_start
        logger.info(
            "FAISS index saved | path=%s faiss_time_s=%.3f total_add_documents_time_s=%.3f",
            index_path,
            faiss_elapsed,
            total_elapsed,
        )

        return _store


def load_store() -> Optional[FAISS]:
    """
    Explicitly reload the store from disk (e.g. after a restart).
    Returns None if no index has been saved yet.
    """
    global _store
    _store = None          # clear cache to force a fresh load
    return get_or_create_store()


def get_knowledge_base_version() -> str:
    """
    Return a lightweight version string for cache invalidation.

    The value changes whenever persisted index metadata or registry file mtime changes.
    """
    index_path = Path(_index_path())
    if not index_path.exists():
        return "empty"

    meta_mtime = _meta_path(str(index_path)).stat().st_mtime if _meta_path(str(index_path)).exists() else 0
    registry_mtime = _doc_registry_path(str(index_path)).stat().st_mtime if _doc_registry_path(str(index_path)).exists() else 0
    index_mtime = index_path.stat().st_mtime
    return f"{settings.embedding_model}:{int(max(meta_mtime, registry_mtime, index_mtime))}"


def reset_vector_store() -> bool:
    """
    Reset active vector store by clearing in-memory cache and deleting persisted index.

    Returns:
        True if persisted index path existed and was removed, False otherwise.
    """
    global _store

    with _write_lock:
        index_path = _index_path()
        existed = Path(index_path).exists()
        _store = None
        _reset_persisted_index(index_path)
        return existed
