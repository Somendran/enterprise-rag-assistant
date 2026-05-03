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
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Optional, Any

import faiss
from langchain_core.embeddings import Embeddings
from langchain.schema import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

from app.services.embedding_service import (
    get_embedding_model,
    embedding_backend_name,
    resolve_embedding_model_name,
)
from app.services import metadata_store
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Module-level reference so the store is only loaded from disk once per
# process.  This acts as a simple in-process cache.
_store: Optional[FAISS] = None
_META_FILENAME = "index_meta.json"
_DOC_REGISTRY_FILENAME = "document_registry.json"
_FAISS_INDEX_FILENAME = "index.index"
_write_lock = threading.Lock()
_migration_checked = False


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
    model_name = resolve_embedding_model_name().strip().lower()

    # Keep directory names readable and filesystem-safe, plus a short hash.
    safe_name = re.sub(r"[^a-z0-9._-]+", "-", model_name).strip("-")[:48] or "unknown-model"
    model_hash = hashlib.sha1(model_name.encode("utf-8")).hexdigest()[:10]
    scoped_path = base_path / f"{safe_name}-{model_hash}"
    return str(scoped_path)


def _configured_embedding_is_supported() -> bool:
    configured = (settings.embedding_model or "").strip()
    return not configured or configured.startswith("sentence-transformers/")


def _find_unsupported_configured_index() -> Optional[Path]:
    """Find a persisted index for an unsupported configured model, if present."""
    if _configured_embedding_is_supported():
        return None

    configured = (settings.embedding_model or "").strip()
    base_path = Path(settings.faiss_index_path).resolve()
    if not base_path.exists():
        return None

    for meta_file in base_path.glob(f"*/{_META_FILENAME}"):
        meta = _read_index_meta(str(meta_file.parent))
        if meta and str(meta.get("embedding_model", "")).strip() == configured:
            return meta_file.parent
    return None


def _meta_path(index_path: str) -> Path:
    """Return metadata file path stored alongside the FAISS index."""
    return Path(index_path) / _META_FILENAME


def _doc_registry_path(index_path: str) -> Path:
    """Return document registry path used for duplicate detection."""
    return Path(index_path) / _DOC_REGISTRY_FILENAME


def _faiss_index_file(index_path: str) -> Path:
    """Return the raw FAISS index file path."""
    return Path(index_path) / _FAISS_INDEX_FILENAME


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
        "embedding_model": resolve_embedding_model_name(),
        "dimension": dimension,
    }
    meta_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _migrate_json_registry_if_needed() -> None:
    """Move the old model-scoped JSON registry into SQLite once."""
    global _migration_checked
    if _migration_checked:
        return

    _migration_checked = True
    index_path = _index_path()
    registry = _read_doc_registry(index_path)
    if not registry:
        return

    for file_hash, meta in registry.items():
        if not isinstance(meta, dict) or metadata_store.document_exists(file_hash):
            continue
        metadata_store.upsert_document(
            file_hash=file_hash,
            filename=str(meta.get("filename", "unknown")),
            chunk_count=int(meta.get("chunk_count", 0) or 0),
            document_id=str(meta.get("document_id", "")),
            embedding_model=str(meta.get("embedding_model", resolve_embedding_model_name())),
            indexed_at=int(meta.get("indexed_at", 0) or 0),
            parsing_method=str(meta.get("parsing_method", "unknown")),
            upload_path=str(meta.get("upload_path", "")),
            upload_status=str(meta.get("upload_status", "indexed")),
            vision_calls_used=int(meta.get("vision_calls_used", 0) or 0),
        )


def _resolve_current_embedding_dimension(index_path: str, embeddings: Embeddings) -> Optional[int]:
    """
    Resolve current embedding dimension.

    Prefer metadata value for the active model to avoid unnecessary embedding calls.
    Fall back to a single probe embedding when metadata is missing/stale.
    """
    meta = _read_index_meta(index_path)
    if meta and meta.get("embedding_model") == resolve_embedding_model_name():
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


def _create_empty_store(embeddings: Embeddings, dimension: Optional[int] = None) -> FAISS:
    """Create an empty LangChain FAISS wrapper without using pickle state."""
    resolved_dimension = dimension
    if resolved_dimension is None or resolved_dimension <= 0:
        try:
            resolved_dimension = len(embeddings.embed_query("dimension probe"))
        except Exception as exc:
            logger.warning("Could not resolve embedding dimension for empty FAISS index: %s", exc)
            resolved_dimension = 1
    index = faiss.IndexFlatL2(int(resolved_dimension))
    return FAISS(embeddings, index, InMemoryDocstore({}), {})


def _persist_raw_faiss_index(store: FAISS, index_path: str) -> None:
    """Persist only the raw FAISS index and non-pickle metadata."""
    path = Path(index_path)
    path.mkdir(parents=True, exist_ok=True)
    faiss.write_index(store.index, str(_faiss_index_file(index_path)))


def _persist_docstore_chunks(store: FAISS) -> None:
    """Persist LangChain docstore contents in SQLite for safe reconstruction."""
    docstore = getattr(store, "docstore", None)
    doc_dict = getattr(docstore, "_dict", {}) if docstore is not None else {}
    index_to_docstore_id = getattr(store, "index_to_docstore_id", {}) or {}
    rows: list[dict[str, Any]] = []
    for position, docstore_id in sorted(index_to_docstore_id.items(), key=lambda item: int(item[0])):
        doc = doc_dict.get(docstore_id)
        if not isinstance(doc, Document):
            continue
        rows.append(
            {
                "position": int(position),
                "docstore_id": str(docstore_id),
                "content": str(doc.page_content or ""),
                "metadata": dict(doc.metadata or {}),
            }
        )
    metadata_store.replace_vector_chunks(resolve_embedding_model_name(), rows)


def _persist_store(store: FAISS, index_path: str) -> None:
    _persist_raw_faiss_index(store, index_path)
    _persist_docstore_chunks(store)
    _write_index_meta(index_path, _get_index_dimension(store))


def _load_store_from_raw_index(index_path: str, embeddings: Embeddings) -> FAISS:
    """Load a raw FAISS index and rebuild LangChain state from SQLite."""
    index_file = _faiss_index_file(index_path)
    if not index_file.exists():
        logger.info("No raw FAISS index found. Initializing an empty index.")
        return _create_empty_store(embeddings, _resolve_current_embedding_dimension(index_path, embeddings))

    index = faiss.read_index(str(index_file))
    rows = metadata_store.list_vector_chunks(resolve_embedding_model_name())
    docstore: dict[str, Document] = {}
    index_to_docstore_id: dict[int, str] = {}

    for row in rows:
        position = int(row.get("position", 0) or 0)
        docstore_id = str(row.get("docstore_id", ""))
        if not docstore_id:
            continue
        docstore[docstore_id] = Document(
            page_content=str(row.get("content", "")),
            metadata=dict(row.get("metadata", {}) or {}),
        )
        index_to_docstore_id[position] = docstore_id

    if int(getattr(index, "ntotal", 0) or 0) != len(index_to_docstore_id):
        logger.warning(
            "FAISS index/docstore mismatch detected. Resetting unsafe persisted index. index_vectors=%s docstore_rows=%s",
            int(getattr(index, "ntotal", 0) or 0),
            len(index_to_docstore_id),
        )
        _reset_persisted_index(index_path)
        metadata_store.clear_vector_chunks(resolve_embedding_model_name())
        return _create_empty_store(embeddings, _resolve_current_embedding_dimension(index_path, embeddings))

    return FAISS(embeddings, index, InMemoryDocstore(docstore), index_to_docstore_id)


def _normalized_chunk_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower().strip())


def _near_duplicate_text(left: str, right: str) -> bool:
    if left == right:
        return True
    if not left or not right:
        return False
    longer = max(len(left), len(right))
    shorter = min(len(left), len(right))
    if longer == 0 or (longer - shorter) / longer > 0.10:
        return False
    return SequenceMatcher(None, left, right).ratio() > 0.95


def _document_group_key(doc: Document) -> str:
    metadata = getattr(doc, "metadata", {}) or {}
    return (
        str(metadata.get("document_id") or "")
        or str(metadata.get("doc_id") or "")
        or str(metadata.get("file_hash") or "")
        or str(metadata.get("source") or metadata.get("filename") or "unknown")
    )


def _prepare_chunks_for_embedding(chunks: List[Document]) -> List[Document]:
    """Apply chunk metadata normalization, tiny-chunk filtering, and per-document dedupe.

    Documents indexed before the chunking overhaul should be reindexed via the
    existing knowledge-base reindex endpoint to benefit from this metadata and
    chunk quality policy. That endpoint calls the full ingestion pipeline, so it
    reaches this function before vectors are persisted.
    """
    grouped: dict[str, list[Document]] = {}
    for chunk in chunks:
        text = str(getattr(chunk, "page_content", "") or "").strip()
        if len(text) < 50:
            continue
        grouped.setdefault(_document_group_key(chunk), []).append(chunk)

    filtered: list[Document] = []
    for group_key, group_chunks in grouped.items():
        seen_normalized: list[str] = []
        kept: list[Document] = []
        for chunk in group_chunks:
            normalized = _normalized_chunk_text(chunk.page_content)
            if any(_near_duplicate_text(normalized, existing) for existing in seen_normalized):
                continue
            seen_normalized.append(normalized)
            kept.append(chunk)

        if len(kept) < 3:
            logger.warning(
                "Document produced fewer than 3 chunks after filtering | document=%s chunks=%d",
                group_key,
                len(kept),
            )

        total_chunks = len(kept)
        for idx, chunk in enumerate(kept):
            metadata = dict(getattr(chunk, "metadata", {}) or {})
            filename = str(metadata.get("filename") or metadata.get("source") or "unknown")
            document_id = str(metadata.get("document_id") or metadata.get("doc_id") or "")
            section_title = metadata.get("section_title")
            if section_title is None:
                section_title = metadata.get("section") or metadata.get("section_hint")
            element_type = metadata.get("element_type")
            if element_type is None:
                element_type = metadata.get("block_type")

            metadata.update(
                {
                    "doc_id": document_id,
                    "filename": filename,
                    "source": str(metadata.get("source") or filename),
                    "page": int(metadata.get("page", 0) or 0),
                    "section_title": str(section_title).strip() if section_title else None,
                    "element_type": str(element_type).strip() if element_type else None,
                    "chunk_index": idx,
                    "total_chunks": total_chunks,
                }
            )
            if metadata["section_title"] and not metadata.get("section"):
                metadata["section"] = metadata["section_title"]
            chunk.metadata = metadata
            filtered.append(chunk)

    return filtered


def is_document_indexed(file_hash: str) -> bool:
    """Return True when the same file hash has already been ingested."""
    _migrate_json_registry_if_needed()
    return metadata_store.document_exists(file_hash)


def register_indexed_document(
    file_hash: str,
    filename: str,
    chunk_count: int,
    document_id: str,
    parsing_method: str = "unknown",
    upload_path: str = "",
    upload_status: str = "indexed",
    vision_calls_used: int = 0,
    owner_user_id: str = "",
    visibility: str = "shared",
    allowed_roles: list[str] | None = None,
    ocr_applied: bool = False,
    text_coverage_ratio: float = 0.0,
    low_text_pages: int = 0,
    ingestion_warnings: list[str] | None = None,
    is_demo: bool = False,
    demo_session_id: str = "",
    expires_at: int = 0,
) -> None:
    """Persist hash and document metadata for duplicate prevention."""
    metadata_store.upsert_document(
        file_hash=file_hash,
        filename=filename,
        chunk_count=chunk_count,
        document_id=document_id,
        embedding_model=resolve_embedding_model_name(),
        indexed_at=int(time.time()),
        parsing_method=parsing_method,
        upload_path=upload_path,
        upload_status=upload_status,
        vision_calls_used=vision_calls_used,
        owner_user_id=owner_user_id,
        visibility=visibility,
        allowed_roles=allowed_roles,
        ocr_applied=ocr_applied,
        text_coverage_ratio=text_coverage_ratio,
        low_text_pages=low_text_pages,
        ingestion_warnings=ingestion_warnings,
        is_demo=is_demo,
        demo_session_id=demo_session_id,
        expires_at=expires_at,
    )


def list_indexed_documents() -> list[dict[str, Any]]:
    """Return persisted indexed document metadata for UI display."""
    _migrate_json_registry_if_needed()
    return metadata_store.list_documents()


def get_indexed_document(file_hash: str) -> Optional[dict[str, Any]]:
    """Return registry metadata for one document hash if present."""
    _migrate_json_registry_if_needed()
    return metadata_store.get_document(file_hash)


def delete_indexed_document(file_hash: str) -> dict[str, Any]:
    """
    Remove one document from FAISS and the document registry.

    The stored upload file is intentionally left on disk so callers can choose
    whether to reindex or fully delete the document.
    """
    global _store

    with _write_lock:
        index_path = _index_path()
        _migrate_json_registry_if_needed()
        meta = metadata_store.get_document(file_hash)
        if not isinstance(meta, dict):
            raise KeyError(f"Document not found: {file_hash}")

        store = get_or_create_store()
        chunks_deleted = 0
        filename = str(meta.get("filename", ""))
        document_id = str(meta.get("document_id", ""))

        if store is not None:
            docstore = getattr(store, "docstore", None)
            doc_dict = getattr(docstore, "_dict", {}) if docstore is not None else {}
            ids_to_delete: list[str] = []

            for docstore_id, doc in doc_dict.items():
                metadata = getattr(doc, "metadata", {}) or {}
                if (
                    metadata.get("file_hash") == file_hash
                    or (document_id and metadata.get("document_id") == document_id)
                    or (filename and metadata.get("source") == filename)
                ):
                    ids_to_delete.append(str(docstore_id))

            if ids_to_delete:
                delete_fn = getattr(store, "delete", None)
                if not callable(delete_fn):
                    raise RuntimeError("The active FAISS store does not support document deletion.")
                delete_fn(ids=ids_to_delete)
                chunks_deleted = len(ids_to_delete)
                _persist_store(store, index_path)

        metadata_store.delete_document(file_hash)

        return {
            "file_hash": file_hash,
            "filename": filename,
            "chunks_deleted": chunks_deleted,
            "upload_path": str(meta.get("upload_path", "")),
        }


def list_document_chunks(
    file_hash: str,
    focus_chunk_index: int | None = None,
    neighbor_window: int = 0,
) -> list[dict[str, Any]]:
    """Return all stored chunks for one indexed document."""
    store = get_or_create_store()
    if store is None:
        return []

    docstore = getattr(store, "docstore", None)
    doc_dict = getattr(docstore, "_dict", {}) if docstore is not None else {}
    chunks: list[dict[str, Any]] = []

    def _inspect_chunk_quality(text: str, metadata: dict[str, Any]) -> tuple[float, list[str]]:
        compact = re.sub(r"\s+", " ", text or "").strip()
        warnings: list[str] = []
        if len(compact) < 80:
            warnings.append("short_chunk")
        if not str(metadata.get("section") or metadata.get("section_hint") or "").strip():
            warnings.append("missing_section")
        alnum = re.sub(r"[^a-zA-Z0-9]", "", compact)
        if compact and (len(alnum) / max(1, len(compact))) < 0.45:
            warnings.append("low_text_signal")
        if re.search(r"([A-Za-z])\1{4,}", compact):
            warnings.append("repeated_ocr_artifact")
        if len(compact) > 1800:
            warnings.append("oversized_chunk")
        score = max(0.0, 1.0 - (0.18 * len(warnings)))
        return round(score, 4), warnings

    for docstore_id, doc in doc_dict.items():
        metadata = getattr(doc, "metadata", {}) or {}
        if metadata.get("file_hash") != file_hash:
            continue
        content = str(getattr(doc, "page_content", "") or "")
        quality_score, quality_warnings = _inspect_chunk_quality(content, metadata)
        chunks.append(
            {
                "id": str(docstore_id),
                "content": content,
                "page": int(metadata.get("page", 0) or 0),
                "section": str(metadata.get("section") or metadata.get("section_hint") or ""),
                "chunk_index": int(metadata.get("chunk_index", 0) or 0),
                "quality_score": quality_score,
                "quality_warnings": quality_warnings,
                "metadata": dict(metadata),
            }
        )

    chunks.sort(key=lambda item: (int(item.get("page", 0)), int(item.get("chunk_index", 0))))
    if focus_chunk_index is not None and neighbor_window > 0:
        low = int(focus_chunk_index) - int(neighbor_window)
        high = int(focus_chunk_index) + int(neighbor_window)
        return [
            item
            for item in chunks
            if low <= int(item.get("chunk_index", 0)) <= high
        ]
    return chunks


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

    unsupported_index = _find_unsupported_configured_index()
    if unsupported_index is not None:
        effective_model = resolve_embedding_model_name()
        raise RuntimeError(
            "Embedding/index mismatch: EMBEDDING_MODEL is set to "
            f"'{settings.embedding_model}', but this runtime only loads local "
            f"sentence-transformers embeddings and would use '{effective_model}'. "
            f"An existing FAISS index for the unsupported model was found at '{unsupported_index}'. "
            "Set EMBEDDING_MODEL to a sentence-transformers model and reindex, or reset the knowledge base."
        )

    index_path = _index_path()
    embeddings = get_embedding_model()

    if _faiss_index_file(index_path).exists():
        logger.info(f"Loading existing FAISS index from: {index_path}")
        _store = _load_store_from_raw_index(index_path, embeddings)

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
            metadata_store.clear_vector_chunks(resolve_embedding_model_name())
            _store = _create_empty_store(embeddings, current_dim)
    else:
        logger.info("No existing FAISS index found. Initializing an empty index.")
        _store = _create_empty_store(embeddings, _resolve_current_embedding_dimension(index_path, embeddings))

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
        prepared_chunks = _prepare_chunks_for_embedding(chunks)
        chunks[:] = prepared_chunks
        if not prepared_chunks:
            raise RuntimeError("No chunks remain after chunk quality filtering.")
        chunks = prepared_chunks

        add_documents_start = time.perf_counter()
        embeddings = get_embedding_model()
        index_path = _index_path()
        batch_size = max(1, settings.embedding_batch_size)
        backend_name = embedding_backend_name(embeddings)

        # Ensure the parent directory exists
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)

        texts = [doc.page_content for doc in chunks]
        metadatas = [doc.metadata for doc in chunks]
        total_chunks = len(texts)
        total_batches = math.ceil(total_chunks / batch_size) if total_chunks else 0

        logger.info(
            "Embedding start | backend=%s total_chunks=%d batch_size=%d total_batches=%d",
            backend_name,
            total_chunks,
            batch_size,
            total_batches,
        )

        embedding_start = time.perf_counter()
        embed_batch = getattr(embeddings, "embed_batch", None)
        if callable(embed_batch):
            vectors = embed_batch(texts, batch_size=batch_size)
        else:
            vectors = embeddings.embed_documents(texts)

        if len(vectors) != len(texts):
            raise RuntimeError(f"Embedding size mismatch: expected {len(texts)}, got {len(vectors)}")

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
            _store = _create_empty_store(embeddings, new_dim)
        else:
            existing_dim = _get_index_dimension(_store)
            logger.info("FAISS merge pre-check dimensions | existing=%s new=%s", existing_dim, new_dim)

            if existing_dim is not None and new_dim is not None and existing_dim != new_dim:
                logger.warning("FAISS index dimension mismatch detected. Resetting index.")
                _store = None
                _reset_persisted_index(index_path)
                metadata_store.clear_vector_chunks(resolve_embedding_model_name())
                _store = _create_empty_store(embeddings, new_dim)

        if text_embeddings:
            logger.info("Adding %d embedded chunk(s) to FAISS index.", len(chunks))
            _store.add_embeddings(text_embeddings, metadatas=metadatas)

        # Persist after every successful write
        _persist_store(_store, index_path)
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
    index_file = _faiss_index_file(str(index_path))
    if not index_file.exists():
        return "empty"

    meta_mtime = _meta_path(str(index_path)).stat().st_mtime if _meta_path(str(index_path)).exists() else 0
    index_mtime = index_file.stat().st_mtime
    return f"{resolve_embedding_model_name()}:{int(max(meta_mtime, index_mtime))}"


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
        metadata_store.clear_documents()
        metadata_store.clear_vector_chunks()
        return existed


def cleanup_expired_demo_documents() -> int:
    """Remove expired public-demo documents from metadata, FAISS, and uploads."""
    expired = metadata_store.list_expired_demo_documents()
    removed = 0
    for item in expired:
        file_hash = str(item.get("file_hash", ""))
        if not file_hash:
            continue
        upload_path = str(item.get("upload_path", ""))
        try:
            delete_indexed_document(file_hash)
        except Exception:
            metadata_store.delete_document(file_hash)
        if upload_path:
            try:
                Path(upload_path).unlink(missing_ok=True)
            except OSError:
                pass
        removed += 1
    return removed
