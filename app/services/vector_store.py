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

import os
from pathlib import Path
from typing import List, Optional

from langchain.schema import Document
from langchain_community.vectorstores import FAISS

from app.services.embedding_service import get_embedding_model
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Module-level reference so the store is only loaded from disk once per
# process.  This acts as a simple in-process cache.
_store: Optional[FAISS] = None


def _index_path() -> str:
    """Return the absolute path used for FAISS index persistence."""
    return str(Path(settings.faiss_index_path).resolve())


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

    embeddings = get_embedding_model()
    index_path = _index_path()

    # Ensure the parent directory exists
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)

    if _store is None:
        logger.info(f"Creating new FAISS index with {len(chunks)} chunk(s).")
        _store = FAISS.from_documents(chunks, embeddings)
    else:
        logger.info(f"Merging {len(chunks)} new chunk(s) into existing FAISS index.")
        new_store = FAISS.from_documents(chunks, embeddings)
        _store.merge_from(new_store)

    # Persist after every successful write
    _store.save_local(index_path)
    logger.info(f"FAISS index saved to: {index_path}")

    return _store


def load_store() -> Optional[FAISS]:
    """
    Explicitly reload the store from disk (e.g. after a restart).
    Returns None if no index has been saved yet.
    """
    global _store
    _store = None          # clear cache to force a fresh load
    return get_or_create_store()
