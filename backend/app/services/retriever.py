"""
retriever.py
────────────
Responsibility: Retrieve relevant chunks for a user question.

This implementation uses a small multi-query strategy to improve recall
for broad/ambiguous questions while preserving the same public interface.
"""

from typing import List
import re

from langchain.schema import Document

from app.services.vector_store import get_or_create_store
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _is_broad_question(question: str) -> bool:
    text = question.lower()
    broad_signals = [
        "about",
        "overview",
        "summary",
        "describe",
        "policy",
        "what is",
    ]
    return any(token in text for token in broad_signals) or len(text.split()) <= 6


def _build_query_variants(question: str) -> list[str]:
    """
    Build lightweight query variants to increase retrieval recall.
    """
    q = question.strip()
    variants = [q]

    # Encourage retrieval of definition/overview chunks for broad prompts.
    if _is_broad_question(q):
        variants.append(f"overview {q}")
        variants.append(f"policy details {q}")
        variants.append(f"key points {q}")

    # Add a normalized variant without punctuation noise.
    normalized = re.sub(r"[^a-zA-Z0-9\s]", " ", q)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if normalized and normalized.lower() != q.lower():
        variants.append(normalized)

    # Preserve order and remove duplicates.
    seen = set()
    deduped: list[str] = []
    for item in variants:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped


def _dedupe_chunks(chunks: list[Document]) -> list[Document]:
    """Deduplicate retrieved chunks by source/page/content signature."""
    seen = set()
    unique: list[Document] = []
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", "?")
        content_head = chunk.page_content[:200].strip().lower()
        key = (source, page, content_head)
        if key in seen:
            continue
        seen.add(key)
        unique.append(chunk)
    return unique


def retrieve_relevant_chunks(question: str) -> List[Document]:
    """
    Perform a similarity search and return the top-k matching chunks.

    Args:
        question: The raw user question string.

    Returns:
        A list of Document objects (chunks) with their original metadata.
        Returns an empty list if the vector store is empty.

    Raises:
        RuntimeError: If the vector store has not been initialised yet
                      (i.e., no documents have been uploaded).
    """
    store = get_or_create_store()

    if store is None:
        # No documents have been uploaded yet — inform the caller gracefully
        raise RuntimeError(
            "The knowledge base is empty. Please upload at least one PDF document first."
        )

    base_k = settings.retrieval_top_k
    variants = _build_query_variants(question)
    per_query_k = max(base_k, 3)

    logger.info(
        "Retrieving chunks | base_k=%d per_query_k=%d query_variants=%d question='%s...'",
        base_k,
        per_query_k,
        len(variants),
        question[:80],
    )

    merged: list[Document] = []
    for q in variants:
        # similarity_search returns Documents ranked by vector similarity.
        merged.extend(store.similarity_search(q, k=per_query_k))

    chunks = _dedupe_chunks(merged)

    # Keep response size controlled to balance quality and latency.
    final_k = base_k if not _is_broad_question(question) else max(base_k, 6)
    chunks = chunks[:final_k]

    logger.info(f"Retrieved {len(chunks)} chunk(s).")
    return chunks
