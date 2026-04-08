"""
retriever.py
────────────
Responsibility: Retrieve relevant chunks for a user question.

This implementation uses a small multi-query strategy to improve recall
for broad/ambiguous questions while preserving the same public interface.
"""

from typing import List, Tuple, Optional
import re
from dataclasses import dataclass

from langchain.schema import Document

from app.services.vector_store import get_or_create_store
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    """Document plus normalized confidence values used downstream."""

    document: Document
    vector_confidence: float
    lexical_score: float
    final_score: float


@dataclass
class RetrievalDebugInfo:
    """Execution metadata for observability and tuning."""

    query_variants_used: list[str]
    is_broad_question: bool
    fallback_applied: bool
    candidates_considered: int


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


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))


def _lexical_overlap_score(question: str, content: str) -> float:
    q_tokens = _tokenize(question)
    if not q_tokens:
        return 0.0
    c_tokens = _tokenize(content)
    if not c_tokens:
        return 0.0
    overlap = len(q_tokens.intersection(c_tokens))
    return min(1.0, overlap / max(1, len(q_tokens)))


def _distance_to_confidence(distance: Optional[float]) -> float:
    if distance is None:
        return 0.0
    # FAISS typically returns distance where lower is better; map to [0,1].
    return 1.0 / (1.0 + max(0.0, float(distance)))


def _build_fallback_variants(question: str) -> list[str]:
    q = question.strip()
    candidates = [
        f"detailed explanation {q}",
        f"important points {q}",
        f"rules and policy {q}",
    ]
    seen: set[str] = set()
    deduped: list[str] = []
    for item in candidates:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _retrieve_candidates(
    store,
    question: str,
    query_variants: list[str],
    k: int,
) -> tuple[list[RetrievedChunk], int]:
    scored: dict[tuple[str, int, str], RetrievedChunk] = {}
    candidates_considered = 0

    for variant in query_variants:
        try:
            results: list[Tuple[Document, float]] = store.similarity_search_with_score(variant, k=k)
            candidates_considered += len(results)
            for doc, distance in results:
                vector_conf = _distance_to_confidence(distance)
                lexical = _lexical_overlap_score(question, doc.page_content)
                final = 0.75 * vector_conf + 0.25 * lexical

                source = str(doc.metadata.get("source", "unknown"))
                page = int(doc.metadata.get("page", 0) or 0)
                content_head = doc.page_content[:200].strip().lower()
                key = (source, page, content_head)
                existing = scored.get(key)
                candidate = RetrievedChunk(
                    document=doc,
                    vector_confidence=vector_conf,
                    lexical_score=lexical,
                    final_score=final,
                )
                if existing is None or candidate.final_score > existing.final_score:
                    scored[key] = candidate
        except Exception:
            # Fallback for stores/backends that do not expose score APIs.
            docs = store.similarity_search(variant, k=k)
            candidates_considered += len(docs)
            for doc in docs:
                lexical = _lexical_overlap_score(question, doc.page_content)
                source = str(doc.metadata.get("source", "unknown"))
                page = int(doc.metadata.get("page", 0) or 0)
                content_head = doc.page_content[:200].strip().lower()
                key = (source, page, content_head)
                existing = scored.get(key)
                candidate = RetrievedChunk(
                    document=doc,
                    vector_confidence=0.0,
                    lexical_score=lexical,
                    final_score=0.25 * lexical,
                )
                if existing is None or candidate.final_score > existing.final_score:
                    scored[key] = candidate

    ranked = sorted(scored.values(), key=lambda item: item.final_score, reverse=True)
    return ranked, candidates_considered


def retrieve_relevant_chunks_with_diagnostics(
    question: str,
) -> tuple[list[RetrievedChunk], RetrievalDebugInfo]:
    """Retrieve, rerank, and optionally perform one deterministic fallback pass."""
    store = get_or_create_store()

    if store is None:
        raise RuntimeError(
            "The knowledge base is empty. Please upload at least one PDF document first."
        )

    base_k = settings.retrieval_top_k
    candidate_k = max(base_k, settings.retrieval_candidate_k)
    is_broad = _is_broad_question(question)
    query_variants = _build_query_variants(question)

    logger.info(
        "Retrieving chunks | base_k=%d candidate_k=%d query_variants=%d question='%s...'",
        base_k,
        candidate_k,
        len(query_variants),
        question[:80],
    )

    ranked, considered = _retrieve_candidates(
        store=store,
        question=question,
        query_variants=query_variants,
        k=candidate_k,
    )

    fallback_applied = False
    top_conf = ranked[0].final_score if ranked else 0.0
    if (
        settings.enable_retrieval_fallback
        and top_conf < settings.retrieval_low_confidence_threshold
    ):
        fallback_applied = True
        fallback_variants = _build_fallback_variants(question)
        query_variants.extend(fallback_variants)
        fallback_ranked, fallback_considered = _retrieve_candidates(
            store=store,
            question=question,
            query_variants=fallback_variants,
            k=candidate_k,
        )
        considered += fallback_considered

        merged: dict[tuple[str, int, str], RetrievedChunk] = {}
        for item in ranked + fallback_ranked:
            source = str(item.document.metadata.get("source", "unknown"))
            page = int(item.document.metadata.get("page", 0) or 0)
            content_head = item.document.page_content[:200].strip().lower()
            key = (source, page, content_head)
            existing = merged.get(key)
            if existing is None or item.final_score > existing.final_score:
                merged[key] = item
        ranked = sorted(merged.values(), key=lambda item: item.final_score, reverse=True)

    final_k = base_k if not is_broad else max(base_k, 6)
    selected = ranked[:final_k]

    logger.info(
        "Retrieved %d chunk(s) after rerank | fallback=%s top_score=%.3f",
        len(selected),
        fallback_applied,
        selected[0].final_score if selected else 0.0,
    )

    return selected, RetrievalDebugInfo(
        query_variants_used=query_variants,
        is_broad_question=is_broad,
        fallback_applied=fallback_applied,
        candidates_considered=considered,
    )


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
    chunks, _ = retrieve_relevant_chunks_with_diagnostics(question)
    return [item.document for item in chunks]
