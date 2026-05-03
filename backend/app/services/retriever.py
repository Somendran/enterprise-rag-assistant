"""
retriever.py
────────────
Responsibility: Retrieve relevant chunks for a user question.

This implementation uses a small multi-query strategy to improve recall
for broad/ambiguous questions while preserving the same public interface.
"""

from typing import List, Tuple, Optional
import re
import time
import math
from collections import Counter
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from langchain.schema import Document

from app.services.vector_store import get_or_create_store
from app.services.reranker import rerank_documents
from app.config import settings
from app.utils.logger import get_logger
from app.utils.query_normalization import normalize_query

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    """Document plus normalized confidence values used downstream."""

    document: Document
    vector_confidence: float
    lexical_score: float
    bm25_score: float
    final_score: float


@dataclass
class RetrievalDebugInfo:
    """Execution metadata for observability and tuning."""

    query_variants_used: list[str]
    query_type: str
    is_broad_question: bool
    is_simple_query: bool
    fast_mode_applied: bool
    fallback_applied: bool
    candidates_considered: int
    reranker_applied: bool
    reranker_skipped_reason: str
    retrieval_ms: float
    rerank_ms: float


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


def classify_query(question: str) -> str:
    """Return a coarse query profile used for retrieval and eval diagnostics."""
    text = normalize_query(question)
    if not text:
        return "general"

    if any(term in text for term in ("summarize", "summary", "overview", "key points", "high level")):
        return "summary"
    if any(term in text for term in ("compare", "difference", "versus", " vs ", "similarities", "tradeoff", "trade-off")):
        return "comparison"
    if any(term in text for term in ("how many", "how much", "what date", "when", "within", "at least")):
        return "lookup"
    if any(term in text for term in ("analyze", "evaluate", "risk", "impact", "why", "implication", "exceptions")):
        return "complex"
    return "general"


def is_simple_query(query: str) -> bool:
    """CPU-cheap, deterministic complexity classifier for adaptive fast mode.

    Simple examples:
    - "leave policy"
    - "what is maternity leave"
    - "how many annual leave days"

    Complex examples:
    - "compare annual leave and sick leave eligibility by tenure"
    - "summarize leave policy risks and exceptions"
    """
    text = normalize_query(query)
    if not text:
        return True

    tokens = text.split()
    token_count = len(tokens)

    direct_question_prefixes = (
        "what is",
        "what are",
        "who is",
        "when is",
        "where is",
        "how many",
        "how much",
        "is there",
        "can i",
        "does",
        "do we",
    )
    heavy_reasoning_keywords = {
        "compare",
        "difference",
        "versus",
        "vs",
        "summarize",
        "summary",
        "analyze",
        "analysis",
        "evaluate",
        "evaluation",
        "tradeoff",
        "trade-off",
        "root cause",
        "implication",
        "exceptions",
        "edge case",
        "limitations",
    }
    synthesis_keywords = {
        "why",
        "how",
        "impact",
        "risk",
        "timeline",
    }

    has_heavy_reasoning = any(term in text for term in heavy_reasoning_keywords)
    has_synthesis_intent = any(term in text for term in synthesis_keywords)
    has_multi_clause = any(marker in text for marker in [";", ":", " and ", " or ", ","])
    is_direct_question = any(text.startswith(prefix) for prefix in direct_question_prefixes)

    if token_count <= int(settings.simple_query_short_token_limit) and not has_heavy_reasoning:
        return True

    if (
        is_direct_question
        and token_count <= int(settings.simple_query_direct_token_limit)
        and not has_heavy_reasoning
        and not has_multi_clause
    ):
        return True

    if has_heavy_reasoning:
        return False

    if has_synthesis_intent and token_count > int(settings.simple_query_short_token_limit):
        return False

    return False


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


def _doc_key(doc: Document) -> tuple[str, int, str]:
    source = str(doc.metadata.get("source", "unknown"))
    page = int(doc.metadata.get("page", 0) or 0)
    content_head = doc.page_content[:200].strip().lower()
    return (source, page, content_head)


def _combine_score(vector_conf: float, lexical: float, bm25: float) -> float:
    vector_weight = float(settings.vector_weight)
    lexical_weight = float(settings.lexical_weight)
    bm25_weight = float(settings.bm25_weight)
    total = max(1e-6, vector_weight + lexical_weight + bm25_weight)
    return (
        (vector_weight * vector_conf)
        + (lexical_weight * lexical)
        + (bm25_weight * bm25)
    ) / total


def _normalize_reranker_score(score: float) -> float:
    """Map raw cross-encoder logit to [0, 1] for stable confidence handling."""
    clipped = max(-12.0, min(12.0, float(score)))
    return 1.0 / (1.0 + math.exp(-clipped))


def _reranker_skip_reason(
    *,
    fast_mode_applied: bool,
    query_type: str,
    initial_candidates: list[RetrievedChunk],
    final_top_n: int,
) -> str:
    if not settings.enable_neural_reranker:
        return "reranker_disabled"

    if bool(settings.complex_query_rerank_always) and query_type in {"summary", "comparison", "complex"}:
        if not initial_candidates:
            return "no_candidates"
        return ""

    if fast_mode_applied:
        return "fast_mode_simple_query"

    min_candidates = max(final_top_n + 1, int(settings.reranker_min_candidates))
    if len(initial_candidates) < min_candidates:
        return f"insufficient_candidates:{len(initial_candidates)}<{min_candidates}"

    if not initial_candidates:
        return "no_candidates"

    top_score = initial_candidates[0].final_score
    if (
        bool(settings.reranker_skip_if_high_confidence)
        and top_score >= float(settings.reranker_high_confidence_threshold)
    ):
        return f"high_confidence:{top_score:.3f}"

    if len(initial_candidates) >= 2:
        score_gap = initial_candidates[0].final_score - initial_candidates[1].final_score
        if (
            bool(settings.reranker_skip_if_score_gap)
            and score_gap >= float(settings.reranker_score_gap_threshold)
        ):
            return f"high_score_gap:{score_gap:.3f}"

    return ""


def _get_store_documents(store) -> list[Document]:
    docstore = getattr(store, "docstore", None)
    backing = getattr(docstore, "_dict", None)
    if not isinstance(backing, dict):
        return []
    return [doc for doc in backing.values() if isinstance(doc, Document) and doc.page_content.strip()]


def _retrieve_bm25_candidates(
    store,
    question: str,
    top_k: int,
    allowed_file_hashes: set[str] | None = None,
) -> tuple[list[RetrievedChunk], int]:
    docs = _get_store_documents(store)
    if allowed_file_hashes is not None:
        docs = [
            doc for doc in docs
            if str(doc.metadata.get("file_hash", "")) in allowed_file_hashes
        ]
    if not docs:
        return [], 0

    tokenized_docs: list[list[str]] = [re.findall(r"[a-zA-Z0-9]+", doc.page_content.lower()) for doc in docs]
    doc_freq: Counter[str] = Counter()
    doc_lens: list[int] = []
    for tokens in tokenized_docs:
        doc_lens.append(len(tokens))
        doc_freq.update(set(tokens))

    n_docs = len(docs)
    avgdl = max(1e-6, sum(doc_lens) / max(1, n_docs))
    query_tokens = re.findall(r"[a-zA-Z0-9]+", question.lower())
    if not query_tokens:
        return [], n_docs

    k1 = 1.2
    b = 0.75
    scores: list[float] = []
    for tokens, dl in zip(tokenized_docs, doc_lens):
        tf = Counter(tokens)
        score = 0.0
        for term in query_tokens:
            f = tf.get(term, 0)
            if f <= 0:
                continue
            df = doc_freq.get(term, 0)
            idf = math.log(1.0 + ((n_docs - df + 0.5) / (df + 0.5)))
            denom = f + k1 * (1.0 - b + b * (dl / avgdl))
            score += idf * ((f * (k1 + 1.0)) / max(1e-6, denom))
        scores.append(score)

    max_score = max(scores) if scores else 0.0
    if max_score <= 0:
        return [], n_docs

    ranked_idx = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)[: max(1, top_k)]
    ranked: list[RetrievedChunk] = []
    for idx in ranked_idx:
        doc = docs[idx]
        bm25_norm = float(scores[idx] / max_score)
        lexical = _lexical_overlap_score(question, doc.page_content)
        ranked.append(
            RetrievedChunk(
                document=doc,
                vector_confidence=0.0,
                lexical_score=lexical,
                bm25_score=bm25_norm,
                final_score=_combine_score(0.0, lexical, bm25_norm),
            )
        )

    return ranked, n_docs


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
    allowed_file_hashes: set[str] | None = None,
) -> tuple[list[RetrievedChunk], int]:
    scored: dict[tuple[str, int, str], RetrievedChunk] = {}
    candidates_considered = 0

    for variant in query_variants:
        try:
            results: list[Tuple[Document, float]] = store.similarity_search_with_score(variant, k=k)
            candidates_considered += len(results)
            for doc, distance in results:
                if allowed_file_hashes is not None and str(doc.metadata.get("file_hash", "")) not in allowed_file_hashes:
                    continue
                vector_conf = _distance_to_confidence(distance)
                lexical = _lexical_overlap_score(question, doc.page_content)

                source = str(doc.metadata.get("source", "unknown"))
                page = int(doc.metadata.get("page", 0) or 0)
                content_head = doc.page_content[:200].strip().lower()
                key = (source, page, content_head)
                existing = scored.get(key)
                candidate = RetrievedChunk(
                    document=doc,
                    vector_confidence=vector_conf,
                    lexical_score=lexical,
                    bm25_score=0.0,
                    final_score=_combine_score(vector_conf, lexical, 0.0),
                )
                if existing is None or candidate.final_score > existing.final_score:
                    scored[key] = candidate
        except Exception:
            # Fallback for stores/backends that do not expose score APIs.
            docs = store.similarity_search(variant, k=k)
            candidates_considered += len(docs)
            for doc in docs:
                if allowed_file_hashes is not None and str(doc.metadata.get("file_hash", "")) not in allowed_file_hashes:
                    continue
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
                    bm25_score=0.0,
                    final_score=_combine_score(0.0, lexical, 0.0),
                )
                if existing is None or candidate.final_score > existing.final_score:
                    scored[key] = candidate

    ranked = sorted(scored.values(), key=lambda item: item.final_score, reverse=True)
    return ranked, candidates_considered


def retrieve_relevant_chunks_with_diagnostics(
    question: str,
    allowed_file_hashes: list[str] | None = None,
) -> tuple[list[RetrievedChunk], RetrievalDebugInfo]:
    """Retrieve, rerank, and optionally perform one deterministic fallback pass."""
    question = normalize_query(question)
    retrieval_start = time.perf_counter()
    store = get_or_create_store()

    if store is None:
        raise RuntimeError(
            "The knowledge base is empty. Please upload at least one PDF document first."
        )

    simple_query = is_simple_query(question)
    query_type = classify_query(question)
    fast_mode_applied = bool(
        settings.fast_mode_enabled
        and simple_query
        and query_type not in {"summary", "comparison", "complex"}
    )

    if query_type == "summary":
        final_top_n = max(1, int(settings.summary_retrieval_top_n))
    elif fast_mode_applied:
        final_top_n = max(1, int(settings.fast_mode_top_n))
    else:
        final_top_n = max(1, int(settings.retrieval_top_n))
    configured_initial_top_k = (
        max(final_top_n, int(settings.fast_mode_initial_top_k))
        if fast_mode_applied
        else max(final_top_n, int(settings.retrieval_initial_top_k))
    )
    access_filter = {str(item) for item in allowed_file_hashes} if allowed_file_hashes is not None else None
    initial_top_k = configured_initial_top_k
    candidate_k = max(initial_top_k, int(settings.retrieval_candidate_k))
    if access_filter is not None:
        if not access_filter:
            raise RuntimeError("No documents are available to your account.")
        candidate_k = max(candidate_k, configured_initial_top_k * 4, len(access_filter) * final_top_n)
    is_broad = _is_broad_question(question)
    query_variants = _build_query_variants(question)

    logger.info(
        "Retrieving chunks | fast_mode=%s simple_query=%s initial_top_k=%d final_top_n=%d candidate_k=%d query_variants=%d question='%s...'",
        fast_mode_applied,
        simple_query,
        initial_top_k,
        final_top_n,
        candidate_k,
        len(query_variants),
        question[:80],
    )

    bm25_k = max(final_top_n, int(settings.bm25_top_k))
    with ThreadPoolExecutor(max_workers=2) as executor:
        vector_future = executor.submit(
            _retrieve_candidates,
            store,
            question,
            query_variants,
            candidate_k,
            access_filter,
        )
        bm25_future = executor.submit(
            _retrieve_bm25_candidates,
            store,
            question,
            bm25_k,
            access_filter,
        )

        ranked_vector, considered_vector = vector_future.result()
        ranked_bm25, considered_bm25 = bm25_future.result()

    merged_ranked: dict[tuple[str, int, str], RetrievedChunk] = {}
    for item in ranked_vector + ranked_bm25:
        key = _doc_key(item.document)
        existing = merged_ranked.get(key)
        if existing is None:
            merged_ranked[key] = item
            continue

        vector_conf = max(existing.vector_confidence, item.vector_confidence)
        lexical = max(existing.lexical_score, item.lexical_score)
        bm25_score = max(existing.bm25_score, item.bm25_score)
        merged_ranked[key] = RetrievedChunk(
            document=existing.document,
            vector_confidence=vector_conf,
            lexical_score=lexical,
            bm25_score=bm25_score,
            final_score=_combine_score(vector_conf, lexical, bm25_score),
        )

    ranked = sorted(merged_ranked.values(), key=lambda item: item.final_score, reverse=True)
    considered = considered_vector + considered_bm25

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
            allowed_file_hashes=access_filter,
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

    # Stage 1: keep initial_top_k retrieved candidates before neural rerank.
    initial_candidates = ranked[:initial_top_k]

    # Stage 2: optional neural rerank (cross-encoder) then select final_top_n.
    reranker_applied = False
    reranker_skipped_reason = _reranker_skip_reason(
        fast_mode_applied=fast_mode_applied,
        query_type=query_type,
        initial_candidates=initial_candidates,
        final_top_n=final_top_n,
    )
    rerank_ms = 0.0

    rerank_pool_k = max(final_top_n, int(settings.rerank_top_k))
    rerank_candidates = initial_candidates[: min(len(initial_candidates), rerank_pool_k)]
    logger.info(
        "Rerank scope | pool_cap=%d initial_candidates=%d rerank_candidates=%d final_top_n=%d",
        rerank_pool_k,
        len(initial_candidates),
        len(rerank_candidates),
        final_top_n,
    )

    if rerank_candidates and not reranker_skipped_reason:
        rerank_start = time.perf_counter()
        pairs = rerank_documents(
            query=question,
            documents=[item.document for item in rerank_candidates],
            top_n=final_top_n,
        )
        rerank_ms = (time.perf_counter() - rerank_start) * 1000.0
        reranker_applied = True

        # Map doc signature back to retrieved chunk metadata.
        by_key: dict[tuple[str, int, str], RetrievedChunk] = {}
        for item in rerank_candidates:
            source = str(item.document.metadata.get("source", "unknown"))
            page = int(item.document.metadata.get("page", 0) or 0)
            content_head = item.document.page_content[:200].strip().lower()
            by_key[(source, page, content_head)] = item

        selected: list[RetrievedChunk] = []
        for doc, rr_score in pairs:
            source = str(doc.metadata.get("source", "unknown"))
            page = int(doc.metadata.get("page", 0) or 0)
            content_head = doc.page_content[:200].strip().lower()
            key = (source, page, content_head)
            original = by_key.get(key)
            if original is None:
                continue

            selected.append(
                RetrievedChunk(
                    document=original.document,
                    vector_confidence=original.vector_confidence,
                    lexical_score=original.lexical_score,
                    bm25_score=original.bm25_score,
                    final_score=_normalize_reranker_score(float(rr_score)),
                )
            )

        if not selected:
            # Safety fallback: keep original ordering if reranker output is empty.
            selected = initial_candidates[:final_top_n]
    else:
        selected = initial_candidates[:final_top_n]

    retrieval_ms = (time.perf_counter() - retrieval_start) * 1000.0

    logger.info(
        "Retrieved %d chunk(s) | reranker_applied=%s rerank_ms=%.1f skip_reason=%s fallback=%s top_score=%.3f",
        len(selected),
        reranker_applied,
        rerank_ms,
        reranker_skipped_reason,
        fallback_applied,
        selected[0].final_score if selected else 0.0,
    )

    return selected, RetrievalDebugInfo(
        query_variants_used=query_variants,
        query_type=query_type,
        is_broad_question=is_broad,
        is_simple_query=simple_query,
        fast_mode_applied=fast_mode_applied,
        fallback_applied=fallback_applied,
        candidates_considered=considered,
        reranker_applied=reranker_applied,
        reranker_skipped_reason=reranker_skipped_reason,
        retrieval_ms=retrieval_ms,
        rerank_ms=rerank_ms,
    )


def retrieve_relevant_chunks(question: str, allowed_file_hashes: list[str] | None = None) -> List[Document]:
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
    chunks, _ = retrieve_relevant_chunks_with_diagnostics(question, allowed_file_hashes=allowed_file_hashes)
    return [item.document for item in chunks]
