"""
reranker.py
───────────
Neural reranking utilities for post-retrieval ordering.

Uses local, open-source BGE reranker models via FlagEmbedding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple
import threading
import time

from langchain.schema import Document

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
_reranker_singleton = None
_reranker_singleton_lock = threading.Lock()


@dataclass
class RerankedDocument:
    """Document plus neural reranker relevance score."""

    document: Document
    score: float


class Reranker:
    """Local neural reranker backed by FlagEmbedding BGE models."""

    def __init__(
        self,
        model_name: str,
        batch_size: int = 16,
        use_fp16: bool = False,
    ) -> None:
        self.model_name = model_name
        self.batch_size = max(1, int(batch_size))
        self.use_fp16 = bool(use_fp16)
        self._model = None
        self._model_lock = threading.Lock()

    def _ensure_model(self):
        if self._model is not None:
            return self._model

        with self._model_lock:
            if self._model is not None:
                return self._model

            try:
                from FlagEmbedding import FlagReranker
            except ImportError as exc:
                raise RuntimeError(
                    "FlagEmbedding is not installed. Add 'FlagEmbedding' to requirements."
                ) from exc

            load_start = time.perf_counter()
            self._model = FlagReranker(
                self.model_name,
                use_fp16=self.use_fp16,
            )
            load_ms = (time.perf_counter() - load_start) * 1000.0
            logger.info(
                "Reranker initialized (first load) | model=%s batch_size=%d fp16=%s load_ms=%.1f",
                self.model_name,
                self.batch_size,
                self.use_fp16,
                load_ms,
            )
        return self._model

    def score_documents(self, query: str, documents: Sequence[Document]) -> List[float]:
        """Return neural relevance scores for query-document pairs."""
        if not documents:
            return []

        model = self._ensure_model()
        pairs = [[query, doc.page_content] for doc in documents]

        scores = model.compute_score(
            pairs,
            batch_size=self.batch_size,
        )

        if isinstance(scores, (float, int)):
            return [float(scores)]

        return [float(s) for s in scores]

    def rerank_documents(
        self,
        query: str,
        documents: Sequence[Document],
        top_n: int,
    ) -> List[RerankedDocument]:
        """Return top_n docs sorted by neural reranker score (descending)."""
        if not documents:
            return []

        requested_top_n = max(1, int(top_n))
        effective_top_n = min(requested_top_n, len(documents))

        scores = self.score_documents(query=query, documents=documents)
        scored = [
            RerankedDocument(document=doc, score=score)
            for doc, score in zip(documents, scores)
        ]
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:effective_top_n]


def get_reranker() -> Reranker:
    """Thread-safe lazy singleton reranker instance for the current process."""
    global _reranker_singleton

    if _reranker_singleton is not None:
        return _reranker_singleton

    with _reranker_singleton_lock:
        if _reranker_singleton is None:
            _reranker_singleton = Reranker(
                model_name=settings.reranker_model_name,
                batch_size=settings.reranker_batch_size,
                use_fp16=settings.reranker_use_fp16,
            )
    return _reranker_singleton


def rerank_documents(
    query: str,
    documents: Sequence[Document],
    top_n: int | None = None,
) -> List[Tuple[Document, float]]:
    """
    Public helper that reranks documents and returns (document, score) tuples.

    Includes safe fallback to original ordering when reranker fails.
    """
    if not documents:
        return []

    resolved_top_n = (
        max(1, int(top_n))
        if top_n is not None
        else max(1, int(settings.retrieval_top_n))
    )

    try:
        reranker = get_reranker()
        rerank_start = time.perf_counter()
        reranked = reranker.rerank_documents(
            query=query,
            documents=documents,
            top_n=resolved_top_n,
        )
        rerank_ms = (time.perf_counter() - rerank_start) * 1000.0
        if reranked:
            logger.info(
                "Rerank | candidates=%d top_n=%d latency_ms=%.1f scores=%s",
                len(documents),
                len(reranked),
                rerank_ms,
                [round(item.score, 4) for item in reranked[:5]],
            )
        return [(item.document, item.score) for item in reranked]
    except Exception as exc:
        logger.warning("Neural reranker unavailable, using fallback ordering: %s", exc)
        # Preserve input order as deterministic fallback with synthetic descending scores.
        fallback_docs = list(documents)[:resolved_top_n]
        fallback_scores = [float(resolved_top_n - i) for i in range(len(fallback_docs))]
        return list(zip(fallback_docs, fallback_scores))
