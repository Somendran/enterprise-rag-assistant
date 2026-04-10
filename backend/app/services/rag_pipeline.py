"""
rag_pipeline.py
───────────────
Responsibility: Orchestrate the full RAG flow:
  1. Retrieve relevant chunks for the question.
  2. Format them into a context string.
  3. Build the prompt and call the LLM.
  4. Return the answer text and deduplicated source references.

This is the only service that "knows about" both retrieval and generation.
All other services are called through this one during a query.
"""

import time
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, List

from app.services.retriever import retrieve_relevant_chunks_with_diagnostics, RetrievedChunk
from app.services.llm_service import generate_answer
from app.services.openai_llm_service import (
    generate_response as generate_openai_response,
    stream_response as stream_openai_response,
)
from app.services.query_cache import (
    build_cache_key,
    build_prompt_fingerprint,
    get_cached_result,
    set_cached_result,
)
from app.services.vector_store import get_knowledge_base_version
from app.prompts.qa_prompt import QA_PROMPT
from app.models.schemas import SourceReference, RetrievalDiagnostics
from app.config import settings
from app.utils.logger import get_logger
from app.utils.query_normalization import normalize_query

logger = get_logger(__name__)


@dataclass
class PipelineResult:
    answer: str
    sources: List[SourceReference]
    confidence_score: float
    confidence_level: str
    diagnostics: RetrievalDiagnostics


def _result_to_payload(result: PipelineResult) -> dict:
    return {
        "answer": result.answer,
        "sources": [src.model_dump() for src in result.sources],
        "confidence_score": result.confidence_score,
        "confidence_level": result.confidence_level,
        "diagnostics": result.diagnostics.model_dump(),
    }


def _payload_to_result(payload: dict) -> PipelineResult:
    return PipelineResult(
        answer=str(payload.get("answer", "")),
        sources=[SourceReference(**src) for src in payload.get("sources", [])],
        confidence_score=float(payload.get("confidence_score", 0.0)),
        confidence_level=str(payload.get("confidence_level", "low")),
        diagnostics=RetrievalDiagnostics(**payload.get("diagnostics", {})),
    )


def _clean_answer_text(answer: str) -> str:
    """
    Remove citation-style noise from LLM output while preserving
    markdown formatting for rich rendering in the frontend.

    The API already returns structured sources, so we strip duplicate
    trailing citation blocks from the answer body.
    """
    lines = [line.rstrip() for line in answer.splitlines()]
    cleaned: list[str] = []

    # Matches lines like:
    # - "Tutorial 3.pdf, Page: 1"
    # - "- Report.pdf (Page 2)"
    citation_line = re.compile(
        r"^\s*(?:[-*]\s*)?.+\.pdf\s*(?:,|\()\s*page\s*[:\d\s\)]",
        re.IGNORECASE,
    )

    for line in lines:
        text = line.strip()
        if not text:
            cleaned.append("")
            continue
        if text.lower() in {"sources:", "source:"}:
            continue
        if citation_line.match(text):
            continue

        # Preserve the original line (including markdown formatting)
        cleaned.append(line)

    # Collapse multiple blank lines and trim outer whitespace.
    result = "\n".join(cleaned)
    result = re.sub(r"\n{3,}", "\n\n", result).strip()
    return result


def _format_context(chunks, max_chars: int) -> str:
    """
    Convert retrieved chunks into a compact context block.

    Heavy text cleanup now runs during ingestion. Query-time formatting only
    applies lightweight compression to keep latency low.
    """
    def _compress_chunk_text(text: str, limit: int = 450) -> str:
        if len(text) <= limit:
            return text

        # Prefer ending on sentence punctuation when available.
        candidates = [text.rfind(".", 0, limit), text.rfind("!", 0, limit), text.rfind("?", 0, limit)]
        sentence_end = max(candidates)
        if sentence_end >= int(limit * 0.6):
            return text[: sentence_end + 1].strip()

        # Otherwise cut on the nearest word boundary.
        space_end = text.rfind(" ", 0, limit)
        if space_end > 0:
            return text[:space_end].strip()

        return text[:limit].strip()

    def _trim_low_signal_lines(text: str) -> str:
        noise_patterns = [
            r"^contact\b",
            r"investor relations",
            r"machine-readable medium",
            r"permission of",
            r"^page\s+\d+\s+of\s+\d+",
        ]
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        kept: list[str] = []
        for line in lines:
            lowered = line.lower()
            if any(re.search(pattern, lowered, re.IGNORECASE) for pattern in noise_patterns):
                continue
            if len(lowered) < 3:
                continue
            kept.append(line)
        return "\n".join(kept).strip()

    # Improve coherence by presenting chunks in source/page order.
    ordered_chunks = sorted(
        chunks,
        key=lambda d: (
            str(d.metadata.get("source", "unknown")),
            int(d.metadata.get("page", 0) or 0),
        ),
    )

    sections = []
    for chunk in ordered_chunks:
        cleaned_text = _trim_low_signal_lines(chunk.page_content.strip())
        cleaned_text = _compress_chunk_text(cleaned_text, limit=450)
        if cleaned_text:
            sections.append(cleaned_text)

    if not sections:
        return ""

    separator = "\n---\n"
    budget = max(500, int(max_chars))
    selected_sections: list[str] = []
    used = 0
    for section in sections:
        section_len = len(section)
        extra = section_len + (len(separator) if selected_sections else 0)
        if selected_sections and (used + extra) > budget:
            break
        if not selected_sections and section_len > budget:
            selected_sections.append(section[:budget].rstrip())
            break
        selected_sections.append(section)
        used += extra

    return separator.join(selected_sections).rstrip()


def _is_summary_request(question: str) -> bool:
    q = question.lower()
    summary_hints = (
        "summarize",
        "summary",
        "key points",
        "overview",
        "high level",
    )
    return any(hint in q for hint in summary_hints)


def _extract_sources(chunks: list[RetrievedChunk]) -> List[SourceReference]:
    """
    Deduplicate the source references from the retrieved chunks.
    A (document, page) pair is unique by definition.
    """
    seen = set()
    sources = []
    for chunk in chunks:
        source = chunk.document.metadata.get("source", "unknown")
        page = chunk.document.metadata.get("page", 0)
        key = (source, page)
        if key not in seen:
            seen.add(key)
            sources.append(
                SourceReference(
                    document=source,
                    page=page,
                    relevance_score=round(float(chunk.final_score), 4),
                )
            )
    return sources


def _confidence_level(confidence: float) -> str:
    if confidence >= 0.65:
        return "high"
    if confidence >= 0.40:
        return "medium"
    return "low"


def _apply_low_confidence_disclaimer(answer: str) -> str:
    disclaimer = (
        "Note: The available document evidence is limited; this answer is best-effort and may be incomplete.\n\n"
    )
    if not answer.strip():
        return disclaimer + "I don't know"
    return disclaimer + answer


def _select_chunks_for_context(
    chunks: list[RetrievedChunk],
    question: str,
    is_simple_query: bool,
) -> list[RetrievedChunk]:
    if not chunks:
        return []

    def _dedupe_by_overlap(items: list[RetrievedChunk]) -> list[RetrievedChunk]:
        seen: set[tuple[str, int, str]] = set()
        unique: list[RetrievedChunk] = []
        for item in items:
            source = str(item.document.metadata.get("source", "unknown"))
            page = int(item.document.metadata.get("page", 0) or 0)
            head = re.sub(r"\s+", " ", item.document.page_content[:220].strip().lower())
            key = (source, page, head)
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)
        return unique

    chunks = _dedupe_by_overlap(chunks)

    if _is_summary_request(question):
        return chunks[: min(3, len(chunks))]

    if len(chunks) == 1:
        return chunks

    gap = chunks[0].final_score - chunks[1].final_score
    if gap >= float(settings.context_dominant_gap_threshold):
        top_count = 1 if chunks[0].final_score >= 0.85 else 2
        return chunks[: min(top_count, len(chunks))]

    # Keep only chunks that remain reasonably close to top score.
    top_score = chunks[0].final_score
    score_floor = max(0.20, top_score * 0.65)
    high_signal = [chunk for chunk in chunks if chunk.final_score >= score_floor]
    candidate_pool = high_signal if high_signal else chunks

    if is_simple_query:
        simple_count = 2 if candidate_pool[0].final_score >= 0.70 else 3
        return candidate_pool[: min(simple_count, len(candidate_pool))]

    return candidate_pool[: min(5, len(candidate_pool))]


def _chunk_identity(chunk: RetrievedChunk) -> str:
    source = str(chunk.document.metadata.get("source", "unknown"))
    page = int(chunk.document.metadata.get("page", 0) or 0)
    head = re.sub(r"\s+", " ", chunk.document.page_content[:120].strip().lower())
    return f"{source}:{page}:{head}"


def _build_generation_prompt(context: str, question: str, trim_prompt: bool) -> str:
    if not trim_prompt:
        return QA_PROMPT.format(context=context, question=question)

    # Short prompt for latency-sensitive fast mode while preserving grounding rules.
    return (
        "Use only CONTENT to answer QUESTION. "
        "No reasoning traces. "
        "If insufficient evidence, output exactly: I don't know.\n\n"
        "Return markdown sections:\n"
        "## Executive Summary\n"
        "## Key Facts\n"
        "## Risks / Limitations\n\n"
        f"CONTENT:\n{context}\n\n"
        f"QUESTION:\n{question}\n\n"
        "ANSWER:"
    )


def run_rag_pipeline(
    question: str,
    stream_callback: Callable[[str], None] | None = None,
) -> PipelineResult:
    """
    Execute the complete Retrieval-Augmented Generation pipeline.

    Steps:
        1. Retrieve top-k relevant chunks from the vector store.
        2. Format chunks into a context block.
        3. Inject context + question into the strict QA prompt.
        4. Call the Gemini LLM and extract the answer.
        5. Return (answer, deduplicated source list).

    Args:
        question: The user's natural-language question.

    Returns:
        A tuple of (answer_text, list_of_source_references).

    Raises:
        RuntimeError: Propagated from retriever if store is empty.
    """
    start_time = time.perf_counter()
    normalized_question = normalize_query(question)
    normalization_applied = normalized_question != question

    logger.info(
        "Query normalized | applied=%s raw='%s' normalized='%s'",
        normalization_applied,
        question[:120],
        normalized_question[:120],
    )

    kb_version = get_knowledge_base_version()
    query_only_cache_key = build_cache_key(question=normalized_question, kb_version=kb_version)

    if settings.enable_query_cache:
        cached_payload = get_cached_result(query_only_cache_key)
        if cached_payload is not None:
            logger.info("Cache hit (query-level) for question: '%s'", question[:80])
            return _payload_to_result(deepcopy(cached_payload))

    # ── Step 1: Retrieve ─────────────────────────────────────────────────────
    chunks, debug = retrieve_relevant_chunks_with_diagnostics(normalized_question)
    retrieval_ms = float(debug.retrieval_ms)

    if not chunks:
        raise RuntimeError(
            "No relevant content could be retrieved. Please upload additional documents."
        )

    # ── Step 2: Build context ────────────────────────────────────────────────
    context_build_start = time.perf_counter()
    chunks_for_generation = _select_chunks_for_context(
        chunks=chunks,
        question=normalized_question,
        is_simple_query=debug.is_simple_query,
    )

    context_cap = (
        max(500, int(settings.fast_mode_max_context_characters))
        if debug.fast_mode_applied
        else max(500, int(settings.max_context_characters))
    )
    context = _format_context(
        [item.document for item in chunks_for_generation],
        max_chars=context_cap,
    )

    context_build_ms = (time.perf_counter() - context_build_start) * 1000.0
    logger.info(
        "Context built | context_chars=%d chunks_used=%d simple_query=%s fast_mode=%s sample=%r",
        len(context),
        len(chunks_for_generation),
        debug.is_simple_query,
        debug.fast_mode_applied,
        context[:300],
    )

    # ── Step 3: Fill the prompt ──────────────────────────────────────────────
    trim_prompt = bool(debug.fast_mode_applied and settings.fast_mode_trim_prompt)
    filled_prompt = _build_generation_prompt(
        context=context,
        question=normalized_question,
        trim_prompt=trim_prompt,
    )
    top_k_ids = [_chunk_identity(item) for item in chunks_for_generation]
    prompt_fingerprint = build_prompt_fingerprint(filled_prompt)
    cache_key = build_cache_key(
        question=normalized_question,
        kb_version=kb_version,
        top_k_ids=top_k_ids,
        prompt_fingerprint=prompt_fingerprint,
    )

    if settings.enable_query_cache:
        cached_payload = get_cached_result(cache_key)
        if cached_payload is not None:
            logger.info(
                "Cache hit | question='%s' top_k_ids=%d prompt_fingerprint=%s",
                question[:80],
                len(top_k_ids),
                prompt_fingerprint[:10],
            )
            return _payload_to_result(deepcopy(cached_payload))

    prompt_build_ms = context_build_ms
    logger.info(
        "RAG timing | retrieval_ms=%.1f rerank_ms=%.1f context_build_ms=%.1f context_chars=%d",
        retrieval_ms,
        debug.rerank_ms,
        prompt_build_ms,
        len(context),
    )

    # ── Step 4: Call centralized generation flow ─────────────────────────────
    confidence_score = sum(item.final_score for item in chunks) / len(chunks)
    confidence_level = _confidence_level(confidence_score)

    generation_ms = 0.0
    llm_retry_count = 0
    llm_retry_reason = ""
    low_confidence_fallback_used = False
    top_chunk_score = max((item.final_score for item in chunks), default=0.0)

    hard_refusal = (
        confidence_score < float(settings.answer_hard_refusal_threshold)
        and top_chunk_score < float(settings.low_confidence_min_chunk_score)
    )

    if hard_refusal:
        answer = (
            "I do not have enough grounded evidence in the indexed documents to answer "
            "this confidently. Please refine your question or upload more relevant material."
        )
        model_used = "none-hard-refusal"
    else:
        generation_start = time.perf_counter()
        if settings.use_openai:
            openai_budget = max(1, int(settings.openai_max_tokens))
            if debug.fast_mode_applied:
                openai_budget = min(openai_budget, max(1, int(settings.fast_mode_llm_max_tokens)))

            try:
                if stream_callback is not None:
                    parts: list[str] = []
                    for text_chunk in stream_openai_response(
                        prompt=filled_prompt,
                        max_tokens=openai_budget,
                        temperature=float(settings.openai_temperature),
                    ):
                        parts.append(text_chunk)
                        stream_callback(text_chunk)
                    answer = "".join(parts).strip()
                else:
                    answer = generate_openai_response(
                        prompt=filled_prompt,
                        max_tokens=openai_budget,
                        temperature=float(settings.openai_temperature),
                    )
                model_used = f"openai:{settings.openai_model}"
                llm_retry_count = 0
                llm_retry_reason = ""
            except Exception as exc:
                logger.warning("OpenAI generation failed; falling back to local LLM. error=%s", exc)
                generation_token_budget = (
                    max(1, int(settings.fast_mode_llm_max_tokens))
                    if debug.fast_mode_applied
                    else max(1, int(settings.llm_max_tokens))
                )
                answer, local_model_used, llm_retry_count, llm_retry_reason = generate_answer(
                    filled_prompt,
                    max_tokens_override=generation_token_budget,
                )
                model_used = f"fallback:{local_model_used}"
        else:
            generation_token_budget = (
                max(1, int(settings.fast_mode_llm_max_tokens))
                if debug.fast_mode_applied
                else max(1, int(settings.llm_max_tokens))
            )
            answer, model_used, llm_retry_count, llm_retry_reason = generate_answer(
                filled_prompt,
                max_tokens_override=generation_token_budget,
            )
        generation_ms = (time.perf_counter() - generation_start) * 1000.0
        answer = _clean_answer_text(answer)
        if confidence_score < float(settings.answer_low_confidence_threshold):
            low_confidence_fallback_used = True
            answer = _apply_low_confidence_disclaimer(answer)
        logger.info("LLM path used=%s question='%s'", model_used, question[:80])
        logger.info(
            "RAG timing | generation_ms=%.1f llm_retry_count=%d llm_retry_reason=%s low_confidence_fallback=%s",
            generation_ms,
            llm_retry_count,
            llm_retry_reason,
            low_confidence_fallback_used,
        )
    if model_used.startswith("none"):
        logger.info("LLM path used=%s question='%s'", model_used, question[:80])

    # ── Step 5: Extract sources ───────────────────────────────────────────────
    sources = _extract_sources(chunks_for_generation)
    total_pipeline_ms = (time.perf_counter() - start_time) * 1000.0
    diagnostics = RetrievalDiagnostics(
        query_variants_used=debug.query_variants_used,
        is_broad_question=debug.is_broad_question,
        is_simple_query=debug.is_simple_query,
        fast_mode_applied=debug.fast_mode_applied,
        fallback_applied=debug.fallback_applied,
        candidates_considered=debug.candidates_considered,
        reranker_applied=debug.reranker_applied,
        reranker_skipped_reason=debug.reranker_skipped_reason,
        retrieval_ms=round(retrieval_ms, 2),
        rerank_ms=round(float(debug.rerank_ms), 2),
        context_build_ms=round(prompt_build_ms, 2),
        generation_ms=round(generation_ms, 2),
        total_pipeline_ms=round(total_pipeline_ms, 2),
        llm_retry_count=llm_retry_count,
        llm_retry_reason=llm_retry_reason,
        normalization_applied=normalization_applied,
        low_confidence_fallback_used=low_confidence_fallback_used,
    )

    logger.info(
        "RAG pipeline completed | total_pipeline_ms=%.1f retrieval_ms=%.1f rerank_ms=%.1f context_build_ms=%.1f generation_ms=%.1f simple_query=%s reranker_skip=%s low_confidence_fallback=%s sources=%s",
        total_pipeline_ms,
        retrieval_ms,
        debug.rerank_ms,
        prompt_build_ms,
        generation_ms,
        debug.is_simple_query,
        debug.reranker_skipped_reason,
        low_confidence_fallback_used,
        [s.document for s in sources],
    )

    result = PipelineResult(
        answer=answer,
        sources=sources,
        confidence_score=round(float(confidence_score), 4),
        confidence_level=confidence_level,
        diagnostics=diagnostics,
    )

    if settings.enable_query_cache:
        # Store both strict and query-level keys:
        # - strict key: question + kb + selected top-k ids + prompt fingerprint
        # - query key: question + kb for early short-circuit on repeated queries
        set_cached_result(
            cache_key=cache_key,
            payload=_result_to_payload(result),
            ttl_seconds=settings.query_cache_ttl_seconds,
        )
        set_cached_result(
            cache_key=query_only_cache_key,
            payload=_result_to_payload(result),
            ttl_seconds=settings.query_cache_ttl_seconds,
        )

    return result
