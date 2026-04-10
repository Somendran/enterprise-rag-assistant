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
from typing import Any, Callable, Dict, List, Optional

from app.services.retriever import retrieve_relevant_chunks_with_diagnostics, RetrievedChunk
from app.services.llm_service import generate_answer
from app.services.embedding_service import get_embedding_model
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

FILLER_PREFIXES = (
    "based on the provided",
    "according to the provided",
    "from the provided",
    "it appears",
    "in summary",
    "overall",
)


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


def _dedupe_answer_bullets(answer: str) -> str:
    """Remove exact duplicate bullets across sections while preserving order."""
    seen: set[str] = set()
    output: list[str] = []
    for line in answer.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            key = re.sub(r"\s+", " ", stripped[2:].lower()).strip()
            if key in seen:
                continue
            seen.add(key)
        output.append(line)
    return "\n".join(output).strip()


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
    source_legend: list[str] = []
    for idx, chunk in enumerate(ordered_chunks, start=1):
        cleaned_text = _trim_low_signal_lines(chunk.page_content.strip())
        cleaned_text = _compress_chunk_text(cleaned_text, limit=450)
        if cleaned_text:
            source = str(chunk.metadata.get("source", "unknown"))
            page = int(chunk.metadata.get("page", 0) or 0)
            source_tag = f"[Source {idx}]"
            source_legend.append(f"- {source_tag} = {source}, p.{page}")
            sections.append(f"{source_tag}\n{cleaned_text}")

    if not sections:
        return ""

    separator = "\n---\n"
    budget = max(500, int(max_chars))
    selected_sections: list[str] = []
    selected_legend: list[str] = []
    used = 0
    for section, legend in zip(sections, source_legend):
        section_len = len(section)
        extra = section_len + (len(separator) if selected_sections else 0)
        if selected_sections and (used + extra) > budget:
            break
        if not selected_sections and section_len > budget:
            selected_sections.append(section[:budget].rstrip())
            selected_legend.append(legend)
            break
        selected_sections.append(section)
        selected_legend.append(legend)
        used += extra

    if not selected_sections:
        return ""

    legend_block = "Source Reference Guide:\n" + "\n".join(selected_legend)
    content_block = separator.join(selected_sections).rstrip()
    return f"{legend_block}\n\nCONTENT EXCERPTS:\n{content_block}".strip()


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


def _base_retrieval_signal(chunks: list[RetrievedChunk]) -> float:
    """Compute a stable retrieval signal from non-reranker components.

    This prevents hard-refusal from depending only on reranker logits,
    which can be very low for broad summary-style questions.
    """
    if not chunks:
        return 0.0

    values = [
        max(
            float(chunk.vector_confidence),
            float(chunk.lexical_score),
            float(chunk.bm25_score),
        )
        for chunk in chunks
    ]
    return sum(values) / len(values)


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
        "Do not repeat information across sections. "
        "If insufficient evidence, output exactly: I don't know.\n\n"
        "Return markdown sections:\n"
        "## Short Answer\n"
        "## Key Facts\n"
        "## Missing Information\n"
        "## Optional Notes\n"
        "## Confidence Explanation\n\n"
        "For Key Facts, add source tags like [Source 1] using only source tags provided in CONTENT.\n\n"
        f"CONTENT:\n{context}\n\n"
        f"QUESTION:\n{question}\n\n"
        "ANSWER:"
    )


def _append_confidence_explanation(
    answer: str,
    confidence_score: float,
    used_chunks: int,
    low_confidence_fallback_used: bool,
) -> str:
    label = _confidence_level(confidence_score).capitalize()
    pct = int(max(0.0, min(1.0, confidence_score)) * 100)

    reasons: list[str] = [f"Based on {used_chunks} retrieved section(s)."]
    if low_confidence_fallback_used:
        reasons.append("Some details may be incomplete due to partial evidence.")
    elif confidence_score >= 0.65:
        reasons.append("Evidence is consistent across the selected context.")
    else:
        reasons.append("Evidence coverage is moderate and may miss edge details.")

    explanation = (
        f"\n\nConfidence: {label} ({pct}%)\n"
        f"Reason: {reasons[0]} {reasons[1]}"
    )

    lowered = answer.lower()
    if "confidence:" in lowered:
        return answer
    return (answer + explanation).strip()


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _source_sort_key(chunk: RetrievedChunk) -> tuple[str, int]:
    return (
        str(chunk.document.metadata.get("source", "unknown")),
        int(chunk.document.metadata.get("page", 0) or 0),
    )


def _ordered_source_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    return sorted(chunks, key=_source_sort_key)


def _source_tag_for_index(idx: int) -> str:
    return f"Source {idx + 1}"


def extract_claims(answer: str) -> List[str]:
    """Extract lightweight atomic claims from answer text without extra LLM calls."""
    if not answer or not answer.strip():
        return []

    text = re.sub(r"```[\s\S]*?```", " ", answer)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    claim_candidates: list[str] = []
    for line in lines:
        lowered = line.lower().strip(":")
        if lowered in {
            "short answer",
            "key facts",
            "missing information",
            "optional notes",
            "confidence explanation",
        }:
            continue
        if lowered.startswith("confidence:") or lowered.startswith("reason:"):
            continue

        normalized = re.sub(r"^[-*]\s+", "", line)
        sentences = re.split(r"(?<=[.!?])\s+", normalized)
        claim_candidates.extend(s.strip() for s in sentences if s.strip())

    claims: list[str] = []
    seen: set[str] = set()
    for raw_claim in claim_candidates:
        claim = re.sub(r"\s+", " ", raw_claim).strip()
        if len(claim) < 20:
            continue
        lowered = claim.lower()
        if lowered.startswith(FILLER_PREFIXES):
            continue
        if lowered in {"i don't know", "insufficient information"}:
            continue

        key = re.sub(r"\s+", " ", lowered)
        if key in seen:
            continue
        seen.add(key)
        claims.append(claim)

    return claims


def _dot_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    size = min(len(a), len(b))
    return float(sum(a[i] * b[i] for i in range(size)))


def verify_claims(claims: List[str], context_chunks: List[RetrievedChunk]) -> List[Dict[str, Any]]:
    """Verify claims by embedding-similarity against retrieved context chunks."""
    if not claims or not context_chunks:
        return []

    chunk_texts: list[str] = []
    chunk_ids: list[str] = []
    for idx, chunk in enumerate(context_chunks):
        text = re.sub(r"\s+", " ", chunk.document.page_content).strip()
        chunk_texts.append(text[:800])
        chunk_ids.append(_source_tag_for_index(idx))

    embedding_model = get_embedding_model()
    batched_inputs = claims + chunk_texts
    vectors = embedding_model.embed_documents(batched_inputs)
    claim_vectors = vectors[: len(claims)]
    chunk_vectors = vectors[len(claims):]

    threshold = float(settings.verification_similarity_threshold)
    verified: list[dict[str, Any]] = []
    for claim, claim_vec in zip(claims, claim_vectors):
        best_idx = -1
        best_score = -1.0
        for idx, chunk_vec in enumerate(chunk_vectors):
            score = _dot_similarity(claim_vec, chunk_vec)
            if score > best_score:
                best_score = score
                best_idx = idx

        matched_chunk_id: Optional[str] = chunk_ids[best_idx] if best_idx >= 0 else None
        verified.append(
            {
                "claim": claim,
                "supported": best_score >= threshold,
                "score": round(_clamp01(best_score), 4),
                "matched_chunk_id": matched_chunk_id,
            }
        )

    return verified


def _extract_source_citations(answer: str) -> list[str]:
    matches = re.findall(r"\[\s*Source\s+(\d+)\s*\]", answer or "", flags=re.IGNORECASE)
    deduped: list[str] = []
    seen: set[str] = set()
    for raw in matches:
        tag = f"Source {int(raw)}"
        if tag in seen:
            continue
        seen.add(tag)
        deduped.append(tag)
    return deduped


def validate_citations(
    answer: str,
    verified_claims: List[Dict[str, Any]],
    valid_chunk_ids: Optional[set[str]] = None,
) -> Dict[str, Any]:
    """Validate citation tags and compute supported-claim coverage."""
    citations = _extract_source_citations(answer)
    claims_total = len(verified_claims)
    claims_supported = sum(1 for item in verified_claims if bool(item.get("supported")))
    coverage = float(claims_supported / claims_total) if claims_total else 0.0

    if valid_chunk_ids is None:
        valid_chunk_ids = {str(item.get("matched_chunk_id")) for item in verified_claims if item.get("matched_chunk_id")}

    supported_citation_ids = {
        str(item.get("matched_chunk_id"))
        for item in verified_claims
        if bool(item.get("supported")) and item.get("matched_chunk_id")
    }

    orphan_or_invalid = [citation for citation in citations if citation not in valid_chunk_ids]
    no_support = [citation for citation in citations if citation in valid_chunk_ids and citation not in supported_citation_ids]
    invalid_citations = sorted(set(orphan_or_invalid + no_support))

    unsupported_claims = [
        str(item.get("claim", "")).strip()
        for item in verified_claims
        if (not bool(item.get("supported"))) and _extract_source_citations(str(item.get("claim", "")))
    ]

    return {
        "invalid_citations": invalid_citations,
        "unsupported_claims": unsupported_claims,
        "coverage": round(_clamp01(coverage), 4),
    }


def recompute_confidence(
    claim_support_ratio: float,
    citation_coverage: float,
    retrieval_score: float,
) -> tuple[float, str]:
    """Blend verification signals with retrieval quality into final confidence."""
    blended = (
        0.5 * _clamp01(claim_support_ratio)
        + 0.3 * _clamp01(citation_coverage)
        + 0.2 * _clamp01(retrieval_score)
    )
    score = round(_clamp01(blended), 4)
    return score, _confidence_level(score)


def _append_verification_warning(answer: str) -> str:
    warning = "\n\nSome parts of this answer may not be fully supported by the retrieved documents."
    if warning.strip() in answer:
        return answer
    return (answer + warning).strip()


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
    rerank_score = sum(item.final_score for item in chunks) / len(chunks)
    base_signal = _base_retrieval_signal(chunks)
    retrieval_score = (0.7 * rerank_score) + (0.3 * base_signal)
    confidence_score = retrieval_score
    confidence_level = _confidence_level(confidence_score)

    generation_ms = 0.0
    verification_ms = 0.0
    llm_retry_count = 0
    llm_retry_reason = ""
    low_confidence_fallback_used = False
    verification_enabled = bool(settings.enable_verification)
    verification_applied = False
    verification_skipped_reason = ""
    verification_failed = False
    claims_total = 0
    claims_verified = 0
    citation_coverage = 0.0
    invalid_citations: list[str] = []
    unsupported_claims_count = 0
    top_chunk_score = max((item.final_score for item in chunks), default=0.0)
    top_base_signal = max(
        (
            max(
                float(item.vector_confidence),
                float(item.lexical_score),
                float(item.bm25_score),
            )
            for item in chunks
        ),
        default=0.0,
    )
    summary_request = _is_summary_request(normalized_question)

    hard_refusal = (
        confidence_score < float(settings.answer_hard_refusal_threshold)
        and top_chunk_score < float(settings.low_confidence_min_chunk_score)
        and top_base_signal < float(settings.low_confidence_min_chunk_score)
        and (not summary_request)
    )

    logger.info(
        "Grounding gate | rerank_score=%.4f base_signal=%.4f blended_score=%.4f top_rerank=%.4f top_base=%.4f summary_request=%s hard_refusal=%s",
        rerank_score,
        base_signal,
        retrieval_score,
        top_chunk_score,
        top_base_signal,
        summary_request,
        hard_refusal,
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
        answer = _dedupe_answer_bullets(answer)

        should_verify = (
            verification_enabled
            and (not debug.fast_mode_applied)
            and len(chunks_for_generation) >= 2
            and len(answer.strip()) >= int(settings.verification_min_answer_chars)
        )

        if should_verify:
            verify_start = time.perf_counter()
            try:
                ordered_chunks = _ordered_source_chunks(chunks_for_generation)
                claims = extract_claims(answer)
                claims_total = len(claims)
                valid_chunk_ids = {
                    _source_tag_for_index(idx) for idx in range(len(ordered_chunks))
                }
                verified_claims = verify_claims(claims, ordered_chunks)
                claims_verified = sum(1 for item in verified_claims if bool(item.get("supported")))
                claim_support_ratio = (
                    float(claims_verified / claims_total) if claims_total else 0.0
                )

                citation_report = validate_citations(
                    answer=answer,
                    verified_claims=verified_claims,
                    valid_chunk_ids=valid_chunk_ids,
                )
                citation_coverage = float(citation_report.get("coverage", 0.0))
                invalid_citations = [str(c) for c in citation_report.get("invalid_citations", [])]
                unsupported_claims_count = len(citation_report.get("unsupported_claims", []))

                confidence_score, confidence_level = recompute_confidence(
                    claim_support_ratio=claim_support_ratio,
                    citation_coverage=citation_coverage,
                    retrieval_score=retrieval_score,
                )
                verification_applied = True

                if claim_support_ratio < float(settings.verification_warning_support_threshold):
                    answer = _append_verification_warning(answer)
            except Exception as exc:
                verification_failed = True
                verification_skipped_reason = "verification_failed"
                confidence_score = min(confidence_score, 0.2)
                confidence_level = "low"
                logger.warning("Verification layer failed; continuing without blocking response. error=%s", exc)
            finally:
                verification_ms = (time.perf_counter() - verify_start) * 1000.0
        else:
            if not verification_enabled:
                verification_skipped_reason = "disabled"
            elif debug.fast_mode_applied:
                verification_skipped_reason = "fast_mode"
            elif len(chunks_for_generation) < 2:
                verification_skipped_reason = "insufficient_chunks"
            else:
                verification_skipped_reason = "answer_too_short"

        if confidence_score < float(settings.answer_low_confidence_threshold):
            low_confidence_fallback_used = True
            answer = _apply_low_confidence_disclaimer(answer)
        answer = _append_confidence_explanation(
            answer=answer,
            confidence_score=confidence_score,
            used_chunks=len(chunks_for_generation),
            low_confidence_fallback_used=low_confidence_fallback_used,
        )
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
        verification_enabled=verification_enabled,
        verification_applied=verification_applied,
        verification_skipped_reason=verification_skipped_reason,
        verification_failed=verification_failed,
        verification_ms=round(verification_ms, 2),
        claims_total=claims_total,
        claims_verified=claims_verified,
        citation_coverage=round(float(citation_coverage), 4),
        invalid_citations=invalid_citations,
        unsupported_claims=unsupported_claims_count,
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
