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
import unicodedata
from copy import deepcopy
from dataclasses import dataclass
from typing import List

from app.services.retriever import retrieve_relevant_chunks_with_diagnostics, RetrievedChunk
from app.services.llm_service import generate_answer
from app.services.query_cache import build_cache_key, get_cached_result, set_cached_result
from app.services.vector_store import get_knowledge_base_version
from app.prompts.qa_prompt import QA_PROMPT
from app.models.schemas import SourceReference, RetrievalDiagnostics
from app.config import settings
from app.utils.logger import get_logger

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


def _format_context(chunks) -> str:
    """
    Convert retrieved chunks into a cleaned, compact context block.

    Cleaning goals:
    - Remove OCR artefacts and encoding noise.
    - Repair broken line wraps and hyphenated word splits.
    - Keep only text content (no metadata labels).
    - Present chunks with explicit separators for local-model readability.
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

    def _clean_chunk_text(text: str) -> str:
        # Normalize line endings first.
        cleaned = text.replace("\r\n", "\n").replace("\r", "\n")

        # Remove common OCR replacement chars and normalize to ASCII.
        cleaned = cleaned.replace("\ufffd", " ").replace("�", " ")
        cleaned = unicodedata.normalize("NFKD", cleaned).encode("ascii", "ignore").decode("ascii")

        # Repair hard hyphenation and intra-word line breaks.
        cleaned = cleaned.replace("-\n", "")
        cleaned = re.sub(r"(?<=\w)\n(?=\w)", " ", cleaned)

        # Remove metadata labels if present in chunk text.
        cleaned = re.sub(r"\[Excerpt\s+\d+\s*\|\s*Document:.*?\]", "", cleaned)

        # Drop common low-signal lines that hurt summarization quality.
        noise_patterns = [
            r"^contact\b",
            r"investor relations",
            r"sustainability@",
            r"org\s*no\b",
            r"annual report\s*\d{4}",
            r"permission of",
            r"machine-readable medium",
            r"@\w+\.\w+",
        ]

        def _is_noise_line(line: str) -> bool:
            lowered = line.lower().strip()
            if len(lowered) < 3:
                return True
            return any(re.search(pattern, lowered, re.IGNORECASE) for pattern in noise_patterns)

        # Merge fragmented lines into paragraph-like sentences.
        lines = [line.strip() for line in cleaned.split("\n") if line.strip()]
        lines = [line for line in lines if not _is_noise_line(line)]

        # Keep a fact-dense subset first when chunk has many lines.
        def _line_score(line: str) -> int:
            has_number = 1 if re.search(r"\d", line) else 0
            length_score = min(len(line) // 40, 3)
            return has_number + length_score

        if len(lines) > 8:
            lines = sorted(lines, key=_line_score, reverse=True)[:8]
        merged: list[str] = []
        current = ""
        for line in lines:
            if not current:
                current = line
                continue

            if re.search(r"[.!?:;)]$", current):
                merged.append(current)
                current = line
            else:
                current = f"{current} {line}"

        if current:
            merged.append(current)

        cleaned = "\n".join(merged)

        # Final whitespace normalization.
        cleaned = re.sub(r"\n{2,}", "\n", cleaned)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        return cleaned.strip()

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
        cleaned_text = _clean_chunk_text(chunk.page_content)
        cleaned_text = _compress_chunk_text(cleaned_text, limit=450)
        if cleaned_text:
            sections.append(cleaned_text)

    if not sections:
        return ""

    context = "\n---\n".join(sections)
    max_chars = max(500, settings.max_context_characters)
    if len(context) > max_chars:
        return context[:max_chars].rstrip()
    return context


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


def run_rag_pipeline(question: str) -> PipelineResult:
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

    if settings.enable_query_cache:
        kb_version = get_knowledge_base_version()
        cache_key = build_cache_key(question=question, kb_version=kb_version)
        cached_payload = get_cached_result(cache_key)
        if cached_payload is not None:
            logger.info("Cache hit for question: '%s'", question[:80])
            return _payload_to_result(deepcopy(cached_payload))

    # ── Step 1: Retrieve ─────────────────────────────────────────────────────
    retrieval_start = time.perf_counter()
    chunks, debug = retrieve_relevant_chunks_with_diagnostics(question)
    retrieval_ms = (time.perf_counter() - retrieval_start) * 1000.0

    if not chunks:
        raise RuntimeError(
            "No relevant content could be retrieved. Please upload additional documents."
        )

    # ── Step 2: Build context ────────────────────────────────────────────────
    prompt_build_start = time.perf_counter()
    chunks_for_generation = chunks
    if _is_summary_request(question):
        chunks_for_generation = chunks[:3]

    context = _format_context([item.document for item in chunks_for_generation])
    logger.info(
        "Cleaned context length=%d chars | chunks_used=%d | sample=%r",
        len(context),
        len(chunks_for_generation),
        context[:300],
    )

    # ── Step 3: Fill the prompt ──────────────────────────────────────────────
    filled_prompt = QA_PROMPT.format(context=context, question=question)
    prompt_build_ms = (time.perf_counter() - prompt_build_start) * 1000.0
    logger.info(
        "RAG timing | retrieval_ms=%.1f prompt_build_ms=%.1f context_chars=%d",
        retrieval_ms,
        prompt_build_ms,
        len(context),
    )

    # ── Step 4: Call centralized generation flow ─────────────────────────────
    confidence_score = sum(item.final_score for item in chunks) / len(chunks)
    confidence_level = _confidence_level(confidence_score)

    if confidence_score < settings.answer_low_confidence_threshold:
        answer = (
            "I do not have enough grounded evidence in the indexed documents to answer "
            "this confidently. Please refine your question or upload more relevant material."
        )
        model_used = "none-low-confidence"
    else:
        generation_start = time.perf_counter()
        answer, model_used = generate_answer(filled_prompt)
        generation_ms = (time.perf_counter() - generation_start) * 1000.0
        answer = _clean_answer_text(answer)
        logger.info("LLM path used=%s question='%s'", model_used, question[:80])
        logger.info("RAG timing | generation_ms=%.1f", generation_ms)
    if model_used.startswith("none"):
        logger.info("LLM path used=%s question='%s'", model_used, question[:80])

    # ── Step 5: Extract sources ───────────────────────────────────────────────
    sources = _extract_sources(chunks_for_generation)
    diagnostics = RetrievalDiagnostics(
        query_variants_used=debug.query_variants_used,
        is_broad_question=debug.is_broad_question,
        fallback_applied=debug.fallback_applied,
        candidates_considered=debug.candidates_considered,
    )

    elapsed = time.perf_counter() - start_time
    logger.info(f"RAG pipeline completed in {elapsed:.2f}s | sources: {[s.document for s in sources]}")

    result = PipelineResult(
        answer=answer,
        sources=sources,
        confidence_score=round(float(confidence_score), 4),
        confidence_level=confidence_level,
        diagnostics=diagnostics,
    )

    if settings.enable_query_cache:
        set_cached_result(
            cache_key=cache_key,
            payload=_result_to_payload(result),
            ttl_seconds=settings.query_cache_ttl_seconds,
        )

    return result
