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
from typing import Tuple, List

from app.services.retriever import retrieve_relevant_chunks
from app.services.llm_service import generate_answer
from app.prompts.qa_prompt import QA_PROMPT
from app.models.schemas import SourceReference
from app.utils.logger import get_logger

logger = get_logger(__name__)


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
    Convert a list of Document chunks into a single context string that
    the prompt can consume.  Each chunk is labelled with its source.
    """
    sections = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", "?")
        sections.append(
            f"[Excerpt {i} | Document: {source} | Page: {page}]\n{chunk.page_content}"
        )
    return "\n\n".join(sections)


def _extract_sources(chunks) -> List[SourceReference]:
    """
    Deduplicate the source references from the retrieved chunks.
    A (document, page) pair is unique by definition.
    """
    seen = set()
    sources = []
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", 0)
        key = (source, page)
        if key not in seen:
            seen.add(key)
            sources.append(SourceReference(document=source, page=page))
    return sources


def run_rag_pipeline(question: str) -> Tuple[str, List[SourceReference]]:
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

    # ── Step 1: Retrieve ─────────────────────────────────────────────────────
    chunks = retrieve_relevant_chunks(question)

    # ── Step 2: Build context ────────────────────────────────────────────────
    context = _format_context(chunks)

    # ── Step 3: Fill the prompt ──────────────────────────────────────────────
    filled_prompt = QA_PROMPT.format(context=context, question=question)

    # ── Step 4: Call the LLM ─────────────────────────────────────────────────
    answer, model_used = generate_answer(filled_prompt)
    answer = _clean_answer_text(answer)
    logger.info("LLM answered using model '%s' for question: '%s'", model_used, question[:80])

    # ── Step 5: Extract sources ───────────────────────────────────────────────
    sources = _extract_sources(chunks)

    elapsed = time.perf_counter() - start_time
    logger.info(f"RAG pipeline completed in {elapsed:.2f}s | sources: {[s.document for s in sources]}")

    return answer, sources
