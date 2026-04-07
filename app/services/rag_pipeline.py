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
from typing import Tuple, List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage

from app.services.retriever import retrieve_relevant_chunks
from app.prompts.qa_prompt import QA_PROMPT
from app.models.schemas import SourceReference
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


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
    llm = ChatGoogleGenerativeAI(
        model=settings.llm_model,
        google_api_key=settings.google_api_key,
        temperature=0,  # deterministic; important for a factual assistant
    )

    logger.info(f"Calling LLM ({settings.llm_model}) for question: '{question[:80]}'")
    response = llm.invoke([HumanMessage(content=filled_prompt)])
    answer = response.content.strip()

    # ── Step 5: Extract sources ───────────────────────────────────────────────
    sources = _extract_sources(chunks)

    elapsed = time.perf_counter() - start_time
    logger.info(f"RAG pipeline completed in {elapsed:.2f}s | sources: {[s.document for s in sources]}")

    return answer, sources
