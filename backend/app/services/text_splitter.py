"""
text_splitter.py
────────────────
Responsibility: Split a list of Documents into smaller, overlapping chunks.

Why chunking matters:
- LLMs have context limits; feeding entire PDFs would exceed them.
- Smaller chunks give the retriever a finer-grained signal.
- Overlap (100 tokens) prevents key sentences from being cut at boundaries.
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List
import re
import unicodedata

from app.config import settings
from app.services.ingestion.metadata_enricher import enrich_chunk_metadata
from app.utils.logger import get_logger

logger = get_logger(__name__)


def chunk_structured_blocks(blocks: List[dict]) -> List[Document]:
    """Build structure-aware chunks from parsed blocks.

    This keeps heading context, carries visual descriptions inline,
    and preserves source block ids for traceability.
    """
    if not blocks:
        return []

    ordered = sorted(
        blocks,
        key=lambda b: (
            int(b.get("page", 0) or 0),
            str(b.get("id", "")),
        ),
    )

    max_chars = max(200, int(settings.chunk_size))
    chunks: list[Document] = []
    section = "General"

    buffer_lines: list[str] = []
    buffer_ids: list[str] = []
    buffer_block_types: set[str] = set()
    buffer_has_visual = False
    buffer_page: int = int(ordered[0].get("page", 1) or 1)
    buffer_meta: dict = {}

    def _flush_buffer() -> None:
        nonlocal buffer_lines, buffer_ids, buffer_block_types, buffer_has_visual, buffer_page, buffer_meta
        if not buffer_lines:
            return

        text = "\n\n".join(line for line in buffer_lines if line.strip()).strip()
        text = _clean_text_for_indexing(text)
        if not text or not _has_enough_signal(text):
            buffer_lines = []
            buffer_ids = []
            buffer_block_types = set()
            buffer_has_visual = False
            return

        metadata = {
            "source": buffer_meta.get("source", "unknown"),
            "page": int(buffer_page or 1),
            "section": section,
            "block_type": next(iter(buffer_block_types)) if len(buffer_block_types) == 1 else "mixed",
            "has_visual": bool(buffer_has_visual),
            "source_block_ids": list(buffer_ids),
            "file_hash": buffer_meta.get("file_hash", ""),
            "document_id": buffer_meta.get("document_id", ""),
            "uploaded_at": buffer_meta.get("uploaded_at", ""),
        }
        chunks.append(Document(page_content=text, metadata=metadata))

        buffer_lines = []
        buffer_ids = []
        buffer_block_types = set()
        buffer_has_visual = False

    for block in ordered:
        block_type = str(block.get("type", "paragraph")).lower().strip()
        block_content = str(block.get("content", "")).strip()
        block_visual = str(block.get("visual_description", "")).strip()
        block_page = int(block.get("page", buffer_page) or buffer_page)

        if block_type == "heading" and block_content:
            _flush_buffer()
            section = block_content
            buffer_page = block_page
            buffer_meta = block
            continue

        parts: list[str] = []
        if block_visual:
            parts.append(block_visual)
            buffer_has_visual = True
        if block_content:
            parts.append(block_content)
        if not parts:
            continue

        block_text = "\n\n".join(parts).strip()
        if not block_text:
            continue

        candidate_text = "\n\n".join(buffer_lines + [block_text]).strip()
        if buffer_lines and len(candidate_text) > max_chars:
            _flush_buffer()
            buffer_page = block_page
            buffer_meta = block

        if not buffer_lines:
            buffer_page = block_page
            buffer_meta = block

        prefix = f"Section: {section}\n\n" if not buffer_lines else ""
        buffer_lines.append((prefix + block_text).strip())
        block_id = str(block.get("id", "")).strip()
        if block_id:
            buffer_ids.append(block_id)
        buffer_block_types.add(block_type or "paragraph")

        if bool(block.get("has_visual", False)):
            buffer_has_visual = True

    _flush_buffer()
    if settings.enable_metadata_enrichment:
        return enrich_chunk_metadata(chunks)
    return chunks


def _clean_text_for_indexing(text: str) -> str:
    """Normalize OCR artifacts and trim noisy boilerplate during ingestion."""
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = cleaned.replace("\ufffd", " ").replace("�", " ")
    cleaned = unicodedata.normalize("NFKD", cleaned).encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"(?<=\b[A-Za-z])\s+(?=[A-Za-z]\b)", "", cleaned)
    cleaned = cleaned.replace("-\n", "")
    cleaned = re.sub(r"(?<=\w)\n(?=\w)", " ", cleaned)
    cleaned = re.sub(r"(?<=\w)\s{2,}(?=\w)", " ", cleaned)
    cleaned = re.sub(r"([A-Za-z])\1{4,}", r"\1\1", cleaned)

    noise_patterns = [
        r"^contact\b",
        r"investor relations",
        r"permission of",
        r"machine-readable medium",
        r"@\w+\.\w+",
    ]

    lines = [line.strip() for line in cleaned.split("\n") if line.strip()]
    kept: list[str] = []
    for line in lines:
        lowered = line.lower()
        if any(re.search(pattern, lowered, re.IGNORECASE) for pattern in noise_patterns):
            continue
        kept.append(line)

    cleaned = "\n".join(kept)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def _has_enough_signal(text: str) -> bool:
    """Drop OCR fragments that contain almost no searchable content."""
    alnum = re.sub(r"[^a-zA-Z0-9]", "", text or "")
    if len(alnum) < 12:
        return False
    letters = re.sub(r"[^a-zA-Z]", "", text or "")
    return len(letters) >= 8


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split a list of Documents into overlapping text chunks.

    The splitter tries to break on paragraphs → sentences → words in that
    order, preserving natural semantic boundaries where possible.

    Args:
        documents: Raw Documents from the PDF loader (one per page).

    Returns:
        A flat list of smaller Document chunks, each inheriting the
        original metadata (source filename, page number).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,           # target character count per chunk
        chunk_overlap=settings.chunk_overlap,     # overlap to avoid cut-off context
        separators=["\n\n", "\n", ". ", " ", ""], # ordered from coarsest to finest
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    # Preserve layout-derived context so retrieval can target the right section/table.
    for chunk in chunks:
        chunk.page_content = _clean_text_for_indexing(chunk.page_content)
        if not chunk.page_content or not _has_enough_signal(chunk.page_content):
            continue

        header_hints = chunk.metadata.get("header_hints") or []
        if isinstance(header_hints, list) and header_hints:
            section_hint = " | ".join(str(h).strip() for h in header_hints[:2] if str(h).strip())
            if section_hint:
                chunk.metadata["section_hint"] = section_hint
                chunk.page_content = f"[SECTION_HINT] {section_hint}\n{chunk.page_content}"

        has_tables = bool(chunk.metadata.get("has_tables", False))
        chunk.metadata["chunk_has_tables"] = has_tables

    chunks = [chunk for chunk in chunks if chunk.page_content.strip() and _has_enough_signal(chunk.page_content)]

    if settings.enable_metadata_enrichment:
        chunks = enrich_chunk_metadata(chunks)

    logger.info(
        f"Split {len(documents)} page(s) into {len(chunks)} chunk(s) "
        f"(size={settings.chunk_size}, overlap={settings.chunk_overlap})"
    )
    return chunks
