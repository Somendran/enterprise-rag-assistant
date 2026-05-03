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

    Docling has already recovered paragraph, heading, list, and table
    boundaries, so use those boundaries directly and only split individual
    oversized elements.
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
    section_title: str | None = None
    pending_short: dict | None = None

    def _base_metadata(block: dict, element_type: str, source_ids: list[str]) -> dict:
        return {
            "source": block.get("source", "unknown"),
            "filename": block.get("source", "unknown"),
            "page": int(block.get("page", 0) or 0),
            "section": section_title or "",
            "section_title": section_title,
            "element_type": element_type,
            "block_type": element_type,
            "source_block_ids": source_ids,
            "file_hash": block.get("file_hash", ""),
            "document_id": block.get("document_id", ""),
            "doc_id": block.get("document_id", ""),
            "uploaded_at": block.get("uploaded_at", ""),
            "has_visual": bool(block.get("has_visual", False)),
        }

    def _append_text_chunks(text: str, block: dict, element_type: str, source_ids: list[str]) -> None:
        cleaned = _clean_text_for_indexing(text)
        if not cleaned:
            return

        metadata = _base_metadata(block, element_type, source_ids)
        if len(cleaned) <= max_chars:
            chunks.append(Document(page_content=cleaned, metadata=metadata))
            return

        splitter = _recursive_splitter()
        for part in splitter.split_text(cleaned):
            part = _clean_text_for_indexing(part)
            if part:
                chunks.append(Document(page_content=part, metadata=dict(metadata)))

    def _append_table_chunks(text: str, block: dict, source_ids: list[str]) -> None:
        cleaned = _clean_text_for_indexing(text)
        if not cleaned:
            return

        metadata = _base_metadata(block, "table", source_ids)
        table_limit = max_chars * 2
        if len(cleaned) <= table_limit:
            chunks.append(Document(page_content=cleaned, metadata=metadata))
            return

        row_buffer: list[str] = []
        for row in cleaned.splitlines():
            candidate = "\n".join(row_buffer + [row]).strip()
            if row_buffer and len(candidate) > table_limit:
                chunks.append(Document(page_content="\n".join(row_buffer).strip(), metadata=dict(metadata)))
                row_buffer = [row]
            else:
                row_buffer.append(row)
        if row_buffer:
            chunks.append(Document(page_content="\n".join(row_buffer).strip(), metadata=dict(metadata)))

    def _emit_pending_short() -> None:
        nonlocal pending_short
        if pending_short is None:
            return
        _append_text_chunks(
            str(pending_short.get("content", "")),
            pending_short,
            str(pending_short.get("type", "paragraph")),
            [str(pending_short.get("id", "")).strip()] if str(pending_short.get("id", "")).strip() else [],
        )
        pending_short = None

    for block in ordered:
        block_type = str(block.get("type", "paragraph")).lower().strip() or "paragraph"
        if block_type not in {"heading", "paragraph", "table", "list"}:
            block_type = "paragraph"

        block_content = str(block.get("content", "")).strip()
        block_visual = str(block.get("visual_description", "")).strip()
        parts = [part for part in [block_visual, block_content] if part]
        block_text = "\n\n".join(parts).strip()
        if not block_text:
            continue

        block_id = str(block.get("id", "")).strip()
        source_ids = [block_id] if block_id else []

        if block_type == "heading":
            _emit_pending_short()
            section_title = block_content or section_title
            pending_short = {**block, "content": block_text, "type": "heading"}
            continue

        if block_type == "table":
            _emit_pending_short()
            _append_table_chunks(block_text, block, source_ids)
            continue

        if pending_short is not None:
            pending_id = str(pending_short.get("id", "")).strip()
            merged = f"{pending_short.get('content', '')}\n\n{block_text}".strip()
            merged_ids = ([pending_id] if pending_id else []) + source_ids
            _append_text_chunks(merged, block, block_type, merged_ids)
            pending_short = None
            continue

        if len(block_text) < 100:
            pending_short = {**block, "content": block_text, "type": block_type}
            continue

        _append_text_chunks(block_text, block, block_type, source_ids)

    _emit_pending_short()
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


def _recursive_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )


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
    splitter = _recursive_splitter()
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
        chunk.metadata.setdefault("filename", chunk.metadata.get("source", "unknown"))
        chunk.metadata.setdefault("doc_id", chunk.metadata.get("document_id", ""))
        chunk.metadata.setdefault("section_title", chunk.metadata.get("section_hint") or None)
        chunk.metadata.setdefault("element_type", "page")

    chunks = [chunk for chunk in chunks if chunk.page_content.strip() and _has_enough_signal(chunk.page_content)]

    if settings.enable_metadata_enrichment:
        chunks = enrich_chunk_metadata(chunks)

    logger.info(
        f"Split {len(documents)} page(s) into {len(chunks)} chunk(s) "
        f"(size={settings.chunk_size}, overlap={settings.chunk_overlap})"
    )
    return chunks
