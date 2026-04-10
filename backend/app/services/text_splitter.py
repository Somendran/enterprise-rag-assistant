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
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _clean_text_for_indexing(text: str) -> str:
    """Normalize OCR artifacts and trim noisy boilerplate during ingestion."""
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = cleaned.replace("\ufffd", " ").replace("�", " ")
    cleaned = unicodedata.normalize("NFKD", cleaned).encode("ascii", "ignore").decode("ascii")
    cleaned = cleaned.replace("-\n", "")
    cleaned = re.sub(r"(?<=\w)\n(?=\w)", " ", cleaned)

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
        if not chunk.page_content:
            continue

        header_hints = chunk.metadata.get("header_hints") or []
        if isinstance(header_hints, list) and header_hints:
            section_hint = " | ".join(str(h).strip() for h in header_hints[:2] if str(h).strip())
            if section_hint:
                chunk.metadata["section_hint"] = section_hint
                chunk.page_content = f"[SECTION_HINT] {section_hint}\n{chunk.page_content}"

        has_tables = bool(chunk.metadata.get("has_tables", False))
        chunk.metadata["chunk_has_tables"] = has_tables

    chunks = [chunk for chunk in chunks if chunk.page_content.strip()]

    logger.info(
        f"Split {len(documents)} page(s) into {len(chunks)} chunk(s) "
        f"(size={settings.chunk_size}, overlap={settings.chunk_overlap})"
    )
    return chunks
