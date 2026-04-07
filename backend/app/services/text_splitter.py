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

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


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

    logger.info(
        f"Split {len(documents)} page(s) into {len(chunks)} chunk(s) "
        f"(size={settings.chunk_size}, overlap={settings.chunk_overlap})"
    )
    return chunks
