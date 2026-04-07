"""
retriever.py
────────────
Responsibility: Retrieve the top-k most semantically similar chunks
from the vector store for a given user question.

Keeping retrieval logic separate from the RAG pipeline makes it easy
to swap retrieval strategies (e.g. MMR, hybrid BM25 + vector) later.
"""

from typing import List

from langchain.schema import Document

from app.services.vector_store import get_or_create_store
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def retrieve_relevant_chunks(question: str) -> List[Document]:
    """
    Perform a similarity search and return the top-k matching chunks.

    Args:
        question: The raw user question string.

    Returns:
        A list of Document objects (chunks) with their original metadata.
        Returns an empty list if the vector store is empty.

    Raises:
        RuntimeError: If the vector store has not been initialised yet
                      (i.e., no documents have been uploaded).
    """
    store = get_or_create_store()

    if store is None:
        # No documents have been uploaded yet — inform the caller gracefully
        raise RuntimeError(
            "The knowledge base is empty. Please upload at least one PDF document first."
        )

    k = settings.retrieval_top_k
    logger.info(f"Retrieving top-{k} chunks for question: '{question[:80]}...'")

    # similarity_search returns Documents ranked by cosine similarity
    chunks = store.similarity_search(question, k=k)

    logger.info(f"Retrieved {len(chunks)} chunk(s).")
    return chunks
