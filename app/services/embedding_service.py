"""
embedding_service.py
────────────────────
Responsibility: Provide a configured Google Gemini embeddings object.

By wrapping the LangChain embeddings in its own module we:
1. Keep auth / model selection in one place.
2. Make it trivial to swap to a different embedding provider later
   (e.g. sentence-transformers, Cohere) without touching the rest of the app.
"""

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def get_embedding_model() -> GoogleGenerativeAIEmbeddings:
    """
    Build and return a configured GoogleGenerativeAIEmbeddings instance.

    The model is set via config (defaulting to models/gemini-embedding-004),
    Google's latest production embedding model.

    Returns:
        A ready-to-use LangChain GoogleGenerativeAIEmbeddings object.
    """
    logger.info(f"Initialising embedding model: {settings.embedding_model}")

    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.google_api_key,
    )
