"""
query.py
────────
POST /query

Accepts a user question, runs the RAG pipeline, and returns a structured
response containing the LLM-generated answer and source references.
"""

import time
from fastapi import APIRouter, HTTPException, status

from app.services.rag_pipeline import run_rag_pipeline
from app.models.schemas import QueryRequest, QueryResponse
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask a question to the knowledge base",
    description=(
        "Retrieves the most relevant document chunks for the given question "
        "and uses an LLM to generate a grounded answer with source citations."
    ),
)
async def query_knowledge_base(request: QueryRequest) -> QueryResponse:
    """Answer a question using the RAG pipeline."""

    start_time = time.perf_counter()
    logger.info(f"Query received: '{request.question[:100]}'")

    try:
        answer, sources = run_rag_pipeline(request.question)
    except RuntimeError as e:
        message = str(e)
        if "quota" in message.lower() or "billing" in message.lower():
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=message,
            )
        # Expected error when no documents have been indexed yet
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message,
        )
    except Exception as e:
        logger.error(f"RAG pipeline error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your question. Please try again.",
        )

    elapsed = time.perf_counter() - start_time
    logger.info(f"Query answered in {elapsed:.2f}s | sources: {[s.document for s in sources]}")

    return QueryResponse(answer=answer, sources=sources)
