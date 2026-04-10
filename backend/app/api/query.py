"""
query.py
────────
POST /query

Accepts a user question, runs the RAG pipeline, and returns a structured
response containing the LLM-generated answer and source references.
"""

import time
import json
import queue
import threading
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from app.services.rag_pipeline import run_rag_pipeline
from app.models.schemas import QueryRequest, QueryResponse
from app.config import settings
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
        result = run_rag_pipeline(request.question)
    except RuntimeError as e:
        message = str(e)
        lowered = message.lower()
        if (
            "quota" in lowered
            or "billing" in lowered
            or "rate limit" in lowered
            or "retry after" in lowered
            or "too many requests" in lowered
        ):
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
    logger.info(
        "Query answered in %.2fs | confidence=%.3f level=%s | sources=%s",
        elapsed,
        result.confidence_score,
        result.confidence_level,
        [s.document for s in result.sources],
    )

    return QueryResponse(
        answer=result.answer,
        sources=result.sources,
        confidence_score=result.confidence_score,
        confidence_level=result.confidence_level,
        diagnostics=result.diagnostics if settings.enable_retrieval_diagnostics else None,
    )


@router.post(
    "/query/stream",
    status_code=status.HTTP_200_OK,
    summary="Ask a question with streaming response",
    description="Streams answer chunks in real-time via Server-Sent Events.",
)
async def query_knowledge_base_stream(request: QueryRequest) -> StreamingResponse:
    """Answer a question using streaming SSE events: chunk, done, error."""

    def _sse_event(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=True)}\n\n"

    events: queue.Queue[tuple[str, dict]] = queue.Queue()

    def _worker() -> None:
        start_time = time.perf_counter()

        def _on_chunk(text: str) -> None:
            if text:
                events.put(("chunk", {"text": text}))

        try:
            logger.info("Streaming query received: '%s'", request.question[:100])
            result = run_rag_pipeline(request.question, stream_callback=_on_chunk)
            elapsed = time.perf_counter() - start_time
            logger.info(
                "Streaming query answered in %.2fs | confidence=%.3f level=%s",
                elapsed,
                result.confidence_score,
                result.confidence_level,
            )
            payload = {
                "answer": result.answer,
                "sources": [src.model_dump() for src in result.sources],
                "confidence_score": result.confidence_score,
                "confidence_level": result.confidence_level,
                "diagnostics": result.diagnostics.model_dump() if settings.enable_retrieval_diagnostics else None,
            }
            events.put(("done", payload))
        except Exception as exc:
            logger.error("Streaming pipeline error: %s", exc, exc_info=True)
            events.put(("error", {"detail": str(exc)}))
        finally:
            events.put(("end", {}))

    def _event_generator():
        worker = threading.Thread(target=_worker, daemon=True)
        worker.start()

        while True:
            event_name, payload = events.get()
            if event_name == "end":
                break
            yield _sse_event(event_name, payload)

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
