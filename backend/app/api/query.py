"""Query endpoints for standard and streaming RAG responses."""

import time
import json
import queue
import threading
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from app.api.security import AuthContext, require_user
from app.services import metadata_store
from app.services.rag_pipeline import run_rag_pipeline
from app.services.vector_store import cleanup_expired_demo_documents
from app.models.schemas import QueryRequest, QueryResponse
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip() or "unknown"
    return request.client.host if request.client else "unknown"


def _enforce_demo_query_rate_limit(current_user: AuthContext, request: Request) -> None:
    if not current_user.is_demo:
        return
    session_result = metadata_store.check_rate_limit(
        key=current_user.id,
        action="query",
        limit=settings.demo_queries_per_hour,
    )
    ip_result = metadata_store.check_rate_limit(
        key=f"ip:{_client_ip(request)}",
        action="query",
        limit=settings.demo_queries_per_hour_ip,
    )
    if not session_result["allowed"] or not ip_result["allowed"]:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Public demo query limit reached. Please try again later.",
        )


def _diagnostics_for_user(current_user: AuthContext, diagnostics):
    if current_user.is_demo:
        return None
    if not (settings.enable_retrieval_diagnostics and (current_user.role == "admin" or current_user.is_system_admin)):
        return None
    return diagnostics


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
async def query_knowledge_base(
    request: QueryRequest,
    http_request: Request,
    current_user: AuthContext = Depends(require_user),
) -> QueryResponse:
    """Answer a question using the RAG pipeline."""

    start_time = time.perf_counter()
    logger.info(f"Query received: '{request.question[:100]}'")
    cleanup_expired_demo_documents()
    _enforce_demo_query_rate_limit(current_user, http_request)

    try:
        allowed_file_hashes = metadata_store.allowed_file_hashes_for_user(current_user.as_user())
        result = run_rag_pipeline(
            request.question,
            allowed_file_hashes=allowed_file_hashes,
            access_scope=current_user.id,
        )
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
    metadata_store.record_audit_event(
        actor_user_id=current_user.id,
        actor_email=current_user.email,
        action="query.run",
        resource_type="query",
        detail={
            "question_preview": request.question[:160],
            "sources": [s.document for s in result.sources],
            "confidence_score": result.confidence_score,
        },
    )

    return QueryResponse(
        answer=result.answer,
        sources=result.sources,
        confidence_score=result.confidence_score,
        confidence_level=result.confidence_level,
        diagnostics=_diagnostics_for_user(current_user, result.diagnostics),
    )


@router.post(
    "/query/stream",
    status_code=status.HTTP_200_OK,
    summary="Ask a question with streaming response",
    description="Streams answer chunks in real-time via Server-Sent Events.",
)
async def query_knowledge_base_stream(
    request: QueryRequest,
    http_request: Request,
    current_user: AuthContext = Depends(require_user),
) -> StreamingResponse:
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
            cleanup_expired_demo_documents()
            _enforce_demo_query_rate_limit(current_user, http_request)
            allowed_file_hashes = metadata_store.allowed_file_hashes_for_user(current_user.as_user())
            result = run_rag_pipeline(
                request.question,
                stream_callback=_on_chunk,
                allowed_file_hashes=allowed_file_hashes,
                access_scope=current_user.id,
            )
            elapsed = time.perf_counter() - start_time
            logger.info(
                "Streaming query answered in %.2fs | confidence=%.3f level=%s",
                elapsed,
                result.confidence_score,
                result.confidence_level,
            )
            diagnostics = _diagnostics_for_user(current_user, result.diagnostics)
            payload = {
                "answer": result.answer,
                "sources": [src.model_dump() for src in result.sources],
                "confidence_score": result.confidence_score,
                "confidence_level": result.confidence_level,
                "diagnostics": diagnostics.model_dump() if diagnostics else None,
            }
            metadata_store.record_audit_event(
                actor_user_id=current_user.id,
                actor_email=current_user.email,
                action="query.stream",
                resource_type="query",
                detail={
                    "question_preview": request.question[:160],
                    "sources": [src.document for src in result.sources],
                    "confidence_score": result.confidence_score,
                },
            )
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
