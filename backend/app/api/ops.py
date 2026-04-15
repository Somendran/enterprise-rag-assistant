"""Operational, feedback, and admin/debug endpoints."""

from __future__ import annotations

import requests
import sys
import threading
from pathlib import Path
from uuid import uuid4
from fastapi import APIRouter, Depends, status

from app.api.security import require_api_key
from app.config import settings
from app.models.schemas import (
    AdminOverviewResponse,
    ChatMessageItem,
    ChatMessageRequest,
    ChatMessagesResponse,
    ChatSessionCreateRequest,
    ChatSessionItem,
    ChatSessionsResponse,
    EvalRunCreateResponse,
    EvalRunItem,
    EvalRunsResponse,
    FeedbackRequest,
    FeedbackResponse,
    ModelHealthItem,
    ModelHealthResponse,
)
from app.services import metadata_store
from app.services.rag_pipeline import run_rag_pipeline
from app.services.vector_store import list_indexed_documents

_repo_root = Path(__file__).resolve().parents[3]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from evals.run_eval import DEFAULT_EVAL_FILE, load_evals, score_eval

router = APIRouter(dependencies=[Depends(require_api_key)])


def _check_import(name: str, module: str) -> ModelHealthItem:
    try:
        __import__(module)
        return ModelHealthItem(name=name, status="ok", detail=f"{module} import succeeded")
    except Exception as exc:
        return ModelHealthItem(name=name, status="error", detail=str(exc))


@router.get(
    "/health/models",
    response_model=ModelHealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Model dependency health",
)
async def model_health() -> ModelHealthResponse:
    checks: list[ModelHealthItem] = [
        ModelHealthItem(
            name="embedding_config",
            status="ok",
            detail=f"{settings.embedding_model} on {settings.embedding_device}",
        ),
        _check_import("huggingface_embeddings", "langchain_huggingface"),
        _check_import("docling", "docling.document_converter"),
        _check_import("flag_embedding_reranker", "FlagEmbedding"),
    ]

    try:
        response = requests.get(
            settings.local_llm_endpoint.replace("/api/generate", "/api/tags"),
            timeout=3,
        )
        response.raise_for_status()
        models = [
            item.get("name", "")
            for item in response.json().get("models", [])
            if isinstance(item, dict)
        ]
        status_value = "ok" if settings.local_llm_model in models else "warning"
        checks.append(
            ModelHealthItem(
                name="ollama",
                status=status_value,
                detail=(
                    f"model '{settings.local_llm_model}' available"
                    if status_value == "ok"
                    else f"reachable, but '{settings.local_llm_model}' was not listed"
                ),
            )
        )
    except Exception as exc:
        checks.append(ModelHealthItem(name="ollama", status="error", detail=str(exc)))

    checks.append(
        ModelHealthItem(
            name="openai",
            status="configured" if settings.openai_api_key else "not_configured",
            detail="USE_OPENAI=true" if settings.use_openai else "optional path disabled",
        )
    )

    return ModelHealthResponse(checks=checks)


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Record answer feedback",
)
async def record_answer_feedback(request: FeedbackRequest) -> FeedbackResponse:
    stored = metadata_store.record_feedback(
        question=request.question,
        answer=request.answer,
        rating=request.rating,
        reason=request.reason,
        comment=request.comment,
        confidence_score=request.confidence_score,
        sources=[src.model_dump() for src in request.sources],
        diagnostics=request.diagnostics.model_dump() if request.diagnostics else {},
    )
    return FeedbackResponse(**stored)


@router.get(
    "/chat/sessions",
    response_model=ChatSessionsResponse,
    status_code=status.HTTP_200_OK,
    summary="List chat sessions",
)
async def list_chat_sessions() -> ChatSessionsResponse:
    sessions = [ChatSessionItem(**item) for item in metadata_store.list_chat_sessions()]
    return ChatSessionsResponse(sessions=sessions)


@router.post(
    "/chat/sessions",
    response_model=ChatSessionItem,
    status_code=status.HTTP_201_CREATED,
    summary="Create chat session",
)
async def create_chat_session(request: ChatSessionCreateRequest) -> ChatSessionItem:
    session_id = uuid4().hex
    session = metadata_store.create_chat_session(session_id, request.title.strip() or "New chat")
    return ChatSessionItem(**session)


@router.get(
    "/chat/sessions/{session_id}/messages",
    response_model=ChatMessagesResponse,
    status_code=status.HTTP_200_OK,
    summary="List chat session messages",
)
async def list_chat_messages(session_id: str) -> ChatMessagesResponse:
    messages = [ChatMessageItem(**item) for item in metadata_store.list_chat_messages(session_id)]
    return ChatMessagesResponse(messages=messages)


@router.post(
    "/chat/sessions/{session_id}/messages",
    response_model=ChatMessageItem,
    status_code=status.HTTP_201_CREATED,
    summary="Store chat message",
)
async def add_chat_message(session_id: str, request: ChatMessageRequest) -> ChatMessageItem:
    if metadata_store.get_chat_session(session_id) is None:
        metadata_store.create_chat_session(session_id, "New chat")

    message = metadata_store.add_chat_message(
        message_id=request.id or uuid4().hex,
        session_id=session_id,
        role=request.role,
        content=request.content,
        sources=[src.model_dump() for src in request.sources],
        diagnostics=request.diagnostics.model_dump() if request.diagnostics else {},
        confidence_score=request.confidence_score,
        confidence_level=request.confidence_level,
    )
    return ChatMessageItem(**{**message, "sources": request.sources, "diagnostics": request.diagnostics})


def _run_eval_background(run_id: str, eval_file: Path) -> None:
    results: list[dict] = []
    try:
        evals = load_evals(eval_file)
        for item in evals:
            try:
                payload = run_rag_pipeline(str(item["question"]))
                result = score_eval(
                    item,
                    {
                        "answer": payload.answer,
                        "sources": [src.model_dump() for src in payload.sources],
                        "confidence_score": payload.confidence_score,
                    },
                )
            except Exception as exc:
                result = type("EvalResultLike", (), {
                    "eval_id": str(item.get("id", "unknown")),
                    "passed": False,
                    "message": str(exc),
                })()
            results.append(
                {
                    "eval_id": result.eval_id,
                    "passed": bool(result.passed),
                    "message": result.message,
                }
            )
            metadata_store.update_eval_run(
                run_id,
                status="running",
                results=results,
                message=f"Completed {len(results)}/{len(evals)} eval(s).",
            )
        metadata_store.update_eval_run(
            run_id,
            status="completed",
            results=results,
            message="Eval run completed.",
        )
    except Exception as exc:
        metadata_store.update_eval_run(
            run_id,
            status="failed",
            results=results,
            message=str(exc),
        )


@router.post(
    "/evals/runs",
    response_model=EvalRunCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start a background eval run",
)
async def create_eval_run() -> EvalRunCreateResponse:
    evals = load_evals(DEFAULT_EVAL_FILE)
    run_id = uuid4().hex
    run = metadata_store.create_eval_run(run_id, total=len(evals))
    threading.Thread(target=_run_eval_background, args=(run_id, DEFAULT_EVAL_FILE), daemon=True).start()
    return EvalRunCreateResponse(run_id=run_id, status=str(run.get("status", "running")), total=len(evals))


@router.get(
    "/evals/runs",
    response_model=EvalRunsResponse,
    status_code=status.HTTP_200_OK,
    summary="List eval runs",
)
async def list_eval_runs() -> EvalRunsResponse:
    return EvalRunsResponse(runs=[EvalRunItem(**item) for item in metadata_store.list_eval_runs()])


@router.get(
    "/evals/runs/{run_id}",
    response_model=EvalRunItem,
    status_code=status.HTTP_200_OK,
    summary="Get eval run",
)
async def get_eval_run(run_id: str) -> EvalRunItem:
    run = metadata_store.get_eval_run(run_id)
    if run is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Eval run not found.")
    return EvalRunItem(**run)


@router.get(
    "/admin/overview",
    response_model=AdminOverviewResponse,
    status_code=status.HTTP_200_OK,
    summary="Local admin/debug overview",
)
async def admin_overview() -> AdminOverviewResponse:
    # Forces one-time migration from the previous JSON document registry.
    list_indexed_documents()
    summary = metadata_store.admin_summary()
    return AdminOverviewResponse(
        **summary,
        embedding_model=settings.embedding_model,
        embedding_device=settings.embedding_device,
        docling_enabled=settings.enable_docling,
        reranker_enabled=settings.enable_neural_reranker,
        openai_enabled=settings.use_openai,
    )
