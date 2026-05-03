"""Operational, feedback, and admin/debug endpoints."""

from __future__ import annotations

import requests
import sys
import threading
from pathlib import Path
from uuid import uuid4
from fastapi import APIRouter, Depends, HTTPException, status

from app.api.security import AuthContext, require_admin, require_user
from app.config import settings
from app.models.schemas import (
    AdminOverviewResponse,
    AuditEventItem,
    AuditEventsResponse,
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

router = APIRouter()


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
async def model_health(_: AuthContext = Depends(require_admin)) -> ModelHealthResponse:
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

    if settings.use_openai and not settings.openai_fallback_to_local:
        checks.append(ModelHealthItem(name="ollama", status="skipped", detail="OpenAI-only generation is enabled."))
    else:
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
async def record_answer_feedback(
    request: FeedbackRequest,
    current_user: AuthContext = Depends(require_user),
) -> FeedbackResponse:
    if current_user.is_demo:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Public demo users cannot record feedback.")
    stored = metadata_store.record_feedback(
        question=request.question,
        answer=request.answer,
        rating=request.rating,
        reason=request.reason,
        comment=request.comment,
        confidence_score=request.confidence_score,
        sources=[src.model_dump() for src in request.sources],
        diagnostics=request.diagnostics.model_dump() if request.diagnostics else {},
        user_id=current_user.id,
    )
    metadata_store.record_audit_event(
        actor_user_id=current_user.id,
        actor_email=current_user.email,
        action="feedback.record",
        resource_type="feedback",
        resource_id=str(stored.get("id", "")),
        detail={"rating": request.rating, "reason": request.reason},
    )
    return FeedbackResponse(**stored)


@router.get(
    "/chat/sessions",
    response_model=ChatSessionsResponse,
    status_code=status.HTTP_200_OK,
    summary="List chat sessions",
)
async def list_chat_sessions(current_user: AuthContext = Depends(require_user)) -> ChatSessionsResponse:
    if current_user.is_demo:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Public demo users cannot use saved chats.")
    sessions = [
        ChatSessionItem(**item)
        for item in metadata_store.list_chat_sessions(
            user_id=current_user.id,
            include_all=current_user.role == "admin" or current_user.is_system_admin,
        )
    ]
    return ChatSessionsResponse(sessions=sessions)


@router.post(
    "/chat/sessions",
    response_model=ChatSessionItem,
    status_code=status.HTTP_201_CREATED,
    summary="Create chat session",
)
async def create_chat_session(
    request: ChatSessionCreateRequest,
    current_user: AuthContext = Depends(require_user),
) -> ChatSessionItem:
    if current_user.is_demo:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Public demo users cannot use saved chats.")
    session_id = uuid4().hex
    session = metadata_store.create_chat_session(session_id, request.title.strip() or "New chat", user_id=current_user.id)
    metadata_store.record_audit_event(
        actor_user_id=current_user.id,
        actor_email=current_user.email,
        action="chat.create",
        resource_type="chat_session",
        resource_id=session_id,
    )
    return ChatSessionItem(**session)


@router.delete(
    "/chat/sessions/{session_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete chat session",
)
async def delete_chat_session(
    session_id: str,
    current_user: AuthContext = Depends(require_user),
) -> None:
    if current_user.is_demo:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Public demo users cannot use saved chats.")
    session = metadata_store.get_chat_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Chat session not found.")
    if current_user.role != "admin" and not current_user.is_system_admin and str(session.get("user_id", "")) != current_user.id:
        raise HTTPException(status_code=403, detail="You cannot delete this chat session.")
    metadata_store.delete_chat_session(session_id)
    metadata_store.record_audit_event(
        actor_user_id=current_user.id,
        actor_email=current_user.email,
        action="chat.delete",
        resource_type="chat_session",
        resource_id=session_id,
    )


@router.get(
    "/chat/sessions/{session_id}/messages",
    response_model=ChatMessagesResponse,
    status_code=status.HTTP_200_OK,
    summary="List chat session messages",
)
async def list_chat_messages(
    session_id: str,
    current_user: AuthContext = Depends(require_user),
) -> ChatMessagesResponse:
    if current_user.is_demo:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Public demo users cannot use saved chats.")
    session = metadata_store.get_chat_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Chat session not found.")
    if current_user.role != "admin" and not current_user.is_system_admin and str(session.get("user_id", "")) != current_user.id:
        raise HTTPException(status_code=403, detail="You cannot access this chat session.")
    messages = [ChatMessageItem(**item) for item in metadata_store.list_chat_messages(session_id)]
    return ChatMessagesResponse(messages=messages)


@router.post(
    "/chat/sessions/{session_id}/messages",
    response_model=ChatMessageItem,
    status_code=status.HTTP_201_CREATED,
    summary="Store chat message",
)
async def add_chat_message(
    session_id: str,
    request: ChatMessageRequest,
    current_user: AuthContext = Depends(require_user),
) -> ChatMessageItem:
    if current_user.is_demo:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Public demo users cannot use saved chats.")
    session = metadata_store.get_chat_session(session_id)
    if session is None:
        metadata_store.create_chat_session(session_id, "New chat", user_id=current_user.id)
    elif current_user.role != "admin" and not current_user.is_system_admin and str(session.get("user_id", "")) != current_user.id:
        raise HTTPException(status_code=403, detail="You cannot update this chat session.")

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


def _run_eval_background(run_id: str, eval_file: Path, current_user: AuthContext) -> None:
    results: list[dict] = []
    try:
        evals = load_evals(eval_file)
        for item in evals:
            try:
                allowed_file_hashes = metadata_store.allowed_file_hashes_for_user(current_user.as_user())
                payload = run_rag_pipeline(
                    str(item["question"]),
                    allowed_file_hashes=allowed_file_hashes,
                    access_scope=current_user.id,
                )
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
async def create_eval_run(current_user: AuthContext = Depends(require_admin)) -> EvalRunCreateResponse:
    evals = load_evals(DEFAULT_EVAL_FILE)
    run_id = uuid4().hex
    run = metadata_store.create_eval_run(run_id, total=len(evals))
    threading.Thread(target=_run_eval_background, args=(run_id, DEFAULT_EVAL_FILE, current_user), daemon=True).start()
    metadata_store.record_audit_event(
        actor_user_id=current_user.id,
        actor_email=current_user.email,
        action="eval.start",
        resource_type="eval_run",
        resource_id=run_id,
        detail={"total": len(evals)},
    )
    return EvalRunCreateResponse(run_id=run_id, status=str(run.get("status", "running")), total=len(evals))


@router.get(
    "/evals/runs",
    response_model=EvalRunsResponse,
    status_code=status.HTTP_200_OK,
    summary="List eval runs",
)
async def list_eval_runs(_: AuthContext = Depends(require_admin)) -> EvalRunsResponse:
    return EvalRunsResponse(runs=[EvalRunItem(**item) for item in metadata_store.list_eval_runs()])


@router.get(
    "/evals/runs/{run_id}",
    response_model=EvalRunItem,
    status_code=status.HTTP_200_OK,
    summary="Get eval run",
)
async def get_eval_run(run_id: str, _: AuthContext = Depends(require_admin)) -> EvalRunItem:
    run = metadata_store.get_eval_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Eval run not found.")
    return EvalRunItem(**run)


@router.get(
    "/admin/overview",
    response_model=AdminOverviewResponse,
    status_code=status.HTTP_200_OK,
    summary="Local admin/debug overview",
)
async def admin_overview(_: AuthContext = Depends(require_admin)) -> AdminOverviewResponse:
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


@router.get(
    "/admin/audit-log",
    response_model=AuditEventsResponse,
    status_code=status.HTTP_200_OK,
    summary="List recent audit events",
)
async def audit_log(_: AuthContext = Depends(require_admin)) -> AuditEventsResponse:
    return AuditEventsResponse(events=[AuditEventItem(**item) for item in metadata_store.list_audit_events()])
