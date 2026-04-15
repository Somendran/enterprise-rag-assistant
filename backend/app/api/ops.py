"""Operational, feedback, and admin/debug endpoints."""

from __future__ import annotations

import requests
from fastapi import APIRouter, Depends, status

from app.api.security import require_api_key
from app.config import settings
from app.models.schemas import (
    AdminOverviewResponse,
    FeedbackRequest,
    FeedbackResponse,
    ModelHealthItem,
    ModelHealthResponse,
)
from app.services import metadata_store
from app.services.vector_store import list_indexed_documents

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
