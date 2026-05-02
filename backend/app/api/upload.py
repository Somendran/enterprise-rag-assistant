"""Upload and knowledge-base management endpoints."""

import os
import time
import shutil
import hashlib
import threading
from dataclasses import dataclass
from uuid import uuid4
from pathlib import Path

from fastapi import APIRouter, Depends, Form, HTTPException, Query, Request, UploadFile, File, status

from app.api.security import AuthContext, require_admin, require_user
from app.api.upload_validation import count_pdf_pages, safe_pdf_filename, validate_pdf_upload
from app.services.document_loader import load_pdf
from app.services.text_splitter import split_documents, chunk_structured_blocks
from app.services.ingestion.doc_parser import parse_document
from app.services.ingestion.quality import assess_pdf_text_quality, summarize_block_text_quality
from app.services.ingestion.vision_enricher import enrich_blocks_with_vision, get_last_vision_calls_used
from app.services.query_cache import clear_query_cache
from app.services import metadata_store
from app.services.vector_store import (
    add_documents,
    cleanup_expired_demo_documents,
    delete_indexed_document,
    get_indexed_document,
    is_document_indexed,
    list_document_chunks,
    list_indexed_documents,
    register_indexed_document,
    reset_vector_store,
)
from app.models.schemas import (
    KnowledgeBaseFilesResponse,
    UploadBatchResponse,
    UploadItemResult,
    IngestionJobResponse,
    IngestionJobStatusResponse,
    ResetKnowledgeBaseResponse,
    DeleteKnowledgeBaseFileResponse,
    DocumentChunksResponse,
    DocumentPermissionsUpdateRequest,
)
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@dataclass
class PreparedChunks:
    chunks: list
    parsing_method: str
    vision_calls_used: int
    ocr_applied: bool
    text_coverage_ratio: float
    low_text_pages: int
    ingestion_warnings: list[str]


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip() or "unknown"
    return request.client.host if request.client else "unknown"


def _enforce_demo_rate_limit(current_user: AuthContext, request: Request, action: str) -> None:
    if not current_user.is_demo:
        return

    session_limit = settings.demo_uploads_per_hour if action == "upload" else settings.demo_queries_per_hour
    ip_limit = settings.demo_uploads_per_hour_ip if action == "upload" else settings.demo_queries_per_hour_ip
    session_result = metadata_store.check_rate_limit(
        key=current_user.id,
        action=action,
        limit=session_limit,
    )
    ip_result = metadata_store.check_rate_limit(
        key=f"ip:{_client_ip(request)}",
        action=action,
        limit=ip_limit,
    )
    if not session_result["allowed"] or not ip_result["allowed"]:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Public demo limit reached. Please try again later.",
        )


def _assert_demo_upload_budget(current_user: AuthContext, request: Request, files: list[UploadFile]) -> None:
    if not current_user.is_demo:
        return

    cleanup_expired_demo_documents()
    if len(files) > max(1, int(settings.demo_max_files_per_request)):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Public demo uploads are limited to {settings.demo_max_files_per_request} file(s) per request.",
        )

    current_count = metadata_store.count_documents_for_owner(current_user.id)
    if current_count + len(files) > max(1, int(settings.demo_max_docs_per_session)):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Public demo sessions are limited to {settings.demo_max_docs_per_session} uploaded document(s).",
        )

    _enforce_demo_rate_limit(current_user, request, "upload")


def _demo_expiry(current_user: AuthContext) -> int:
    if not current_user.is_demo:
        return 0
    return int(time.time()) + max(1, int(settings.demo_doc_ttl_hours)) * 3600


def _demo_upload_limit_bytes(current_user: AuthContext) -> tuple[int, int]:
    if current_user.is_demo:
        max_mb = max(1, int(settings.demo_max_upload_mb))
        return max_mb * 1024 * 1024, max_mb
    return settings.max_upload_size_bytes, settings.max_upload_size_mb


def _stored_upload_path(file_hash: str, filename: str) -> Path:
    return Path(settings.upload_dir) / f"{file_hash[:12]}_{filename}"


def _find_stored_upload(file_hash: str, filename: str, upload_path: str = "") -> Path | None:
    if upload_path:
        candidate = Path(upload_path)
        if candidate.exists() and candidate.is_file():
            return candidate

    expected = _stored_upload_path(file_hash, filename)
    if expected.exists() and expected.is_file():
        return expected

    upload_dir = Path(settings.upload_dir)
    matches = sorted(upload_dir.glob(f"{file_hash[:12]}_*.pdf")) if upload_dir.exists() else []
    return matches[0] if matches else None


def _delete_stored_upload(file_hash: str, filename: str, upload_path: str = "") -> bool:
    stored_path = _find_stored_upload(file_hash, filename, upload_path)
    if stored_path is None:
        return False
    stored_path.unlink(missing_ok=True)
    return True


def _prepare_chunks_for_indexing(
    file_path: Path,
    filename: str,
    file_hash: str,
    document_id: str,
    owner_user_id: str = "",
    visibility: str = "shared",
    allowed_roles: list[str] | None = None,
) -> PreparedChunks:
    chunks = []
    vision_calls_used = 0
    parsing_method = "legacy_pdf"
    ocr_applied = False
    quality = assess_pdf_text_quality(file_path)
    ingestion_warnings = list(quality.warnings)
    parsed_text_coverage = float(quality.text_coverage_ratio)
    parsed_low_text_pages = int(quality.low_text_pages)

    if settings.enable_docling:
        try:
            force_ocr = bool(settings.enable_ocr)
            blocks = parse_document(str(file_path), force_ocr=force_ocr)
            if settings.enable_vision_enrichment:
                blocks = enrich_blocks_with_vision(blocks)
                vision_calls_used = get_last_vision_calls_used()

            block_coverage, block_low_pages = summarize_block_text_quality(blocks)
            should_retry_ocr = (
                (not force_ocr)
                and quality.ocr_recommended
                and block_coverage < float(settings.ingestion_ocr_page_ratio_threshold)
            )

            if should_retry_ocr:
                logger.info(
                    "Retrying structured ingestion with OCR | file=%s text_coverage=%.3f low_text_pages=%d",
                    filename,
                    quality.text_coverage_ratio,
                    quality.low_text_pages,
                )
                ocr_blocks = parse_document(str(file_path), force_ocr=True)
                if settings.enable_vision_enrichment:
                    ocr_blocks = enrich_blocks_with_vision(ocr_blocks)
                    vision_calls_used = get_last_vision_calls_used()
                if ocr_blocks:
                    blocks = ocr_blocks
                    ocr_applied = True
                    parsing_method = "docling_ocr"
                    block_coverage, block_low_pages = summarize_block_text_quality(blocks)

            chunks = chunk_structured_blocks(blocks)
            if not ocr_applied:
                ocr_applied = force_ocr
                parsing_method = "docling_ocr" if force_ocr else "docling"
            if block_coverage < quality.text_coverage_ratio:
                block_coverage = quality.text_coverage_ratio
            low_text_pages = max(quality.low_text_pages, block_low_pages)
            parsed_text_coverage = float(block_coverage)
            parsed_low_text_pages = int(low_text_pages)
            logger.info(
                "Structured ingestion completed | file=%s blocks=%d chunks=%d parsing_method=%s text_coverage=%.3f low_text_pages=%d vision_calls_used=%d",
                filename,
                len(blocks),
                len(chunks),
                parsing_method,
                block_coverage,
                low_text_pages,
                vision_calls_used,
            )
        except Exception as docling_exc:
            parsing_method = "legacy_pdf_fallback"
            ingestion_warnings.append(f"Structured parser failed; used legacy PDF parser: {docling_exc}")
            logger.warning(
                "Structured ingestion failed; falling back to legacy parser | file=%s error=%s",
                filename,
                docling_exc,
            )

    if not chunks:
        documents = load_pdf(file_path)
        chunks = split_documents(documents)

    if not chunks:
        if quality.ocr_recommended:
            raise ValueError(
                "Document appears to be scanned and OCR did not produce parseable text. "
                "Enable OCR support or upload a text-searchable PDF."
            )
        raise ValueError("Document contains no parseable text.")

    text_coverage_ratio = float(parsed_text_coverage)
    low_text_pages = int(parsed_low_text_pages)

    indexed_at = int(time.time())
    for idx, chunk in enumerate(chunks):
        chunk.metadata.update(
            {
                "source": filename,
                "file_hash": file_hash,
                "document_id": document_id,
                "owner_user_id": owner_user_id,
                "visibility": visibility,
                "allowed_roles": allowed_roles or [],
                "parsing_method": parsing_method,
                "ocr_applied": ocr_applied,
                "text_coverage_ratio": text_coverage_ratio,
                "low_text_pages": low_text_pages,
                "ingestion_warnings": ingestion_warnings,
                "chunk_index": idx,
                "indexed_at": indexed_at,
            }
        )

    return PreparedChunks(
        chunks=chunks,
        parsing_method=parsing_method,
        vision_calls_used=vision_calls_used,
        ocr_applied=ocr_applied,
        text_coverage_ratio=text_coverage_ratio,
        low_text_pages=low_text_pages,
        ingestion_warnings=ingestion_warnings,
    )


def _result_to_dict(item: UploadItemResult) -> dict:
    return item.model_dump()


def _job_response(payload: dict) -> IngestionJobStatusResponse:
    normalized = {
        **payload,
        "job_id": str(payload.get("id", "")),
        "results": [
            item if isinstance(item, UploadItemResult) else UploadItemResult(**item)
            for item in payload.get("results", [])
        ],
    }
    return IngestionJobStatusResponse(**normalized)


def _parse_allowed_roles(raw: str = "") -> list[str]:
    return [
        item.strip().lower()
        for item in raw.split(",")
        if item.strip()
    ]


def _normalize_visibility(raw: str = "shared") -> str:
    value = (raw or "shared").strip().lower()
    return value if value in {"private", "shared", "role"} else "shared"


def _ingest_saved_files(
    saved_files: list[dict],
    job_id: str | None = None,
    current_user: AuthContext | None = None,
    visibility: str = "shared",
    allowed_roles: list[str] | None = None,
) -> UploadBatchResponse:
    results: list[UploadItemResult] = []
    total_chunks_indexed = 0

    for idx, item in enumerate(saved_files, start=1):
        safe_filename = str(item["filename"])
        file_hash = str(item["file_hash"])
        document_id = str(item["document_id"])
        file_path = Path(str(item["file_path"]))

        if job_id:
            metadata_store.update_ingestion_job(
                job_id,
                status="running",
                stage="parsing",
                message=f"Parsing {safe_filename}",
                processed_files=idx - 1,
                total_chunks_indexed=total_chunks_indexed,
                results=[_result_to_dict(result) for result in results],
            )

        if is_document_indexed(file_hash):
            existing = get_indexed_document(file_hash) or {}
            if file_path.exists():
                os.remove(file_path)
            if current_user is not None and not metadata_store.can_user_access_document(current_user.as_user(), existing):
                results.append(
                    UploadItemResult(
                        filename=safe_filename,
                        chunks_indexed=0,
                        status="failed",
                        message="Duplicate document already exists, but your account cannot access it.",
                        file_hash=file_hash,
                        document_id=str(existing.get("document_id", document_id)),
                    )
                )
                continue
            results.append(
                UploadItemResult(
                    filename=safe_filename,
                    chunks_indexed=int(existing.get("chunk_count", 0) or 0),
                    status="duplicate",
                    message="Duplicate document detected. Existing index entry was reused.",
                    file_hash=file_hash,
                    document_id=str(existing.get("document_id", document_id)),
                    parsing_method=str(existing.get("parsing_method", "")) or None,
                    vision_calls_used=int(existing.get("vision_calls_used", 0) or 0),
                    ocr_applied=bool(existing.get("ocr_applied", False)),
                    text_coverage_ratio=float(existing.get("text_coverage_ratio", 0.0) or 0.0),
                    low_text_pages=int(existing.get("low_text_pages", 0) or 0),
                    ingestion_warnings=list(existing.get("ingestion_warnings", [])),
                )
            )
            continue

        try:
            prepared = _prepare_chunks_for_indexing(
                file_path=file_path,
                filename=safe_filename,
                file_hash=file_hash,
                document_id=document_id,
                owner_user_id=current_user.id if current_user else "",
                visibility=visibility,
                allowed_roles=allowed_roles,
            )

            if job_id:
                metadata_store.update_ingestion_job(
                    job_id,
                    status="running",
                    stage="embedding",
                    message=f"Embedding and indexing {safe_filename}",
                    processed_files=idx - 1,
                    total_chunks_indexed=total_chunks_indexed,
                    results=[_result_to_dict(result) for result in results],
                )

            add_documents(prepared.chunks)
            register_indexed_document(
                file_hash=file_hash,
                filename=safe_filename,
                chunk_count=len(prepared.chunks),
                document_id=document_id,
                parsing_method=prepared.parsing_method,
                upload_path=str(file_path),
                upload_status="indexed",
                vision_calls_used=prepared.vision_calls_used,
                owner_user_id=current_user.id if current_user else "",
                visibility=visibility,
                allowed_roles=allowed_roles,
                is_demo=bool(current_user.is_demo) if current_user else False,
                demo_session_id=current_user.id if current_user and current_user.is_demo else "",
                expires_at=_demo_expiry(current_user) if current_user else 0,
                ocr_applied=prepared.ocr_applied,
                text_coverage_ratio=prepared.text_coverage_ratio,
                low_text_pages=prepared.low_text_pages,
                ingestion_warnings=prepared.ingestion_warnings,
            )
            total_chunks_indexed += len(prepared.chunks)
            results.append(
                UploadItemResult(
                    filename=safe_filename,
                    chunks_indexed=len(prepared.chunks),
                    status="success",
                    message="Document ingested successfully.",
                    file_hash=file_hash,
                    document_id=document_id,
                    parsing_method=prepared.parsing_method,
                    vision_calls_used=prepared.vision_calls_used,
                    ocr_applied=prepared.ocr_applied,
                    text_coverage_ratio=prepared.text_coverage_ratio,
                    low_text_pages=prepared.low_text_pages,
                    ingestion_warnings=prepared.ingestion_warnings,
                )
            )
        except Exception as exc:
            logger.error("Ingestion failed for '%s': %s", safe_filename, exc, exc_info=True)
            if file_path.exists():
                os.remove(file_path)
            results.append(
                UploadItemResult(
                    filename=safe_filename,
                    chunks_indexed=0,
                    status="failed",
                    message=f"Document ingestion failed: {exc}",
                    file_hash=file_hash,
                    document_id=document_id,
                )
            )

    processed_files = sum(1 for result in results if result.status in {"success", "duplicate"})
    return UploadBatchResponse(
        files=results,
        total_files=len(saved_files),
        processed_files=processed_files,
        total_chunks_indexed=total_chunks_indexed,
        message=f"Upload completed. Processed {processed_files}/{len(saved_files)} files.",
    )


@router.get(
    "/knowledge-base/files",
    response_model=KnowledgeBaseFilesResponse,
    status_code=status.HTTP_200_OK,
    summary="List indexed files",
    description="Returns persisted indexed document metadata for UI hydration on reload.",
)
async def list_knowledge_base_files(
    current_user: AuthContext = Depends(require_user),
) -> KnowledgeBaseFilesResponse:
    cleanup_expired_demo_documents()
    list_indexed_documents()
    files = metadata_store.list_documents_for_user(current_user.as_user())
    return KnowledgeBaseFilesResponse(files=files)


@router.get(
    "/knowledge-base/files/{file_hash}/chunks",
    response_model=DocumentChunksResponse,
    status_code=status.HTTP_200_OK,
    summary="Inspect indexed chunks for one file",
    description="Returns stored chunk text and metadata for source inspection/debugging.",
)
async def list_knowledge_base_file_chunks(
    file_hash: str,
    focus_chunk_index: int | None = Query(default=None, ge=0),
    neighbor_window: int = Query(default=0, ge=0, le=20),
    current_user: AuthContext = Depends(require_user),
) -> DocumentChunksResponse:
    meta = get_indexed_document(file_hash)
    if meta is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )
    if not metadata_store.can_user_access_document(current_user.as_user(), meta):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You cannot access this document.")
    chunks = list_document_chunks(
        file_hash,
        focus_chunk_index=focus_chunk_index,
        neighbor_window=neighbor_window,
    )
    return DocumentChunksResponse(
        file_hash=file_hash,
        filename=str(meta.get("filename", "")),
        chunks=chunks,
        focus_chunk_index=focus_chunk_index,
    )


@router.patch(
    "/knowledge-base/files/{file_hash}/permissions",
    response_model=KnowledgeBaseFilesResponse,
    status_code=status.HTTP_200_OK,
    summary="Update document permissions",
)
async def update_knowledge_base_file_permissions(
    file_hash: str,
    request: DocumentPermissionsUpdateRequest,
    current_user: AuthContext = Depends(require_user),
) -> KnowledgeBaseFilesResponse:
    if current_user.is_demo:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Public demo users cannot manage document permissions.",
        )
    meta = get_indexed_document(file_hash)
    if meta is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    if not metadata_store.can_user_access_document(current_user.as_user(), meta, write=True):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You cannot manage this document.")

    updated = metadata_store.update_document_permissions(
        file_hash,
        visibility=request.visibility,
        allowed_roles=request.allowed_roles,
    )
    if updated is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    clear_query_cache()
    metadata_store.record_audit_event(
        actor_user_id=current_user.id,
        actor_email=current_user.email,
        action="document.permissions.update",
        resource_type="document",
        resource_id=file_hash,
        detail={
            "filename": str(updated.get("filename", "")),
            "visibility": str(updated.get("visibility", "")),
            "allowed_roles": list(updated.get("allowed_roles", [])),
        },
    )
    files = metadata_store.list_documents_for_user(current_user.as_user())
    return KnowledgeBaseFilesResponse(files=files)


@router.post(
    "/knowledge-base/reset",
    response_model=ResetKnowledgeBaseResponse,
    status_code=status.HTTP_200_OK,
    summary="Reset knowledge base",
    description="Deletes uploaded PDFs and clears persisted vector index data.",
)
async def reset_knowledge_base(
    current_user: AuthContext = Depends(require_admin),
) -> ResetKnowledgeBaseResponse:
    """Clear indexed data and uploaded files for a fresh knowledge base state."""
    uploads_deleted = 0
    upload_dir = Path(settings.upload_dir)

    try:
        if upload_dir.exists():
            for child in upload_dir.iterdir():
                if child.is_file():
                    child.unlink(missing_ok=True)
                    uploads_deleted += 1
                elif child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)

        index_cleared = reset_vector_store()
        clear_query_cache()
        metadata_store.record_audit_event(
            actor_user_id=current_user.id,
            actor_email=current_user.email,
            action="knowledge_base.reset",
            resource_type="knowledge_base",
            detail={"uploads_deleted": uploads_deleted, "index_cleared": index_cleared},
        )
        logger.info(
            "Knowledge base reset completed | uploads_deleted=%d index_cleared=%s",
            uploads_deleted,
            index_cleared,
        )
        return ResetKnowledgeBaseResponse(
            message="Knowledge base reset successfully.",
            index_cleared=index_cleared,
            uploads_deleted=uploads_deleted,
        )
    except Exception as e:
        logger.error("Failed to reset knowledge base: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reset knowledge base. Please try again.",
        )


@router.delete(
    "/knowledge-base/files/{file_hash}",
    response_model=DeleteKnowledgeBaseFileResponse,
    status_code=status.HTTP_200_OK,
    summary="Delete one indexed file",
    description="Removes one document from FAISS, the document registry, and stored uploads.",
)
async def delete_knowledge_base_file(
    file_hash: str,
    current_user: AuthContext = Depends(require_user),
) -> DeleteKnowledgeBaseFileResponse:
    if current_user.is_demo:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Public demo users cannot delete documents.",
        )
    try:
        meta = get_indexed_document(file_hash)
        if meta is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found.",
            )
        if not metadata_store.can_user_access_document(current_user.as_user(), meta, write=True):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You cannot delete this document.")

        delete_result = delete_indexed_document(file_hash)
        upload_deleted = _delete_stored_upload(
            file_hash=file_hash,
            filename=str(meta.get("filename", "")),
            upload_path=str(meta.get("upload_path", "")),
        )
        clear_query_cache()
        metadata_store.record_audit_event(
            actor_user_id=current_user.id,
            actor_email=current_user.email,
            action="document.delete",
            resource_type="document",
            resource_id=file_hash,
            detail={
                "filename": str(meta.get("filename", "")),
                "chunks_deleted": int(delete_result.get("chunks_deleted", 0) or 0),
            },
        )
        return DeleteKnowledgeBaseFileResponse(
            file_hash=file_hash,
            filename=str(meta.get("filename", "")),
            chunks_deleted=int(delete_result.get("chunks_deleted", 0) or 0),
            upload_deleted=upload_deleted,
            message="Document deleted successfully.",
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to delete indexed document '%s': %s", file_hash, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete document. Please try again.",
        )


@router.post(
    "/knowledge-base/files/{file_hash}/reindex",
    response_model=UploadItemResult,
    status_code=status.HTTP_200_OK,
    summary="Reindex one stored file",
    description="Rebuilds chunks and vectors for a stored PDF without requiring another upload.",
)
async def reindex_knowledge_base_file(
    file_hash: str,
    current_user: AuthContext = Depends(require_user),
) -> UploadItemResult:
    if current_user.is_demo:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Public demo users cannot reindex documents.",
        )
    meta = get_indexed_document(file_hash)
    if meta is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )
    if not metadata_store.can_user_access_document(current_user.as_user(), meta, write=True):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You cannot reindex this document.")

    filename = str(meta.get("filename", ""))
    stored_path = _find_stored_upload(
        file_hash=file_hash,
        filename=filename,
        upload_path=str(meta.get("upload_path", "")),
    )
    if stored_path is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Stored PDF is missing. Upload the document again before reindexing.",
        )

    document_id = str(meta.get("document_id", "")) or hashlib.sha1(
        f"{filename}:{file_hash}".encode("utf-8")
    ).hexdigest()[:16]

    try:
        prepared = _prepare_chunks_for_indexing(
            file_path=stored_path,
            filename=filename,
            file_hash=file_hash,
            document_id=document_id,
            owner_user_id=str(meta.get("owner_user_id", "")),
            visibility=str(meta.get("visibility", "shared")),
            allowed_roles=list(meta.get("allowed_roles", [])),
        )
        delete_indexed_document(file_hash)
        add_documents(prepared.chunks)
        register_indexed_document(
            file_hash=file_hash,
            filename=filename,
            chunk_count=len(prepared.chunks),
            document_id=document_id,
            parsing_method=prepared.parsing_method,
            upload_path=str(stored_path),
            upload_status="indexed",
            vision_calls_used=prepared.vision_calls_used,
            owner_user_id=str(meta.get("owner_user_id", "")),
            visibility=str(meta.get("visibility", "shared")),
            allowed_roles=list(meta.get("allowed_roles", [])),
            ocr_applied=prepared.ocr_applied,
            text_coverage_ratio=prepared.text_coverage_ratio,
            low_text_pages=prepared.low_text_pages,
            ingestion_warnings=prepared.ingestion_warnings,
        )
        clear_query_cache()
        metadata_store.record_audit_event(
            actor_user_id=current_user.id,
            actor_email=current_user.email,
            action="document.reindex",
            resource_type="document",
            resource_id=file_hash,
            detail={"filename": filename, "chunks_indexed": len(prepared.chunks)},
        )
        return UploadItemResult(
            filename=filename,
            chunks_indexed=len(prepared.chunks),
            status="success",
            message="Document reindexed successfully.",
            file_hash=file_hash,
            document_id=document_id,
            parsing_method=prepared.parsing_method,
            vision_calls_used=prepared.vision_calls_used,
            ocr_applied=prepared.ocr_applied,
            text_coverage_ratio=prepared.text_coverage_ratio,
            low_text_pages=prepared.low_text_pages,
            ingestion_warnings=prepared.ingestion_warnings,
        )
    except Exception as exc:
        logger.error("Failed to reindex document '%s': %s", file_hash, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reindex document. Please try again.",
        )


@router.post(
    "/upload",
    response_model=UploadBatchResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload and index one or more PDF documents",
    description=(
        "Accepts one or more PDF files, extracts text, splits into chunks, "
        "embeds each chunk, and stores them in the vector store. "
        "Returns per-file outcomes and aggregate indexing counts."
    ),
)
async def upload_document(
    request: Request,
    files: list[UploadFile] = File(...),
    visibility: str = Form(default="shared"),
    allowed_roles: str = Form(default=""),
    current_user: AuthContext = Depends(require_user),
) -> UploadBatchResponse:
    """Ingest one or more PDFs into the RAG knowledge base."""

    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files received. Please upload at least one .pdf file.",
        )

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved_files: list[dict] = []
    failed_results: list[UploadItemResult] = []

    for file in files:
        safe_filename = safe_pdf_filename(file.filename)
        if safe_filename is None:
            failed_results.append(
                UploadItemResult(
                    filename=file.filename or "unknown",
                    chunks_indexed=0,
                    status="failed",
                    message="Only PDF files are supported. Please upload a .pdf file.",
                )
            )
            continue

        logger.info("Upload received: '%s'", file.filename)

        try:
            content = await file.read()
            max_bytes, max_mb = _demo_upload_limit_bytes(current_user)
            validation_error = validate_pdf_upload(
                filename=file.filename,
                content_type=file.content_type,
                content=content,
                max_upload_size_bytes=max_bytes,
                max_upload_size_mb=max_mb,
            )
            if validation_error:
                failed_results.append(
                    UploadItemResult(
                        filename=file.filename or "unknown",
                        chunks_indexed=0,
                        status="failed",
                        message=validation_error,
                    )
                )
                continue
            if current_user.is_demo:
                try:
                    page_count = count_pdf_pages(content)
                except ValueError as exc:
                    failed_results.append(
                        UploadItemResult(
                            filename=file.filename or "unknown",
                            chunks_indexed=0,
                            status="failed",
                            message=str(exc),
                        )
                    )
                    continue
                if page_count > max(1, int(settings.demo_max_pages)):
                    failed_results.append(
                        UploadItemResult(
                            filename=file.filename or "unknown",
                            chunks_indexed=0,
                            status="failed",
                            message=f"Public demo PDFs are limited to {settings.demo_max_pages} page(s).",
                        )
                    )
                    continue

            file_hash = hashlib.sha256(content).hexdigest()
            document_id = hashlib.sha1(f"{safe_filename}:{file_hash}".encode("utf-8")).hexdigest()[:16]
            file_path = _stored_upload_path(file_hash, safe_filename)
            with open(file_path, "wb") as f:
                f.write(content)
            saved_files.append(
                {
                    "filename": safe_filename,
                    "file_hash": file_hash,
                    "document_id": document_id,
                    "file_path": str(file_path),
                }
            )
        except Exception as e:
            logger.error("Failed to save file '%s': %s", file.filename, e)
            failed_results.append(
                UploadItemResult(
                    filename=file.filename,
                    chunks_indexed=0,
                    status="failed",
                    message=f"Could not save uploaded file: {e}",
                )
            )

    normalized_visibility = "private" if current_user.is_demo else _normalize_visibility(visibility)
    parsed_roles = _parse_allowed_roles(allowed_roles)
    response = _ingest_saved_files(
        saved_files,
        current_user=current_user,
        visibility=normalized_visibility,
        allowed_roles=parsed_roles,
    )
    results = failed_results + response.files
    processed_files = sum(1 for item in results if item.status in {"success", "duplicate"})
    total_chunks_indexed = sum(item.chunks_indexed for item in results if item.status == "success")
    logger.info(
        "Batch ingest completed | total_files=%d processed_files=%d total_chunks_indexed=%d",
        len(files),
        processed_files,
        total_chunks_indexed,
    )

    metadata_store.record_audit_event(
        actor_user_id=current_user.id,
        actor_email=current_user.email,
        action="document.upload",
        resource_type="document",
        detail={
            "total_files": len(files),
            "processed_files": processed_files,
            "total_chunks_indexed": total_chunks_indexed,
            "visibility": normalized_visibility,
            "allowed_roles": parsed_roles,
            "public_demo": current_user.is_demo,
        },
    )

    return UploadBatchResponse(
        files=results,
        total_files=len(files),
        processed_files=processed_files,
        total_chunks_indexed=total_chunks_indexed,
        message=f"Upload completed. Processed {processed_files}/{len(files)} files.",
    )


@router.post(
    "/upload/jobs",
    response_model=IngestionJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload PDFs and ingest them in the background",
)
async def upload_document_job(
    request: Request,
    files: list[UploadFile] = File(...),
    visibility: str = Form(default="shared"),
    allowed_roles: str = Form(default=""),
    current_user: AuthContext = Depends(require_user),
) -> IngestionJobResponse:
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files received. Please upload at least one .pdf file.",
        )
    _assert_demo_upload_budget(current_user, request, files)
    _assert_demo_upload_budget(current_user, request, files)

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    job_id = uuid4().hex
    normalized_visibility = "private" if current_user.is_demo else _normalize_visibility(visibility)
    parsed_roles = _parse_allowed_roles(allowed_roles)
    metadata_store.create_ingestion_job(job_id, total_files=len(files), created_by_user_id=current_user.id)
    saved_files: list[dict] = []
    early_results: list[UploadItemResult] = []

    for file in files:
        safe_filename = safe_pdf_filename(file.filename)
        if safe_filename is None:
            early_results.append(
                UploadItemResult(
                    filename=file.filename or "unknown",
                    chunks_indexed=0,
                    status="failed",
                    message="Only PDF files are supported. Please upload a .pdf file.",
                )
            )
            continue
        try:
            content = await file.read()
            max_bytes, max_mb = _demo_upload_limit_bytes(current_user)
            validation_error = validate_pdf_upload(
                filename=file.filename,
                content_type=file.content_type,
                content=content,
                max_upload_size_bytes=max_bytes,
                max_upload_size_mb=max_mb,
            )
            if validation_error:
                early_results.append(
                    UploadItemResult(
                        filename=file.filename or "unknown",
                        chunks_indexed=0,
                        status="failed",
                        message=validation_error,
                    )
                )
                continue
            if current_user.is_demo:
                try:
                    page_count = count_pdf_pages(content)
                except ValueError as exc:
                    early_results.append(
                        UploadItemResult(
                            filename=file.filename or "unknown",
                            chunks_indexed=0,
                            status="failed",
                            message=str(exc),
                        )
                    )
                    continue
                if page_count > max(1, int(settings.demo_max_pages)):
                    early_results.append(
                        UploadItemResult(
                            filename=file.filename or "unknown",
                            chunks_indexed=0,
                            status="failed",
                            message=f"Public demo PDFs are limited to {settings.demo_max_pages} page(s).",
                        )
                    )
                    continue
            file_hash = hashlib.sha256(content).hexdigest()
            document_id = hashlib.sha1(f"{safe_filename}:{file_hash}".encode("utf-8")).hexdigest()[:16]
            file_path = _stored_upload_path(file_hash, safe_filename)
            with open(file_path, "wb") as f:
                f.write(content)
            saved_files.append(
                {
                    "filename": safe_filename,
                    "file_hash": file_hash,
                    "document_id": document_id,
                    "file_path": str(file_path),
                }
            )
        except Exception as exc:
            early_results.append(
                UploadItemResult(
                    filename=file.filename or "unknown",
                    chunks_indexed=0,
                    status="failed",
                    message=f"Could not save uploaded file: {exc}",
                )
            )

    metadata_store.update_ingestion_job(
        job_id,
        status="queued",
        stage="queued",
        message=f"Queued {len(saved_files)} valid file(s).",
        results=[_result_to_dict(result) for result in early_results],
    )

    def _worker() -> None:
        try:
            response = _ingest_saved_files(
                saved_files,
                job_id=job_id,
                current_user=current_user,
                visibility=normalized_visibility,
                allowed_roles=parsed_roles,
            )
            results = early_results + response.files
            processed = sum(1 for result in results if result.status in {"success", "duplicate"})
            total_chunks = sum(result.chunks_indexed for result in results if result.status == "success")
            status_value = "failed" if processed == 0 and results else "completed"
            metadata_store.update_ingestion_job(
                job_id,
                status=status_value,
                stage="complete" if status_value == "completed" else "failed",
                message=f"Processed {processed}/{len(files)} file(s).",
                processed_files=processed,
                total_chunks_indexed=total_chunks,
                results=[_result_to_dict(result) for result in results],
            )
            clear_query_cache()
            metadata_store.record_audit_event(
                actor_user_id=current_user.id,
                actor_email=current_user.email,
                action="document.upload_job.completed",
                resource_type="ingestion_job",
                resource_id=job_id,
                detail={
                    "total_files": len(files),
                    "processed_files": processed,
                    "total_chunks_indexed": total_chunks,
                    "visibility": normalized_visibility,
                    "allowed_roles": parsed_roles,
                    "public_demo": current_user.is_demo,
                },
            )
        except Exception as exc:
            logger.error("Background ingestion job failed | job_id=%s error=%s", job_id, exc, exc_info=True)
            metadata_store.update_ingestion_job(
                job_id,
                status="failed",
                stage="failed",
                message=str(exc),
                results=[_result_to_dict(result) for result in early_results],
            )
            metadata_store.record_audit_event(
                actor_user_id=current_user.id,
                actor_email=current_user.email,
                action="document.upload_job.failed",
                resource_type="ingestion_job",
                resource_id=job_id,
                detail={"error": str(exc)},
            )

    threading.Thread(target=_worker, daemon=True).start()
    payload = metadata_store.get_ingestion_job(job_id) or {}
    return IngestionJobResponse(
        job_id=job_id,
        status=str(payload.get("status", "queued")),
        stage=str(payload.get("stage", "queued")),
        message=str(payload.get("message", "")),
    )


@router.get(
    "/upload/jobs/{job_id}",
    response_model=IngestionJobStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Get background ingestion job status",
)
async def get_upload_job(
    job_id: str,
    current_user: AuthContext = Depends(require_user),
) -> IngestionJobStatusResponse:
    payload = metadata_store.get_ingestion_job(job_id)
    if payload is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ingestion job not found.")
    if current_user.role != "admin" and str(payload.get("created_by_user_id", "")) not in {"", current_user.id}:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You cannot access this ingestion job.")
    return _job_response(payload)
