"""Upload and knowledge-base management endpoints."""

import os
import time
import shutil
import hashlib
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status

from app.api.security import require_api_key
from app.api.upload_validation import safe_pdf_filename, validate_pdf_upload
from app.services.document_loader import load_pdf
from app.services.text_splitter import split_documents, chunk_structured_blocks
from app.services.ingestion.doc_parser import parse_document
from app.services.ingestion.vision_enricher import enrich_blocks_with_vision, get_last_vision_calls_used
from app.services.query_cache import clear_query_cache
from app.services.vector_store import (
    add_documents,
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
    ResetKnowledgeBaseResponse,
    DeleteKnowledgeBaseFileResponse,
    DocumentChunksResponse,
)
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(dependencies=[Depends(require_api_key)])


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
) -> tuple[list, str, int]:
    chunks = []
    vision_calls_used = 0
    parsing_method = "legacy_pdf"

    if settings.enable_docling:
        try:
            blocks = parse_document(str(file_path))
            if settings.enable_vision_enrichment:
                blocks = enrich_blocks_with_vision(blocks)
                vision_calls_used = get_last_vision_calls_used()

            chunks = chunk_structured_blocks(blocks)
            parsing_method = "docling"
            logger.info(
                "Structured ingestion completed | file=%s blocks=%d chunks=%d vision_calls_used=%d",
                filename,
                len(blocks),
                len(chunks),
                vision_calls_used,
            )
        except Exception as docling_exc:
            parsing_method = "legacy_pdf_fallback"
            logger.warning(
                "Structured ingestion failed; falling back to legacy parser | file=%s error=%s",
                filename,
                docling_exc,
            )

    if not chunks:
        documents = load_pdf(file_path)
        chunks = split_documents(documents)

    if not chunks:
        raise ValueError("Document contains no parseable text.")

    indexed_at = int(time.time())
    for idx, chunk in enumerate(chunks):
        chunk.metadata.update(
            {
                "source": filename,
                "file_hash": file_hash,
                "document_id": document_id,
                "parsing_method": parsing_method,
                "chunk_index": idx,
                "indexed_at": indexed_at,
            }
        )

    return chunks, parsing_method, vision_calls_used


@router.get(
    "/knowledge-base/files",
    response_model=KnowledgeBaseFilesResponse,
    status_code=status.HTTP_200_OK,
    summary="List indexed files",
    description="Returns persisted indexed document metadata for UI hydration on reload.",
)
async def list_knowledge_base_files() -> KnowledgeBaseFilesResponse:
    files = list_indexed_documents()
    return KnowledgeBaseFilesResponse(files=files)


@router.get(
    "/knowledge-base/files/{file_hash}/chunks",
    response_model=DocumentChunksResponse,
    status_code=status.HTTP_200_OK,
    summary="Inspect indexed chunks for one file",
    description="Returns stored chunk text and metadata for source inspection/debugging.",
)
async def list_knowledge_base_file_chunks(file_hash: str) -> DocumentChunksResponse:
    meta = get_indexed_document(file_hash)
    if meta is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )
    chunks = list_document_chunks(file_hash)
    return DocumentChunksResponse(
        file_hash=file_hash,
        filename=str(meta.get("filename", "")),
        chunks=chunks,
    )


@router.post(
    "/knowledge-base/reset",
    response_model=ResetKnowledgeBaseResponse,
    status_code=status.HTTP_200_OK,
    summary="Reset knowledge base",
    description="Deletes uploaded PDFs and clears persisted vector index data.",
)
async def reset_knowledge_base() -> ResetKnowledgeBaseResponse:
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
async def delete_knowledge_base_file(file_hash: str) -> DeleteKnowledgeBaseFileResponse:
    try:
        meta = get_indexed_document(file_hash)
        if meta is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found.",
            )

        delete_result = delete_indexed_document(file_hash)
        upload_deleted = _delete_stored_upload(
            file_hash=file_hash,
            filename=str(meta.get("filename", "")),
            upload_path=str(meta.get("upload_path", "")),
        )
        clear_query_cache()
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
async def reindex_knowledge_base_file(file_hash: str) -> UploadItemResult:
    meta = get_indexed_document(file_hash)
    if meta is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

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
        chunks, parsing_method, vision_calls_used = _prepare_chunks_for_indexing(
            file_path=stored_path,
            filename=filename,
            file_hash=file_hash,
            document_id=document_id,
        )
        delete_indexed_document(file_hash)
        add_documents(chunks)
        register_indexed_document(
            file_hash=file_hash,
            filename=filename,
            chunk_count=len(chunks),
            document_id=document_id,
            parsing_method=parsing_method,
            upload_path=str(stored_path),
            upload_status="indexed",
            vision_calls_used=vision_calls_used,
        )
        clear_query_cache()
        return UploadItemResult(
            filename=filename,
            chunks_indexed=len(chunks),
            status="success",
            message="Document reindexed successfully.",
            file_hash=file_hash,
            document_id=document_id,
            parsing_method=parsing_method,
            vision_calls_used=vision_calls_used,
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
async def upload_document(files: list[UploadFile] = File(...)) -> UploadBatchResponse:
    """Ingest one or more PDFs into the RAG knowledge base."""

    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files received. Please upload at least one .pdf file.",
        )

    start_time = time.perf_counter()
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    results: list[UploadItemResult] = []
    total_chunks_indexed = 0

    for file in files:
        file_start = time.perf_counter()
        safe_filename = safe_pdf_filename(file.filename)
        if safe_filename is None:
            results.append(
                UploadItemResult(
                    filename=file.filename or "unknown",
                    chunks_indexed=0,
                    status="failed",
                    message="Only PDF files are supported. Please upload a .pdf file.",
                )
            )
            continue

        logger.info("Upload received: '%s'", file.filename)

        content = b""
        file_hash = ""
        document_id = ""
        file_path: Path | None = None
        try:
            content = await file.read()
            validation_error = validate_pdf_upload(
                filename=file.filename,
                content_type=file.content_type,
                content=content,
                max_upload_size_bytes=settings.max_upload_size_bytes,
                max_upload_size_mb=settings.max_upload_size_mb,
            )
            if validation_error:
                results.append(
                    UploadItemResult(
                        filename=file.filename or "unknown",
                        chunks_indexed=0,
                        status="failed",
                        message=validation_error,
                    )
                )
                continue

            file_hash = hashlib.sha256(content).hexdigest()
            document_id = hashlib.sha1(f"{safe_filename}:{file_hash}".encode("utf-8")).hexdigest()[:16]
            file_path = _stored_upload_path(file_hash, safe_filename)
            with open(file_path, "wb") as f:
                f.write(content)
        except Exception as e:
            logger.error("Failed to save file '%s': %s", file.filename, e)
            results.append(
                UploadItemResult(
                    filename=file.filename,
                    chunks_indexed=0,
                    status="failed",
                    message=f"Could not save uploaded file: {e}",
                )
            )
            continue

        if file_hash and is_document_indexed(file_hash):
            existing = get_indexed_document(file_hash) or {}
            logger.info("Duplicate upload skipped for '%s' (hash=%s)", file.filename, file_hash[:12])
            if file_path is not None and file_path.exists():
                os.remove(file_path)
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
                )
            )
            continue

        try:
            chunking_start = time.perf_counter()
            chunks, parsing_method, vision_calls_used = _prepare_chunks_for_indexing(
                file_path=file_path,
                filename=safe_filename,
                file_hash=file_hash,
                document_id=document_id,
            )
            chunking_elapsed = time.perf_counter() - chunking_start

            index_start = time.perf_counter()
            add_documents(chunks)
            index_elapsed = time.perf_counter() - index_start
            if file_hash:
                register_indexed_document(
                    file_hash=file_hash,
                    filename=safe_filename,
                    chunk_count=len(chunks),
                    document_id=document_id,
                    parsing_method=parsing_method,
                    upload_path=str(file_path),
                    upload_status="indexed",
                    vision_calls_used=vision_calls_used,
                )

            total_chunks_indexed += len(chunks)
            total_file_elapsed = time.perf_counter() - file_start
            logger.info(
                "File indexing complete | file=%s chunks=%d chunking_time_s=%.3f embed_faiss_time_s=%.3f total_time_s=%.3f",
                file.filename,
                len(chunks),
                chunking_elapsed,
                index_elapsed,
                total_file_elapsed,
            )
            results.append(
                UploadItemResult(
                    filename=safe_filename,
                    chunks_indexed=len(chunks),
                    status="success",
                    message="Document ingested successfully.",
                    file_hash=file_hash,
                    document_id=document_id,
                    parsing_method=parsing_method,
                    vision_calls_used=vision_calls_used,
                )
            )
        except Exception as e:
            logger.error("Ingestion failed for '%s': %s", file.filename, e)
            if file_path is not None and file_path.exists():
                os.remove(file_path)
            results.append(
                UploadItemResult(
                    filename=safe_filename,
                    chunks_indexed=0,
                    status="failed",
                    message=f"Document ingestion failed: {e}",
                )
            )

    processed_files = sum(1 for item in results if item.status in {"success", "duplicate"})
    elapsed = time.perf_counter() - start_time
    logger.info(
        "Batch ingest completed in %.2fs | total_files=%d processed_files=%d total_chunks_indexed=%d",
        elapsed,
        len(files),
        processed_files,
        total_chunks_indexed,
    )

    return UploadBatchResponse(
        files=results,
        total_files=len(files),
        processed_files=processed_files,
        total_chunks_indexed=total_chunks_indexed,
        message=f"Upload completed. Processed {processed_files}/{len(files)} files.",
    )
