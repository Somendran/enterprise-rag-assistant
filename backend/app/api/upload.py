"""
upload.py
─────────
POST /upload

Accepts a PDF file, runs the full ingestion pipeline
(load → split → embed → store), and returns the number of chunks indexed.
"""

import os
import time
import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, status

from app.services.document_loader import load_pdf
from app.services.text_splitter import split_documents
from app.services.query_cache import clear_query_cache
from app.services.vector_store import (
    add_documents,
    is_document_indexed,
    register_indexed_document,
    reset_vector_store,
)
from app.models.schemas import (
    UploadBatchResponse,
    UploadItemResult,
    ResetKnowledgeBaseResponse,
)
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


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
        if not file.filename or not file.filename.lower().endswith(".pdf"):
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

        file_path = upload_dir / file.filename
        try:
            content = await file.read()
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

        try:
            documents = load_pdf(file_path)
            file_hash = str(documents[0].metadata.get("file_hash", "")) if documents else ""
            document_id = str(documents[0].metadata.get("document_id", "")) if documents else ""

            if file_hash and is_document_indexed(file_hash):
                logger.info("Duplicate upload skipped for '%s' (hash=%s)", file.filename, file_hash[:12])
                if file_path.exists():
                    os.remove(file_path)
                results.append(
                    UploadItemResult(
                        filename=file.filename,
                        chunks_indexed=0,
                        status="duplicate",
                        message="Duplicate document detected. Existing index entry was reused.",
                    )
                )
                continue

            chunks = split_documents(documents)
            if not chunks:
                raise ValueError("Document contains no parseable text.")

            add_documents(chunks)
            if file_hash:
                register_indexed_document(
                    file_hash=file_hash,
                    filename=file.filename,
                    chunk_count=len(chunks),
                    document_id=document_id,
                )

            total_chunks_indexed += len(chunks)
            results.append(
                UploadItemResult(
                    filename=file.filename,
                    chunks_indexed=len(chunks),
                    status="success",
                    message="Document ingested successfully.",
                )
            )
        except Exception as e:
            logger.error("Ingestion failed for '%s': %s", file.filename, e)
            if file_path.exists():
                os.remove(file_path)
            results.append(
                UploadItemResult(
                    filename=file.filename,
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
