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
from app.models.schemas import UploadResponse, ResetKnowledgeBaseResponse
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
    response_model=UploadResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload and index a PDF document",
    description=(
        "Accepts a PDF file, extracts its text, splits it into chunks, "
        "embeds each chunk, and stores them in the vector store. "
        "Returns the number of chunks successfully indexed."
    ),
)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    """Ingest a PDF into the RAG knowledge base."""

    # ── Validate file type ────────────────────────────────────────────────────
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported. Please upload a .pdf file.",
        )

    start_time = time.perf_counter()
    logger.info(f"Upload received: '{file.filename}'")

    # ── Save file to disk ─────────────────────────────────────────────────────
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_path = upload_dir / file.filename
    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
    except Exception as e:
        logger.error(f"Failed to save file '{file.filename}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not save uploaded file: {str(e)}",
        )

    # ── Run ingestion pipeline ────────────────────────────────────────────────
    try:
        # 1. Load PDF pages into LangChain Documents
        documents = load_pdf(file_path)
        file_hash = str(documents[0].metadata.get("file_hash", "")) if documents else ""
        document_id = str(documents[0].metadata.get("document_id", "")) if documents else ""

        if file_hash and is_document_indexed(file_hash):
            logger.info("Duplicate upload skipped for '%s' (hash=%s)", file.filename, file_hash[:12])
            if file_path.exists():
                os.remove(file_path)
            return UploadResponse(
                filename=file.filename,
                chunks_indexed=0,
                message="Duplicate document detected. Existing index entry was reused.",
            )

        # 2. Split pages into overlapping chunks
        chunks = split_documents(documents)

        if not chunks:
            raise ValueError("Document contains no parseable text.")

        # 3. Embed chunks and store in FAISS
        add_documents(chunks)
        if file_hash:
            register_indexed_document(
                file_hash=file_hash,
                filename=file.filename,
                chunk_count=len(chunks),
                document_id=document_id,
            )

    except Exception as e:
        logger.error(f"Ingestion failed for '{file.filename}': {e}")
        # Clean up the saved file if ingestion fails to keep uploads dir clean
        if file_path.exists():
            os.remove(file_path)

        message = str(e)
        if "quota" in message.lower() or "429" in message:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Document ingestion throttled by Gemini quota: {message}",
            )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document ingestion failed: {message}",
        )

    elapsed = time.perf_counter() - start_time
    logger.info(
        f"Ingested '{file.filename}' → {len(chunks)} chunk(s) in {elapsed:.2f}s"
    )

    return UploadResponse(
        filename=file.filename,
        chunks_indexed=len(chunks),
    )
