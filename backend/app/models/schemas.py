"""
schemas.py
──────────
Pydantic models that define the shape of every API request and response.
Keeping these in one place makes the contract between client and server
explicit and easy to change.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


# ── /upload ──────────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    """Returned after a PDF is successfully ingested."""

    filename: str = Field(..., description="Original name of the uploaded file.")
    chunks_indexed: int = Field(
        ..., description="Number of text chunks stored in the vector store."
    )
    message: str = Field(default="Document ingested successfully.")


class UploadItemResult(BaseModel):
    """Per-file ingestion outcome in a batch upload."""

    filename: str = Field(..., description="Original uploaded filename.")
    chunks_indexed: int = Field(..., description="Number of chunks indexed for this file.")
    status: str = Field(..., description="success, duplicate, or failed.")
    message: str = Field(..., description="Human-readable outcome message for this file.")
    file_hash: Optional[str] = Field(default=None, description="SHA-256 file hash for document operations.")
    document_id: Optional[str] = Field(default=None, description="Stable internal document id.")
    parsing_method: Optional[str] = Field(default=None, description="Parser path used for ingestion.")
    vision_calls_used: int = Field(default=0, description="Vision enrichment calls used during ingestion.")


class UploadBatchResponse(BaseModel):
    """Returned after processing one or more uploaded PDFs."""

    files: List[UploadItemResult] = Field(default_factory=list)
    total_files: int = Field(..., description="Total files received in this request.")
    processed_files: int = Field(..., description="Files processed successfully or as duplicates.")
    total_chunks_indexed: int = Field(..., description="Total chunks indexed across successful files.")
    message: str = Field(default="Upload completed.")


class ResetKnowledgeBaseResponse(BaseModel):
    """Returned after knowledge base reset completes."""

    message: str = Field(..., description="Result message for reset action.")
    index_cleared: bool = Field(..., description="Whether FAISS index data was removed.")
    uploads_deleted: int = Field(..., description="Number of uploaded files deleted.")


class KnowledgeBaseFileItem(BaseModel):
    """Indexed document metadata used by sidebar file list."""

    file_hash: str = Field(..., description="SHA-256 file hash used as document key.")
    document_id: str = Field(default="", description="Stable internal document id.")
    filename: str = Field(..., description="Indexed document filename.")
    chunk_count: int = Field(..., description="Number of indexed chunks for the document.")
    indexed_at: int = Field(..., description="Unix timestamp when document was indexed.")
    parsing_method: str = Field(default="unknown", description="Parser path used for ingestion.")
    upload_status: str = Field(default="indexed", description="Current document status.")
    vision_calls_used: int = Field(default=0, description="Vision enrichment calls used during ingestion.")
    embedding_model: str = Field(default="", description="Embedding model used for this document.")


class KnowledgeBaseFilesResponse(BaseModel):
    """List of indexed files currently present in the registry."""

    files: List[KnowledgeBaseFileItem] = Field(default_factory=list)


class DeleteKnowledgeBaseFileResponse(BaseModel):
    """Returned after deleting a single indexed document."""

    file_hash: str = Field(..., description="Deleted document hash.")
    filename: str = Field(default="", description="Deleted document filename.")
    chunks_deleted: int = Field(default=0, description="Number of vector chunks removed.")
    upload_deleted: bool = Field(default=False, description="Whether the stored PDF was deleted.")
    message: str = Field(..., description="Human-readable result message.")


# ── /query ───────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Incoming question from the user."""

    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="The question to answer using the knowledge base.",
        examples=["What is our remote-work policy?"],
    )


class SourceReference(BaseModel):
    """Points to the exact document chunk used to generate part of the answer."""

    document: str = Field(..., description="Source filename (PDF name).")
    page: int = Field(..., description="Page number within the PDF (1-indexed).")
    relevance_score: Optional[float] = Field(
        default=None,
        description="Normalized relevance confidence in range [0, 1].",
    )
    snippet: Optional[str] = Field(
        default=None,
        description="Short excerpt from the retrieved chunk for quick context preview.",
    )
    section: Optional[str] = Field(default=None, description="Detected section or heading for the chunk.")
    vector_score: Optional[float] = Field(default=None, description="Vector similarity confidence.")
    lexical_score: Optional[float] = Field(default=None, description="Lexical overlap score.")
    bm25_score: Optional[float] = Field(default=None, description="BM25-style score.")
    final_score: Optional[float] = Field(default=None, description="Final score after reranking or blending.")
    reranker_applied: Optional[bool] = Field(default=None, description="Whether neural reranking affected ranking.")


class RetrievalDiagnostics(BaseModel):
    """Debug-friendly retrieval diagnostics for quality tuning and observability."""

    query_variants_used: List[str] = Field(default_factory=list)
    is_broad_question: bool = Field(default=False)
    is_simple_query: bool = Field(default=False)
    fast_mode_applied: bool = Field(default=False)
    fallback_applied: bool = Field(default=False)
    candidates_considered: int = Field(default=0)
    reranker_applied: bool = Field(default=False)
    reranker_skipped_reason: str = Field(default="")
    retrieval_ms: float = Field(default=0.0)
    rerank_ms: float = Field(default=0.0)
    context_build_ms: float = Field(default=0.0)
    generation_ms: float = Field(default=0.0)
    total_pipeline_ms: float = Field(default=0.0)
    llm_retry_count: int = Field(default=0)
    llm_retry_reason: str = Field(default="")
    normalization_applied: bool = Field(default=False)
    low_confidence_fallback_used: bool = Field(default=False)
    verification_enabled: bool = Field(default=False)
    verification_applied: bool = Field(default=False)
    verification_skipped_reason: str = Field(default="")
    verification_failed: bool = Field(default=False)
    verification_ms: float = Field(default=0.0)
    claims_total: int = Field(default=0)
    claims_verified: int = Field(default=0)
    citation_coverage: float = Field(default=0.0)
    invalid_citations: List[str] = Field(default_factory=list)
    unsupported_claims: int = Field(default=0)


class QueryResponse(BaseModel):
    """Full structured answer returned to the client."""

    answer: str = Field(..., description="LLM-generated answer grounded in documents.")
    sources: List[SourceReference] = Field(
        default_factory=list,
        description="List of document chunks that informed the answer.",
    )
    confidence_score: Optional[float] = Field(
        default=None,
        description="Overall normalized grounding confidence in range [0, 1].",
    )
    confidence_level: Optional[str] = Field(
        default=None,
        description="Confidence bucket derived from confidence_score: high, medium, low.",
    )
    diagnostics: Optional[RetrievalDiagnostics] = Field(
        default=None,
        description="Retrieval diagnostics emitted when enabled in configuration.",
    )
