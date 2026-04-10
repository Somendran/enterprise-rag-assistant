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
