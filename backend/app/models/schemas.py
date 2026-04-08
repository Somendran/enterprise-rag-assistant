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
    fallback_applied: bool = Field(default=False)
    candidates_considered: int = Field(default=0)


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
