"""
schemas.py
──────────
Pydantic models that define the shape of every API request and response.
Keeping these in one place makes the contract between client and server
explicit and easy to change.
"""

from pydantic import BaseModel, Field
from typing import List


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


class QueryResponse(BaseModel):
    """Full structured answer returned to the client."""

    answer: str = Field(..., description="LLM-generated answer grounded in documents.")
    sources: List[SourceReference] = Field(
        default_factory=list,
        description="List of document chunks that informed the answer.",
    )
