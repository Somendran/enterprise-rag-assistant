"""
schemas.py
──────────
Pydantic models that define the shape of every API request and response.
Keeping these in one place makes the contract between client and server
explicit and easy to change.
"""

from pydantic import BaseModel, Field
from typing import Any, List, Optional


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
    ocr_applied: bool = Field(default=False, description="Whether OCR was used during ingestion.")
    text_coverage_ratio: float = Field(default=0.0, description="Share of pages with enough extractable text.")
    low_text_pages: int = Field(default=0, description="Pages with very little extractable text.")
    ingestion_warnings: List[str] = Field(default_factory=list, description="Non-fatal ingestion quality warnings.")


class UploadBatchResponse(BaseModel):
    """Returned after processing one or more uploaded PDFs."""

    files: List[UploadItemResult] = Field(default_factory=list)
    total_files: int = Field(..., description="Total files received in this request.")
    processed_files: int = Field(..., description="Files processed successfully or as duplicates.")
    total_chunks_indexed: int = Field(..., description="Total chunks indexed across successful files.")
    message: str = Field(default="Upload completed.")


class IngestionJobResponse(BaseModel):
    """Returned when a background ingestion job is created."""

    job_id: str
    status: str
    stage: str = ""
    message: str = ""


class IngestionJobStatusResponse(BaseModel):
    """Current status for a background ingestion job."""

    job_id: str
    status: str
    stage: str = ""
    message: str = ""
    total_files: int = 0
    processed_files: int = 0
    total_chunks_indexed: int = 0
    results: List[UploadItemResult] = Field(default_factory=list)
    created_at: int = 0
    updated_at: int = 0


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
    owner_user_id: str = Field(default="", description="User id that uploaded or owns this document.")
    visibility: str = Field(default="shared", description="shared, private, or role.")
    allowed_roles: List[str] = Field(default_factory=list, description="Roles that can access role-restricted docs.")
    ocr_applied: bool = Field(default=False, description="Whether OCR was used during ingestion.")
    text_coverage_ratio: float = Field(default=0.0, description="Share of pages with enough extractable text.")
    low_text_pages: int = Field(default=0, description="Pages with very little extractable text.")
    ingestion_warnings: List[str] = Field(default_factory=list, description="Non-fatal ingestion quality warnings.")


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


class DocumentChunkItem(BaseModel):
    """Inspectable chunk stored for one document."""

    id: str = Field(..., description="Internal vector-store document id.")
    content: str = Field(..., description="Chunk text.")
    page: int = Field(default=0, description="Source page.")
    section: str = Field(default="", description="Detected section or heading.")
    chunk_index: int = Field(default=0, description="Chunk order within the document.")
    quality_score: float = Field(default=0.0, description="Heuristic quality score in range [0, 1].")
    quality_warnings: List[str] = Field(default_factory=list, description="Chunk inspection warnings.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Raw chunk metadata.")


class DocumentChunksResponse(BaseModel):
    """Returned when inspecting stored chunks for one document."""

    file_hash: str = Field(..., description="Document hash.")
    filename: str = Field(default="", description="Document filename.")
    chunks: List[DocumentChunkItem] = Field(default_factory=list)
    focus_chunk_index: Optional[int] = None


class DocumentPermissionsUpdateRequest(BaseModel):
    visibility: str = Field(default="shared", description="shared, private, or role.")
    allowed_roles: List[str] = Field(default_factory=list)


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

    file_hash: Optional[str] = Field(default=None, description="Source document hash.")
    document: str = Field(..., description="Source filename (PDF name).")
    page: int = Field(..., description="Page number within the PDF (1-indexed).")
    chunk_index: Optional[int] = Field(default=None, description="Chunk index within the source document.")
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
    metadata: dict[str, Any] = Field(default_factory=dict, description="Raw chunk metadata used for citation.")


class RetrievalDiagnostics(BaseModel):
    """Debug-friendly retrieval diagnostics for quality tuning and observability."""

    query_variants_used: List[str] = Field(default_factory=list)
    query_type: str = Field(default="general")
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


class FeedbackRequest(BaseModel):
    """User feedback attached to an answer."""

    question: str = Field(..., min_length=1, max_length=2000)
    answer: str = Field(..., min_length=1, max_length=20000)
    rating: str = Field(..., description="helpful, not_helpful, wrong_source, or missing_info.")
    reason: str = Field(default="", max_length=200)
    comment: str = Field(default="", max_length=2000)
    confidence_score: Optional[float] = None
    sources: List[SourceReference] = Field(default_factory=list)
    diagnostics: Optional[RetrievalDiagnostics] = None


class FeedbackResponse(BaseModel):
    """Stored feedback response."""

    id: int
    created_at: int
    rating: str
    reason: str = ""
    comment: str = ""
    message: str = "Feedback recorded."


class ModelHealthItem(BaseModel):
    """One runtime dependency health check."""

    name: str
    status: str
    detail: str = ""


class ModelHealthResponse(BaseModel):
    """Runtime model and dependency health."""

    checks: List[ModelHealthItem] = Field(default_factory=list)


class AdminOverviewResponse(BaseModel):
    """Summary for local admin/debug panel."""

    document_count: int
    chunk_count: int
    feedback_count: int
    query_diagnostic_count: int = 0
    chat_session_count: int = 0
    eval_run_count: int = 0
    user_count: int = 0
    audit_event_count: int = 0
    recent_feedback: List[dict[str, Any]] = Field(default_factory=list)
    metadata_db_path: str
    embedding_model: str
    embedding_device: str
    docling_enabled: bool
    reranker_enabled: bool
    openai_enabled: bool


class ChatSessionCreateRequest(BaseModel):
    title: str = Field(default="New chat", max_length=120)


class ChatSessionItem(BaseModel):
    id: str
    title: str
    created_at: int
    updated_at: int


class ChatSessionsResponse(BaseModel):
    sessions: List[ChatSessionItem] = Field(default_factory=list)


class ChatMessageRequest(BaseModel):
    id: Optional[str] = None
    role: str
    content: str
    sources: List[SourceReference] = Field(default_factory=list)
    diagnostics: Optional[RetrievalDiagnostics] = None
    confidence_score: Optional[float] = None
    confidence_level: Optional[str] = None


class ChatMessageItem(BaseModel):
    id: str
    session_id: str
    role: str
    content: str
    created_at: int
    sources: List[SourceReference] = Field(default_factory=list)
    diagnostics: Optional[RetrievalDiagnostics] = None
    confidence_score: Optional[float] = None
    confidence_level: Optional[str] = None


class ChatMessagesResponse(BaseModel):
    messages: List[ChatMessageItem] = Field(default_factory=list)


class EvalRunCreateResponse(BaseModel):
    run_id: str
    status: str
    total: int = 0


class EvalRunResultItem(BaseModel):
    eval_id: str
    passed: bool
    message: str


class EvalRunItem(BaseModel):
    id: str
    created_at: int
    status: str
    total: int = 0
    passed: int = 0
    failed: int = 0
    message: str = ""
    results: List[EvalRunResultItem] = Field(default_factory=list)


class EvalRunsResponse(BaseModel):
    runs: List[EvalRunItem] = Field(default_factory=list)


class AuthStatusResponse(BaseModel):
    auth_enabled: bool
    has_users: bool
    bootstrap_required: bool


class AuthBootstrapRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=254)
    password: str = Field(..., min_length=8, max_length=200)
    display_name: str = Field(default="", max_length=120)


class AuthLoginRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=254)
    password: str = Field(..., min_length=1, max_length=200)


class UserItem(BaseModel):
    id: str
    email: str
    display_name: str = ""
    role: str
    disabled: int = 0
    created_at: int
    updated_at: int


class AuthTokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_at: int
    user: UserItem


class CurrentUserResponse(BaseModel):
    user: UserItem


class UserCreateRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=254)
    password: str = Field(..., min_length=8, max_length=200)
    display_name: str = Field(default="", max_length=120)
    role: str = Field(default="user", description="admin or user")


class UsersResponse(BaseModel):
    users: List[UserItem] = Field(default_factory=list)


class AuditEventItem(BaseModel):
    id: int
    created_at: int
    actor_user_id: str = ""
    actor_email: str = ""
    action: str
    resource_type: str = ""
    resource_id: str = ""
    detail: dict[str, Any] = Field(default_factory=dict)


class AuditEventsResponse(BaseModel):
    events: List[AuditEventItem] = Field(default_factory=list)
