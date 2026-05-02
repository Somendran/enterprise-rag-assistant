"""FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import auth, upload, query, ops
from app.config import settings
from app.services.vector_store import load_store
from app.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup / shutdown lifecycle hook.

    On startup: try to pre-load the FAISS index from disk so the first
    query doesn't incur the cold-start cost of reading it from disk.
    """
    logger.info("Starting Enterprise RAG Assistant...")
    store = load_store()
    if store is not None:
        logger.info("FAISS index loaded successfully from disk.")
    else:
        logger.info("No existing FAISS index found. Ready to receive uploads.")
    yield
    logger.info("Shutting down Enterprise RAG Assistant.")


app = FastAPI(
    title="Enterprise Internal Knowledge Assistant",
    description=(
        "A RAG-based AI Copilot that lets employees ask questions about "
        "internal company documents. Upload PDFs and get grounded answers."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key", "X-Demo-Session-Id"],
)

# Routers
app.include_router(upload.router, tags=["Ingestion"])
app.include_router(query.router, tags=["Retrieval & Generation"])
app.include_router(ops.router, tags=["Ops"])
app.include_router(auth.router)


# Health check
@app.get("/health", tags=["Ops"], summary="Health check")
async def health_check():
    """Returns 200 if the service is running."""
    return {"status": "ok", "service": "Enterprise RAG Assistant"}
