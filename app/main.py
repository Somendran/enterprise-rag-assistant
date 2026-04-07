"""
main.py
───────
FastAPI application entry point.

Responsibilities:
- Create and configure the FastAPI app.
- Register all API routers.
- Mount startup logic (e.g. pre-loading the FAISS index from disk).
- Provide a health-check endpoint.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import upload, query
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

# ── CORS ──────────────────────────────────────────────────────────────────────
# For backend-only usage this doesn't matter much, but it's good practice
# to configure it explicitly so adding a frontend later is painless.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Restrict to specific origins in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(upload.router, tags=["Ingestion"])
app.include_router(query.router, tags=["Retrieval & Generation"])


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["Ops"], summary="Health check")
async def health_check():
    """Returns 200 if the service is running."""
    return {"status": "ok", "service": "Enterprise RAG Assistant"}
