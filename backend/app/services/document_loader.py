"""
document_loader.py
───────────────────
Responsibility: Load a PDF from disk and return a list of LangChain Document objects.

Each Document has:
  - page_content : the raw text of one page
  - metadata     : {"source": <filename>, "page": <page_number>}

We deliberately keep this thin — one function, one job.
"""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from typing import List

from app.utils.logger import get_logger

logger = get_logger(__name__)


def compute_file_hash(file_path: str | Path) -> str:
    """Compute a stable SHA-256 hex digest for duplicate detection."""
    path = Path(file_path)
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_pdf(file_path: str | Path) -> List[Document]:
    """
    Load a PDF file and return its pages as LangChain Documents.

    Args:
        file_path: Absolute or relative path to the PDF on disk.

    Returns:
        List of Documents, one per page, each carrying source metadata.

    Raises:
        FileNotFoundError: If the PDF doesn't exist at the given path.
        ValueError: If the file is not a PDF (basic extension check).
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path.suffix}")

    logger.info(f"Loading PDF: {path.name}")
    file_hash = compute_file_hash(path)
    uploaded_at = datetime.now(timezone.utc).isoformat()
    document_id = hashlib.sha1(f"{path.name}:{file_hash}".encode("utf-8")).hexdigest()[:16]

    loader = PyPDFLoader(str(path))
    documents = loader.load()

    # Normalise metadata so every document has a clean 'source' key
    # PyPDFLoader already sets metadata['source'] and metadata['page'],
    # but we rename 'source' to just the filename (not the full path)
    # to avoid leaking internal filesystem structure to API clients.
    for doc in documents:
        doc.metadata["source"] = path.name
        # PyPDFLoader uses 0-based page numbers; convert to 1-based for humans.
        doc.metadata["page"] = doc.metadata.get("page", 0) + 1
        doc.metadata["file_hash"] = file_hash
        doc.metadata["document_id"] = document_id
        doc.metadata["uploaded_at"] = uploaded_at

    logger.info(f"Loaded {len(documents)} page(s) from '{path.name}'")
    return documents
