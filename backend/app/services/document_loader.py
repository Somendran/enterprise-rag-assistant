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
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from typing import List

from app.utils.logger import get_logger

logger = get_logger(__name__)


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

    logger.info(f"Loaded {len(documents)} page(s) from '{path.name}'")
    return documents
