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
import re
import pdfplumber
from langchain.schema import Document
from typing import List

from app.utils.logger import get_logger

logger = get_logger(__name__)


def _to_ascii_compact(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\ufffd", " ").replace("�", " ")
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _is_heading_like(line: str) -> bool:
    line = line.strip()
    if len(line) < 4 or len(line) > 120:
        return False
    has_numeric_prefix = bool(re.match(r"^\d+(?:\.\d+)*\s+", line))
    upper_ratio = sum(ch.isupper() for ch in line) / max(1, sum(ch.isalpha() for ch in line))
    looks_title_case = bool(re.match(r"^[A-Z][A-Za-z0-9&/(),\-\s]{3,}$", line))
    return has_numeric_prefix or upper_ratio > 0.55 or looks_title_case


def _extract_header_hints(page_text: str, max_hints: int = 3) -> list[str]:
    hints: list[str] = []
    for raw in page_text.splitlines()[:20]:
        line = raw.strip()
        if not line:
            continue
        if _is_heading_like(line):
            hints.append(line)
        if len(hints) >= max_hints:
            break
    return hints


def _table_to_markdown(table: list[list[object]]) -> str:
    # Convert extracted rows into compact markdown-ish table text for retrieval.
    rows: list[list[str]] = []
    for row in table:
        cells = [str(cell).strip() if cell is not None else "" for cell in row]
        if any(cells):
            rows.append(cells)

    if not rows:
        return ""

    width = max(len(r) for r in rows)
    normalized = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(r) + " |" for r in normalized]
    return "\n".join(lines)


def _extract_page_content_with_layout(page) -> tuple[str, list[str], bool]:
    text = page.extract_text() or ""
    text = _to_ascii_compact(text)
    headers = _extract_header_hints(text)

    table_blocks: list[str] = []
    try:
        for table in page.extract_tables() or []:
            rendered = _table_to_markdown(table)
            if rendered:
                table_blocks.append(rendered)
    except Exception:
        # Keep ingestion resilient for malformed tables.
        pass

    has_tables = len(table_blocks) > 0

    if has_tables:
        text = (
            f"{text}\n\n[STRUCTURED_TABLES]\n"
            + "\n\n".join(table_blocks)
        ).strip()

    return text, headers, has_tables


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

    documents: list[Document] = []
    with pdfplumber.open(str(path)) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            page_text, header_hints, has_tables = _extract_page_content_with_layout(page)
            documents.append(
                Document(
                    page_content=page_text,
                    metadata={
                        "source": path.name,
                        "page": page_idx,
                        "file_hash": file_hash,
                        "document_id": document_id,
                        "uploaded_at": uploaded_at,
                        "header_hints": header_hints,
                        "has_tables": has_tables,
                    },
                )
            )

    # Remove fully empty pages to avoid polluting retrieval candidates.
    documents = [doc for doc in documents if doc.page_content.strip()]

    logger.info(f"Loaded {len(documents)} page(s) from '{path.name}'")
    return documents
