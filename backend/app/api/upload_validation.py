"""Pure upload validation helpers."""

from pathlib import Path

PDF_MAGIC = b"%PDF"
ALLOWED_PDF_CONTENT_TYPES = {
    "application/pdf",
    "application/x-pdf",
    "application/octet-stream",
}


def safe_pdf_filename(filename: str | None) -> str | None:
    if not filename:
        return None

    normalized = filename.strip()
    if not normalized or "/" in normalized or "\\" in normalized:
        return None

    safe_name = Path(normalized).name
    if safe_name != normalized or safe_name in {".", ".."}:
        return None

    if not safe_name.lower().endswith(".pdf"):
        return None

    return safe_name


def validate_pdf_upload(
    *,
    filename: str | None,
    content_type: str | None,
    content: bytes,
    max_upload_size_bytes: int,
    max_upload_size_mb: int,
) -> str | None:
    safe_name = safe_pdf_filename(filename)
    if safe_name is None:
        return "Only simple .pdf filenames are supported."

    if len(content) > max_upload_size_bytes:
        return f"PDF is too large. Maximum size is {max_upload_size_mb} MB."

    normalized_content_type = (content_type or "").lower().strip()
    if normalized_content_type and normalized_content_type not in ALLOWED_PDF_CONTENT_TYPES:
        return "Only PDF files are supported."

    if not content.startswith(PDF_MAGIC):
        return "Uploaded file does not look like a valid PDF."

    return None
