"""Ingestion quality checks for deciding when OCR is needed."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re

import pdfplumber

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class IngestionQuality:
    total_pages: int = 0
    text_coverage_ratio: float = 0.0
    low_text_pages: int = 0
    ocr_recommended: bool = False
    warnings: list[str] = field(default_factory=list)


def _visible_text_length(text: str) -> int:
    """Count meaningful extracted characters, ignoring whitespace noise."""
    return len(re.sub(r"\s+", "", text or ""))


def summarize_block_text_quality(blocks: list[dict]) -> tuple[float, int]:
    """Return page coverage and low-text page count for parsed blocks."""
    by_page: dict[int, int] = {}
    for block in blocks:
        try:
            page = int(block.get("page", 0) or 0)
        except Exception:
            page = 0
        text = str(block.get("content", "") or "")
        visual = str(block.get("visual_description", "") or "")
        by_page[page] = by_page.get(page, 0) + _visible_text_length(text + visual)

    if not by_page:
        return 0.0, 0

    low_text_limit = max(1, int(settings.ingestion_low_text_page_chars))
    low_pages = sum(1 for length in by_page.values() if length < low_text_limit)
    coverage = 1.0 - (low_pages / max(1, len(by_page)))
    return round(max(0.0, min(1.0, coverage)), 4), low_pages


def assess_pdf_text_quality(file_path: str | Path) -> IngestionQuality:
    """Inspect native PDF text extraction to identify likely scanned documents."""
    path = Path(file_path)
    low_text_limit = max(1, int(settings.ingestion_low_text_page_chars))
    ocr_ratio_threshold = max(0.0, min(1.0, float(settings.ingestion_ocr_page_ratio_threshold)))

    page_lengths: list[int] = []
    warnings: list[str] = []
    try:
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                page_lengths.append(_visible_text_length(page.extract_text() or ""))
    except Exception as exc:
        logger.warning("Could not assess PDF text quality for '%s': %s", path.name, exc)
        return IngestionQuality(
            warnings=[f"Could not assess extracted text quality: {exc}"],
        )

    total_pages = len(page_lengths)
    if total_pages == 0:
        return IngestionQuality(
            total_pages=0,
            warnings=["PDF has no readable pages."],
            ocr_recommended=True,
        )

    low_text_pages = sum(1 for length in page_lengths if length < low_text_limit)
    low_text_ratio = low_text_pages / max(1, total_pages)
    coverage = 1.0 - low_text_ratio
    ocr_recommended = low_text_ratio >= ocr_ratio_threshold

    if low_text_pages:
        warnings.append(
            f"{low_text_pages}/{total_pages} page(s) have very little extractable text."
        )
    if ocr_recommended:
        warnings.append("Document appears to be scanned; OCR ingestion is recommended.")

    return IngestionQuality(
        total_pages=total_pages,
        text_coverage_ratio=round(max(0.0, min(1.0, coverage)), 4),
        low_text_pages=low_text_pages,
        ocr_recommended=ocr_recommended,
        warnings=warnings,
    )
