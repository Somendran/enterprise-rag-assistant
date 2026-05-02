"""Docling-backed structured parser for ingestion.

This module is intentionally defensive:
- If Docling is unavailable or parsing fails, callers can fallback.
- Returned blocks are normalized dicts with stable keys for downstream chunking.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import os
import tempfile
from typing import Any, Dict, List, Optional
from uuid import uuid4

from app.services.document_loader import compute_file_hash, load_pdf
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
DOCLING_BATCH_SIZE = 10


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _safe_bbox(item: Any) -> Optional[dict[str, float]]:
    bbox = getattr(item, "bbox", None)
    if bbox is None:
        return None

    if isinstance(bbox, dict):
        return bbox

    keys = ("x0", "y0", "x1", "y1")
    out: dict[str, float] = {}
    for key in keys:
        raw = getattr(bbox, key, None)
        if raw is None:
            continue
        try:
            out[key] = float(raw)
        except Exception:
            continue
    return out or None


def _normalize_block_type(raw_type: str) -> str:
    value = raw_type.lower().strip()
    if "heading" in value or value in {"title", "header", "h1", "h2", "h3", "h4", "h5", "h6"}:
        return "heading"
    if "table" in value:
        return "table"
    if value in {"image", "figure", "chart", "visual"}:
        return "image"
    return "paragraph"


def _extract_content(item: Any) -> str:
    for attr in ("text", "content", "markdown"):
        text = _to_text(getattr(item, attr, ""))
        if text:
            return text

    for method_name in ("to_markdown", "export_to_markdown", "get_text"):
        method = getattr(item, method_name, None)
        if callable(method):
            try:
                text = _to_text(method())
                if text:
                    return text
            except Exception:
                continue

    return ""


def _extract_image_bytes(item: Any) -> Optional[bytes]:
    direct = getattr(item, "image_bytes", None)
    if isinstance(direct, (bytes, bytearray)) and direct:
        return bytes(direct)

    for attr in ("image", "pil_image"):
        img_obj = getattr(item, attr, None)
        if img_obj is None:
            continue
        try:
            from io import BytesIO

            buffer = BytesIO()
            img_obj.save(buffer, format="PNG")
            payload = buffer.getvalue()
            if payload:
                return payload
        except Exception:
            continue

    return None


def _blocks_from_markdown(markdown: str, source: str, page: int) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    lines = [line.rstrip() for line in (markdown or "").splitlines()]

    paragraph_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if paragraph_lines:
                content = "\n".join(paragraph_lines).strip()
                if content:
                    blocks.append(
                        {
                            "id": str(uuid4()),
                            "type": "paragraph",
                            "content": content,
                            "page": page,
                            "bbox": None,
                            "source": source,
                        }
                    )
                paragraph_lines = []
            continue

        if stripped.startswith("#"):
            if paragraph_lines:
                content = "\n".join(paragraph_lines).strip()
                if content:
                    blocks.append(
                        {
                            "id": str(uuid4()),
                            "type": "paragraph",
                            "content": content,
                            "page": page,
                            "bbox": None,
                            "source": source,
                        }
                    )
                paragraph_lines = []

            heading = stripped.lstrip("#").strip()
            if heading:
                blocks.append(
                    {
                        "id": str(uuid4()),
                        "type": "heading",
                        "content": heading,
                        "page": page,
                        "bbox": None,
                        "source": source,
                    }
                )
            continue

        if stripped.startswith("|") and stripped.endswith("|"):
            if paragraph_lines:
                content = "\n".join(paragraph_lines).strip()
                if content:
                    blocks.append(
                        {
                            "id": str(uuid4()),
                            "type": "paragraph",
                            "content": content,
                            "page": page,
                            "bbox": None,
                            "source": source,
                        }
                    )
                paragraph_lines = []

            blocks.append(
                {
                    "id": str(uuid4()),
                    "type": "table",
                    "content": stripped,
                    "page": page,
                    "bbox": None,
                    "source": source,
                }
            )
            continue

        paragraph_lines.append(stripped)

    if paragraph_lines:
        content = "\n".join(paragraph_lines).strip()
        if content:
            blocks.append(
                {
                    "id": str(uuid4()),
                    "type": "paragraph",
                    "content": content,
                    "page": page,
                    "bbox": None,
                    "source": source,
                }
            )

    return blocks


def _merge_markdown_table_rows(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    table_buffer: list[str] = []
    table_meta: dict[str, Any] | None = None

    for block in blocks:
        if block["type"] == "table" and block.get("content", "").strip().startswith("|"):
            if table_meta is None:
                table_meta = dict(block)
                table_meta["id"] = str(uuid4())
                table_buffer = []
            table_buffer.append(block["content"].strip())
            continue

        if table_meta is not None:
            table_meta["content"] = "\n".join(table_buffer).strip()
            merged.append(table_meta)
            table_buffer = []
            table_meta = None

        merged.append(block)

    if table_meta is not None:
        table_meta["content"] = "\n".join(table_buffer).strip()
        merged.append(table_meta)

    return merged


def _build_converter(document_converter_cls: type, force_ocr: bool | None = None):
    use_ocr = bool(settings.enable_ocr if force_ocr is None else force_ocr)

    try:
        from docling.datamodel.base_models import InputFormat  # type: ignore
        from docling.datamodel.pipeline_options import PdfPipelineOptions  # type: ignore
        from docling.document_converter import PdfFormatOption  # type: ignore

        pdf_options = PdfPipelineOptions()
        if hasattr(pdf_options, "do_ocr"):
            pdf_options.do_ocr = use_ocr
        if hasattr(pdf_options, "ocr_options") and hasattr(pdf_options.ocr_options, "force_full_page_ocr"):
            pdf_options.ocr_options.force_full_page_ocr = use_ocr

        return document_converter_cls(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options),
            }
        )
    except Exception as exc:
        logger.warning("Could not apply OCR-off Docling options, using default converter. error=%s", exc)
        return document_converter_cls()


def _extract_blocks_from_doc(
    doc: Any,
    source_name: str,
    file_hash: str,
    document_id: str,
    uploaded_at: str,
    page_offset: int,
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []

    pages = getattr(doc, "pages", None)
    if pages:
        for page_index, page in enumerate(pages, start=1):
            items = getattr(page, "items", None) or getattr(page, "elements", None) or []
            for item in items:
                raw_type = _to_text(getattr(item, "type", "")) or item.__class__.__name__
                block_type = _normalize_block_type(raw_type)
                content = _extract_content(item)
                if block_type == "image" and not content:
                    content = "Visual element extracted from document."
                if not content and block_type != "image":
                    continue

                page_no = int(getattr(page, "page_no", page_index) or page_index)
                block: dict[str, Any] = {
                    "id": str(uuid4()),
                    "type": block_type,
                    "content": content,
                    "page": page_offset + page_no,
                    "bbox": _safe_bbox(item),
                    "source": source_name,
                    "file_hash": file_hash,
                    "document_id": document_id,
                    "uploaded_at": uploaded_at,
                }

                image_bytes = _extract_image_bytes(item)
                if image_bytes:
                    block["image_bytes"] = image_bytes

                blocks.append(block)

    if blocks:
        return blocks

    markdown = ""
    for method_name in ("export_to_markdown", "to_markdown"):
        method = getattr(doc, method_name, None)
        if callable(method):
            try:
                markdown = _to_text(method())
                if markdown:
                    break
            except Exception:
                continue

    if not markdown:
        return []

    markdown_blocks = _blocks_from_markdown(markdown, source=source_name, page=page_offset + 1)
    for block in markdown_blocks:
        block["file_hash"] = file_hash
        block["document_id"] = document_id
        block["uploaded_at"] = uploaded_at
    return markdown_blocks


def _fallback_parse_batch(
    batch_pdf_path: str,
    source_name: str,
    file_hash: str,
    document_id: str,
    uploaded_at: str,
    page_offset: int,
) -> list[dict[str, Any]]:
    docs = load_pdf(batch_pdf_path)
    fallback_blocks: list[dict[str, Any]] = []
    for idx, doc in enumerate(docs, start=1):
        text = str(doc.page_content or "").strip()
        if not text:
            continue
        fallback_blocks.append(
            {
                "id": str(uuid4()),
                "type": "paragraph",
                "content": text,
                "page": page_offset + idx,
                "bbox": None,
                "source": source_name,
                "file_hash": file_hash,
                "document_id": document_id,
                "uploaded_at": uploaded_at,
            }
        )
    return fallback_blocks


def parse_document(file_path: str, force_ocr: bool | None = None) -> List[Dict]:
    """Parse a PDF into structured blocks using Docling.

    Returns normalized blocks with keys:
    {
      "id": str,
      "type": "heading" | "paragraph" | "table" | "image",
      "content": str,
      "page": int,
      "bbox": Optional[dict],
    }

    Raises RuntimeError when Docling parsing is unavailable/failed so callers can fallback.
    """
    path = Path(file_path)
    if not path.exists():
        raise RuntimeError(f"File not found for Docling parse: {file_path}")

    try:
        from docling.document_converter import DocumentConverter  # type: ignore
        from pypdf import PdfReader, PdfWriter
    except Exception as exc:
        raise RuntimeError(f"Docling is not installed or import failed: {exc}")

    file_hash = compute_file_hash(path)
    uploaded_at = datetime.now(timezone.utc).isoformat()
    document_id = hashlib.sha1(f"{path.name}:{file_hash}".encode("utf-8")).hexdigest()[:16]

    converter = _build_converter(DocumentConverter, force_ocr=force_ocr)

    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise RuntimeError(f"Failed to open PDF for batched parsing: {exc}")

    total_pages = len(reader.pages)
    if total_pages == 0:
        raise RuntimeError("PDF has zero pages.")

    batch_size = max(1, int(settings.docling_batch_size or DOCLING_BATCH_SIZE))
    blocks: list[dict[str, Any]] = []

    for batch_start in range(0, total_pages, batch_size):
        batch_end = min(batch_start + batch_size, total_pages)
        writer = PdfWriter()
        for page_idx in range(batch_start, batch_end):
            writer.add_page(reader.pages[page_idx])

        tmp_path = ""
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
                writer.write(tmp_file)
                tmp_path = tmp_file.name

            try:
                result = converter.convert(tmp_path)
                doc = getattr(result, "document", result)
                parsed = _extract_blocks_from_doc(
                    doc=doc,
                    source_name=path.name,
                    file_hash=file_hash,
                    document_id=document_id,
                    uploaded_at=uploaded_at,
                    page_offset=batch_start,
                )
                if not parsed:
                    raise RuntimeError("Docling returned empty batch output.")
                blocks.extend(parsed)
            except Exception as batch_exc:
                logger.warning(
                    "Docling failed on page batch %d-%d; using fallback parser. error=%s",
                    batch_start + 1,
                    batch_end,
                    batch_exc,
                )
                blocks.extend(
                    _fallback_parse_batch(
                        batch_pdf_path=tmp_path,
                        source_name=path.name,
                        file_hash=file_hash,
                        document_id=document_id,
                        uploaded_at=uploaded_at,
                        page_offset=batch_start,
                    )
                )
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    if not blocks:
        raise RuntimeError("Docling conversion returned no structured content.")

    merged = _merge_markdown_table_rows(blocks)

    logger.info(
        "Docling parsed '%s' into %d block(s)",
        path.name,
        len(merged),
    )

    return merged
