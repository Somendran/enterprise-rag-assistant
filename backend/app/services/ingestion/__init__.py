"""Structured ingestion helpers (Docling parsing + vision enrichment)."""

from .doc_parser import parse_document
from .vision_enricher import (
    enrich_blocks_with_vision,
    generate_visual_description,
    get_last_vision_calls_used,
    is_visual_block,
)
from .metadata_enricher import enrich_chunk_metadata, extract_keywords, generate_summary, fallback_summary

__all__ = [
    "parse_document",
    "is_visual_block",
    "generate_visual_description",
    "enrich_blocks_with_vision",
    "get_last_vision_calls_used",
    "extract_keywords",
    "generate_summary",
    "fallback_summary",
    "enrich_chunk_metadata",
]
