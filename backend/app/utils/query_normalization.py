"""
query_normalization.py
──────────────────────
Shared query normalization for cache keys, retrieval, and prompt fingerprinting.
"""

import re


def normalize_query(query: str) -> str:
    """Return a canonical query form for semantically identical user input variants."""
    text = (query or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\s\?\!\.,;:]+$", "", text)
    return text
