"""
query_cache.py
──────────────
Simple in-memory TTL cache for query responses.

This cache intentionally stays process-local and dependency-free.
"""

import hashlib
import json
import threading
import time
from typing import Any, Optional

from app.utils.query_normalization import normalize_query

_cache_lock = threading.Lock()
_cache_store: dict[str, tuple[float, dict[str, Any]]] = {}

def _normalize_ids(values: list[str] | None) -> list[str]:
    if not values:
        return []
    return [str(v).strip() for v in values if str(v).strip()]


def build_prompt_fingerprint(prompt: str) -> str:
    normalized = " ".join(prompt.strip().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def build_cache_key(
    question: str,
    kb_version: str,
    top_k_ids: list[str] | None = None,
    prompt_fingerprint: str | None = None,
) -> str:
    payload = {
        "q": normalize_query(question),
        "kb": kb_version,
        "ids": _normalize_ids(top_k_ids),
        "pf": (prompt_fingerprint or "").strip(),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get_cached_result(cache_key: str) -> Optional[dict[str, Any]]:
    now = time.time()
    with _cache_lock:
        entry = _cache_store.get(cache_key)
        if entry is None:
            return None
        expires_at, payload = entry
        if expires_at <= now:
            _cache_store.pop(cache_key, None)
            return None
        return payload


def set_cached_result(cache_key: str, payload: dict[str, Any], ttl_seconds: int) -> None:
    expires_at = time.time() + max(1, ttl_seconds)
    with _cache_lock:
        _cache_store[cache_key] = (expires_at, payload)


def clear_query_cache() -> None:
    with _cache_lock:
        _cache_store.clear()
