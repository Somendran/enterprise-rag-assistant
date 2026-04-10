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

_cache_lock = threading.Lock()
_cache_store: dict[str, tuple[float, dict[str, Any]]] = {}


def _normalize_question(question: str) -> str:
    return " ".join(question.lower().strip().split())


def build_cache_key(question: str, kb_version: str) -> str:
    payload = {
        "q": _normalize_question(question),
        "kb": kb_version,
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
