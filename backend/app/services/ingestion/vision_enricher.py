"""Vision enrichment helpers for structured ingestion blocks."""

from __future__ import annotations

import base64
from typing import Any, Dict, List

import requests
from openai import OpenAI

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

VISION_PROMPT = """Describe this visual for use in a retrieval system.

Focus on:

* trends (increase/decrease)
* key values
* comparisons
* entities involved

Be concise and factual (1–2 sentences max).

Example:
"Line chart showing revenue increasing from $100 to $150 between January and March."
"""

_LAST_VISION_CALLS_USED = 0


def is_visual_block(block: Dict) -> bool:
    block_type = str(block.get("type", "")).strip().lower()
    text = str(block.get("content", "") or "")

    # Direct visual types
    if block_type in {"image", "figure", "chart"}:
        return True

    # Treat tables as semantic visuals
    if block_type != "table":
        return False

    text_len = max(len(text), 1)

    # Heuristic 1: numeric density
    numeric_ratio = sum(c.isdigit() for c in text) / text_len

    # Heuristic 2: tabular structure
    has_rows = ("\n" in text) and ("|" in text or "\t" in text)

    # Heuristic 3: semantic keywords
    keywords = ["rate", "hours", "days", "entitlement", "%", "rm", "salary"]
    keyword_hit = any(k in text.lower() for k in keywords)

    if numeric_ratio > 0.15 or has_rows or keyword_hit:
        return True

    return False


def summarize_table(block: Dict) -> str:
    """Create fast deterministic 1-2 sentence summary for table blocks."""
    text = str(block.get("content", "") or "")
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    if len(lines) >= 2:
        header = lines[0]
        sample = lines[1:4]
        preview = "; ".join(sample)
        return f"Table showing {header}. Example entries: {preview}"

    return "Table containing structured data."


def _extract_openai_text(response: Any) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    if message is None:
        return ""
    content = getattr(message, "content", "")
    return str(content or "").strip()


def _openai_vision_description(image_bytes: bytes) -> str:
    if not settings.openai_api_key.strip():
        raise RuntimeError("OPENAI_API_KEY missing for vision enrichment.")

    model_name = (settings.openai_vision_model or settings.openai_model).strip()
    client = OpenAI(api_key=settings.openai_api_key, timeout=float(settings.openai_timeout_seconds))
    encoded = base64.b64encode(image_bytes).decode("ascii")
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VISION_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encoded}",
                        },
                    },
                ],
            }
        ],
        max_tokens=120,
        temperature=0.0,
    )
    text = _extract_openai_text(response)
    if not text:
        raise RuntimeError("OpenAI vision returned empty output.")
    return text


def _local_vision_description(image_bytes: bytes) -> str:
    model = (settings.local_vision_model or "").strip()
    if not model:
        raise RuntimeError("LOCAL_VISION_MODEL is not configured.")

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": VISION_PROMPT,
                "images": [base64.b64encode(image_bytes).decode("ascii")],
            }
        ],
        "stream": False,
    }

    endpoint = settings.local_vision_endpoint
    response = requests.post(
        endpoint,
        json=payload,
        timeout=(
            max(1, int(settings.local_llm_connect_timeout_seconds)),
            max(1, int(settings.local_llm_read_timeout_seconds)),
        ),
    )
    response.raise_for_status()
    data = response.json()

    message = data.get("message") if isinstance(data, dict) else None
    text = ""
    if isinstance(message, dict):
        text = str(message.get("content", "")).strip()

    if not text:
        text = str(data.get("response", "")).strip() if isinstance(data, dict) else ""

    if not text:
        raise RuntimeError("Local vision model returned empty output.")

    return text


def generate_visual_description(image_bytes: bytes) -> str:
    """Generate factual 1-2 sentence visual description.

    Uses OpenAI vision path when USE_OPENAI=true; otherwise local vision model.
    """
    if not image_bytes:
        raise RuntimeError("Missing image bytes for vision description.")

    if settings.use_openai:
        return _openai_vision_description(image_bytes)
    return _local_vision_description(image_bytes)


def get_last_vision_calls_used() -> int:
    return _LAST_VISION_CALLS_USED


def enrich_blocks_with_vision(blocks: List[Dict]) -> List[Dict]:
    """Attach visual descriptions for visual/complex-table blocks.

    Vision failures are non-fatal by design and silently skipped.
    """
    global _LAST_VISION_CALLS_USED

    if not settings.enable_vision_enrichment or not blocks:
        _LAST_VISION_CALLS_USED = 0
        return blocks

    max_calls = max(0, int(settings.max_vision_calls_per_doc))
    calls_used = 0

    for block in blocks:
        block_type = str(block.get("type", "")).strip().lower()

        if block_type == "table":
            if not is_visual_block(block):
                continue
            logger.info(
                "Visual detected: %s | triggering enrichment",
                block_type,
            )
            block["visual_description"] = summarize_table(block)
            block["has_visual"] = True
            continue

        if not is_visual_block(block):
            continue

        if calls_used >= max_calls:
            break

        logger.info(
            "Visual detected: %s | triggering enrichment",
            block_type,
        )

        image_bytes = block.get("image_bytes")
        if not isinstance(image_bytes, (bytes, bytearray)) or not image_bytes:
            continue

        try:
            description = generate_visual_description(bytes(image_bytes))
        except Exception as exc:
            logger.warning(
                "Vision enrichment skipped for block=%s page=%s error=%s",
                block.get("id", "unknown"),
                block.get("page", "?"),
                exc,
            )
            continue

        if description:
            block["visual_description"] = description.strip()
            block["has_visual"] = True
            calls_used += 1

    _LAST_VISION_CALLS_USED = calls_used
    logger.info("Vision enrichment complete | vision_calls_used=%d", calls_used)
    return blocks
