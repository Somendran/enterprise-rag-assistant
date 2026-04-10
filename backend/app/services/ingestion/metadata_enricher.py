"""Chunk-level metadata enrichment for ingestion.

Features:
- Deterministic keyword extraction
- Optional strict one-sentence summary generation
- Performance-safe summary gating
"""

from __future__ import annotations

from collections import Counter
import re
from typing import List

import requests
from langchain.schema import Document
from openai import OpenAI

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

STOPWORDS = {
    "the", "and", "is", "of", "to", "in", "for", "on", "with", "a", "an",
    "this", "that", "these", "those", "are", "was", "were", "be", "as", "at",
    "by", "or", "from", "it", "its", "their", "them", "we", "you", "our", "your",
    "can", "may", "must", "shall", "will", "not", "if", "than", "then", "also",
}

DOMAIN_TERMS = {
    "policy", "compliance", "benefits", "leave", "salary", "entitlement", "employment",
    "probation", "termination", "hours", "overtime", "allowance", "claim", "approval",
    "security", "privacy", "governance", "risk", "audit", "contract", "notice",
}

SUMMARY_PROMPT = (
    "Summarize this text in ONE short sentence (max 20 words).\n"
    "Be factual and concise. Do not add information.\n\n"
    "Text:\n{chunk}"
)


def _truncate_tokens(text: str, max_tokens: int) -> str:
    tokens = re.findall(r"\S+", text.strip())
    if not tokens:
        return ""
    return " ".join(tokens[: max(1, max_tokens)])


def extract_keywords(text: str, max_keywords: int = 5) -> List[str]:
    words = re.findall(r"\b[a-zA-Z]{3,}\b", (text or "").lower())
    filtered = [w for w in words if w not in STOPWORDS]
    if not filtered:
        return []

    counts = Counter(filtered)
    scored: list[tuple[str, float]] = []
    for term, freq in counts.items():
        # Slight boost for known domain terms to improve enterprise retrieval.
        domain_boost = 1.25 if term in DOMAIN_TERMS else 1.0
        uniqueness_penalty = 1.0 / (1.0 + (freq - 1) * 0.15)
        score = float(freq) * domain_boost * uniqueness_penalty
        scored.append((term, score))

    scored.sort(key=lambda x: (-x[1], x[0]))
    return [term for term, _ in scored[: max(1, max_keywords)]]


def fallback_summary(text: str) -> str:
    sentences = re.split(r"[.!?]", text or "")
    first = (sentences[0] if sentences else "").strip()
    first = first[:150].strip()
    return _truncate_tokens(first, int(settings.summary_max_tokens))


def _generate_summary_openai(text: str) -> str:
    if not settings.openai_api_key.strip():
        raise RuntimeError("OPENAI_API_KEY is required for OpenAI summarization.")

    client = OpenAI(
        api_key=settings.openai_api_key,
        timeout=float(settings.openai_timeout_seconds),
    )

    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {
                "role": "user",
                "content": SUMMARY_PROMPT.format(chunk=text),
            }
        ],
        max_tokens=max(8, int(settings.summary_max_tokens) * 2),
        temperature=0.0,
        stream=False,
    )

    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    if message is None:
        return ""
    return str(getattr(message, "content", "") or "").strip()


def _generate_summary_local(text: str) -> str:
    prompt = SUMMARY_PROMPT.format(chunk=text)
    payload = {
        "model": settings.local_llm_model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": max(12, int(settings.summary_max_tokens) * 2),
            "num_gpu": settings.local_llm_num_gpu,
        },
    }

    response = requests.post(
        settings.local_llm_endpoint,
        json=payload,
        timeout=(
            max(1, int(settings.local_llm_connect_timeout_seconds)),
            max(1, int(settings.local_llm_read_timeout_seconds)),
        ),
    )
    response.raise_for_status()
    data = response.json()
    text_out = str(data.get("response", "") or "").strip()
    if not text_out and isinstance(data.get("message"), dict):
        text_out = str(data["message"].get("content", "") or "").strip()
    return text_out


def generate_summary(text: str) -> str:
    candidate = ""
    if settings.use_openai:
        candidate = _generate_summary_openai(text)
    else:
        candidate = _generate_summary_local(text)

    # Enforce strict one-sentence and token limits.
    sentence = re.split(r"(?<=[.!?])\s+", candidate.strip())[0] if candidate else ""
    sentence = sentence.strip().replace("\n", " ")
    sentence = _truncate_tokens(sentence, int(settings.summary_max_tokens))
    return sentence


def enrich_chunk_metadata(chunks: List[Document]) -> List[Document]:
    if not chunks:
        return chunks

    summarized = 0
    for idx, chunk in enumerate(chunks):
        text = str(chunk.page_content or "").strip()
        if not text:
            continue

        keywords = extract_keywords(text, max_keywords=5)
        chunk.metadata["keywords"] = keywords

        summary = ""
        can_summarize = (
            bool(settings.enable_summary)
            and idx < max(0, int(settings.summary_max_chunks))
            and len(text) >= max(1, int(settings.summary_min_chars))
        )

        if can_summarize:
            try:
                summary = generate_summary(text)
            except Exception:
                summary = fallback_summary(text)

        if summary:
            chunk.metadata["summary"] = summary
            summarized += 1

        # Embed enriched text while preserving original chunk body below.
        if keywords or summary:
            prefix_parts: list[str] = []
            if summary:
                prefix_parts.append(summary)
            if keywords:
                prefix_parts.append(f"Keywords: {', '.join(keywords)}")
            prefix = "\n".join(prefix_parts).strip()
            chunk.page_content = f"{prefix}\n\n{text}".strip()

    logger.info(
        "Metadata enrichment applied | chunks=%d summarized=%d",
        len(chunks),
        summarized,
    )
    return chunks
