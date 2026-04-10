"""
openai_llm_service.py
─────────────────────
OpenAI-backed generation service for production low-latency responses.
"""

from __future__ import annotations

import time
from typing import Iterator

from openai import OpenAI, APIConnectionError, APITimeoutError, RateLimitError, APIError

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _build_client() -> OpenAI:
    if not settings.openai_api_key.strip():
        raise RuntimeError("OPENAI_API_KEY is missing while USE_OPENAI=true.")
    return OpenAI(api_key=settings.openai_api_key, timeout=float(settings.openai_timeout_seconds))


def _extract_text_from_response(response) -> str:
    # SDK returns response.choices[0].message.content in chat completions.
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""

    message = getattr(choices[0], "message", None)
    if message is None:
        return ""

    content = getattr(message, "content", "")
    return str(content or "").strip()


def generate_response(
    prompt: str,
    max_tokens: int,
    temperature: float = 0.2,
) -> str:
    """Generate grounded output using OpenAI Chat Completions.

    Retries at most once and only for network/timeout faults.
    """
    client = _build_client()
    model = settings.openai_model
    attempts = max(1, int(settings.openai_network_retry_attempts) + 1)

    messages = [
        {
            "role": "system",
            "content": "You answer questions using only provided context. If context is insufficient, answer exactly: I don't know.",
        },
        {"role": "user", "content": prompt},
    ]

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        start = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max(1, int(max_tokens)),
                temperature=float(temperature),
                stream=False,
            )
            latency_ms = (time.perf_counter() - start) * 1000.0
            answer = _extract_text_from_response(response)
            usage = getattr(response, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", None) if usage is not None else None
            completion_tokens = getattr(usage, "completion_tokens", None) if usage is not None else None

            logger.info(
                "LLM used=openai model=%s latency_ms=%.1f tokens_in=%s tokens_out=%s response_empty=%s",
                model,
                latency_ms,
                prompt_tokens,
                completion_tokens,
                len(answer) == 0,
            )

            if not answer:
                raise RuntimeError("OpenAI returned an empty response.")

            return answer

        except (APIConnectionError, APITimeoutError) as exc:
            last_error = exc
            logger.warning(
                "OpenAI network failure | model=%s attempt=%d/%d error=%s",
                model,
                attempt,
                attempts,
                exc,
            )
            if attempt >= attempts:
                break
            continue
        except RateLimitError as exc:
            logger.error("OpenAI rate limit error | model=%s error=%s", model, exc)
            raise RuntimeError(f"OpenAI rate limit error: {exc}")
        except APIError as exc:
            logger.error("OpenAI API error | model=%s error=%s", model, exc)
            raise RuntimeError(f"OpenAI API error: {exc}")
        except Exception:
            raise

    raise RuntimeError(f"OpenAI request failed after {attempts} attempt(s): {last_error}")


def stream_response(
    prompt: str,
    max_tokens: int,
    temperature: float = 0.2,
) -> Iterator[str]:
    """Stream grounded output using OpenAI Chat Completions.

    Retries at most once and only for network/timeout faults.
    Yields plain text chunks as they arrive.
    """
    client = _build_client()
    model = settings.openai_model
    attempts = max(1, int(settings.openai_network_retry_attempts) + 1)

    messages = [
        {
            "role": "system",
            "content": "You answer questions using only provided context. If context is insufficient, answer exactly: I don't know.",
        },
        {"role": "user", "content": prompt},
    ]

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        start = time.perf_counter()
        chunk_count = 0
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max(1, int(max_tokens)),
                temperature=float(temperature),
                stream=True,
            )

            for event in stream:
                choices = getattr(event, "choices", None) or []
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                if delta is None:
                    continue
                content = getattr(delta, "content", None)
                if not content:
                    continue
                text = str(content)
                chunk_count += 1
                yield text

            latency_ms = (time.perf_counter() - start) * 1000.0
            logger.info(
                "LLM used=openai model=%s latency_ms=%.1f stream_chunks=%d",
                model,
                latency_ms,
                chunk_count,
            )
            return

        except (APIConnectionError, APITimeoutError) as exc:
            last_error = exc
            logger.warning(
                "OpenAI stream network failure | model=%s attempt=%d/%d error=%s",
                model,
                attempt,
                attempts,
                exc,
            )
            if attempt >= attempts:
                break
            continue
        except RateLimitError as exc:
            logger.error("OpenAI stream rate limit error | model=%s error=%s", model, exc)
            raise RuntimeError(f"OpenAI rate limit error: {exc}")
        except APIError as exc:
            logger.error("OpenAI stream API error | model=%s error=%s", model, exc)
            raise RuntimeError(f"OpenAI API error: {exc}")
        except Exception:
            raise

    raise RuntimeError(f"OpenAI streaming failed after {attempts} attempt(s): {last_error}")
