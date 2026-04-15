"""Local Ollama generation service."""

from typing import Any
import json
import threading
import time
from urllib.parse import urlparse, urlunparse

import requests

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
_local_models_lock = threading.Lock()
_local_models_cache: tuple[float, list[str]] = (0.0, [])


def _extract_ollama_text(data: dict[str, Any]) -> str:
    """Best-effort extraction of final answer text from Ollama payloads."""
    candidates = [
        data.get("response"),
        data.get("output_text"),
        ((data.get("message") or {}).get("content") if isinstance(data.get("message"), dict) else None),
    ]

    for candidate in candidates:
        if candidate is None:
            continue
        text = str(candidate).strip()
        if text:
            return text

    return ""


def _parse_ollama_response(response: requests.Response, stream_enabled: bool) -> dict[str, Any]:
    """Parse Ollama response in stream and non-stream modes into one payload."""
    if not stream_enabled:
        return response.json()

    response_parts: list[str] = []
    thinking_parts: list[str] = []
    last_event: dict[str, Any] = {}

    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError:
            continue

        if not isinstance(event, dict):
            continue

        last_event = event

        chunk_text = str(event.get("response", ""))
        if chunk_text:
            response_parts.append(chunk_text)

        chunk_thinking = str(event.get("thinking", ""))
        if chunk_thinking:
            thinking_parts.append(chunk_thinking)

    aggregated = dict(last_event)
    aggregated["response"] = "".join(response_parts)
    aggregated["thinking"] = "".join(thinking_parts)
    aggregated["_stream_chunk_count"] = len(response_parts)
    return aggregated


def _ollama_tags_url() -> str:
    """Convert generate endpoint URL to Ollama tags URL."""
    parsed = urlparse(settings.local_llm_endpoint)
    return urlunparse((parsed.scheme, parsed.netloc, "/api/tags", "", "", ""))


def _get_available_local_models(force_refresh: bool = False) -> list[str]:
    """Fetch available local model tags from Ollama with a short cache."""
    global _local_models_cache

    cache_ttl_seconds = 30.0
    now = time.time()

    with _local_models_lock:
        cached_at, cached_models = _local_models_cache
        if not force_refresh and cached_models and (now - cached_at) < cache_ttl_seconds:
            return list(cached_models)

    tags_url = _ollama_tags_url()
    try:
        response = requests.get(
            tags_url,
            timeout=(
                max(1, int(settings.local_llm_connect_timeout_seconds)),
                max(1, int(settings.local_llm_read_timeout_seconds)),
            ),
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        logger.warning("Could not query Ollama tags endpoint '%s': %s", tags_url, exc)
        return []

    models: list[str] = []
    for item in payload.get("models", []):
        if isinstance(item, dict):
            name = item.get("name")
            if isinstance(name, str) and name.strip():
                models.append(name.strip())

    models = sorted(set(models))
    with _local_models_lock:
        _local_models_cache = (now, models)

    return models


def _validate_configured_local_model_exists() -> None:
    """Fail fast with a clear error when configured local model is missing."""
    if not settings.local_llm_validate_model:
        return

    configured = settings.local_llm_model.strip()
    available = _get_available_local_models(force_refresh=False)
    if not available:
        logger.warning(
            "Skipped strict local model validation: no models discovered from Ollama tags endpoint."
        )
        return

    if configured not in available:
        raise RuntimeError(
            "Configured local model is not installed in Ollama: "
            f"'{configured}'. Available models: {available}."
        )


def call_local_llm(
    prompt: str,
    max_tokens_override: int | None = None,
) -> tuple[str, int, str]:
    """Generate an answer from local Ollama with at most two total calls."""
    _validate_configured_local_model_exists()

    prompt_for_model = (
        "Provide only the final concise answer. "
        "Do not include reasoning traces or analysis tags. "
        "If the context is insufficient, output exactly: I don't know.\n\n"
        f"{prompt}"
    )

    default_budget = max(1, int(settings.llm_max_tokens or settings.local_llm_num_predict))
    base_num_predict = max(1, int(max_tokens_override or default_budget))
    retry_num_predict = max(1, int(settings.local_llm_retry_num_predict))
    max_attempts = min(2, max(1, int(settings.local_llm_max_attempts)))
    last_error: str = "Local LLM returned no output."
    retry_count = 0
    retry_reason = ""

    def _run_local_request(num_predict: int, stream_enabled: bool) -> tuple[dict[str, Any], float]:
        payload = {
            "model": settings.local_llm_model,
            "prompt": prompt_for_model,
            "stream": stream_enabled,
            "options": {
                "temperature": settings.local_llm_temperature,
                "num_predict": num_predict,
                "num_gpu": settings.local_llm_num_gpu,
            },
        }

        logger.info(
            "Calling LOCAL model | endpoint=%s model=%s stream=%s num_predict=%d num_gpu=%s",
            settings.local_llm_endpoint,
            settings.local_llm_model,
            stream_enabled,
            num_predict,
            settings.local_llm_num_gpu,
        )

        try:
            request_start = time.perf_counter()
            response = requests.post(
                settings.local_llm_endpoint,
                json=payload,
                timeout=(
                    max(1, int(settings.local_llm_connect_timeout_seconds)),
                    max(1, int(settings.local_llm_read_timeout_seconds)),
                ),
                stream=stream_enabled,
            )
            response.raise_for_status()
        except requests.HTTPError as exc:
            body_preview = ""
            if exc.response is not None:
                body_preview = exc.response.text[:300]
            logger.error(
                "Local LLM HTTP error | status=%s body=%s",
                getattr(exc.response, "status_code", "unknown"),
                body_preview,
            )
            raise RuntimeError(
                "Local LLM request failed with HTTP error "
                f"{getattr(exc.response, 'status_code', 'unknown')}: {body_preview}"
            )
        except requests.RequestException as exc:
            logger.error("Local LLM request exception: %s", exc)
            raise RuntimeError(f"Local LLM request failed: {exc}") from exc

        try:
            data = _parse_ollama_response(response, stream_enabled=stream_enabled)
        except Exception as exc:
            logger.error("Local LLM invalid JSON response: %s", exc)
            raise RuntimeError(f"Local LLM returned invalid JSON: {exc}") from exc

        elapsed_s = time.perf_counter() - request_start
        return data, elapsed_s

    for attempt in range(1, max_attempts + 1):
        num_predict = base_num_predict if attempt == 1 else retry_num_predict
        stream_enabled = bool(settings.local_llm_stream) if attempt == 1 else False
        data, elapsed_s = _run_local_request(num_predict=num_predict, stream_enabled=stream_enabled)

        done_reason = data.get("done_reason")
        eval_count = data.get("eval_count")
        prompt_eval_count = data.get("prompt_eval_count")
        raw_thinking = data.get("thinking", "")
        stream_chunk_count = data.get("_stream_chunk_count")
        answer_text = _extract_ollama_text(data)
        response_empty = len(answer_text.strip()) == 0

        if stream_enabled and response_empty and int(stream_chunk_count or 0) == 0:
            logger.warning("Local stream produced no chunks. Falling back to non-stream for same attempt.")
            data, elapsed_s = _run_local_request(num_predict=num_predict, stream_enabled=False)
            done_reason = data.get("done_reason")
            eval_count = data.get("eval_count")
            prompt_eval_count = data.get("prompt_eval_count")
            raw_thinking = data.get("thinking", "")
            stream_chunk_count = data.get("_stream_chunk_count")
            answer_text = _extract_ollama_text(data)
            response_empty = len(answer_text.strip()) == 0

        logger.info(
            "Local generation stats | prompt_len=%d num_predict=%d prompt_eval_count=%s eval_count=%s done_reason=%s response_empty=%s retry_count=%d elapsed_s=%.3f stream_chunks=%s",
            len(prompt_for_model),
            num_predict,
            prompt_eval_count,
            eval_count,
            done_reason,
            response_empty,
            retry_count,
            elapsed_s,
            stream_chunk_count,
        )
        logger.info("Local response length: %d", len(answer_text))
        logger.info("Local response raw: %r", answer_text)
        logger.info("Ollama raw payload preview: %r", repr(data)[:2000])

        if not response_empty:
            return answer_text, retry_count, retry_reason

        last_error = (
            "Local LLM returned empty response. "
            f"done_reason={done_reason}, eval_count={eval_count}, "
            f"thinking_len={len(str(raw_thinking or ''))}"
        )

        if response_empty and done_reason == "length" and attempt < max_attempts:
            retry_count = 1
            retry_reason = "length"
            logger.warning("Local LLM hit length cap. Retrying once with higher token budget.")
            continue

        logger.warning(
            "Local LLM empty response | thinking_len=%d done_reason=%s eval_count=%s",
            len(str(raw_thinking or "")),
            done_reason,
            eval_count,
        )

    if retry_reason == "length":
        logger.warning(
            "Local retries exhausted with done_reason=length and empty output. Returning conservative fallback."
        )
        return "I don't know", retry_count, retry_reason

    raise RuntimeError(last_error)


def generate_answer(
    prompt: str,
    max_tokens_override: int | None = None,
) -> tuple[str, str, int, str]:
    """Generate an answer using the configured local Ollama model."""
    try:
        logger.info("Calling LOCAL model...")
        local_answer, retry_count, retry_reason = call_local_llm(
            prompt,
            max_tokens_override=max_tokens_override,
        )
        cleaned = local_answer.strip()
        logger.info("Local response length: %d", len(cleaned))
        logger.info("Local response raw: %r", cleaned)

        if not cleaned:
            raise RuntimeError("Local LLM returned an empty response.")

        return cleaned, "local", retry_count, retry_reason

    except Exception as exc:
        logger.error("Local model failed: %s", str(exc))
        raise RuntimeError(f"Local generation failed. Error: {exc}") from exc
