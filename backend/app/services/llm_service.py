"""
llm_service.py
──────────────
Responsibility: Provide local-first generation with Gemini fallback support.

Design choices:
1. Prefer local Ollama generation for low-latency/offline operation.
2. Keep Gemini fallback to a single call (no retry loops).
3. Return precise, actionable runtime errors for model and quota issues.
"""

from typing import Any
from collections import deque
import json
import threading
import time
from urllib.parse import urlparse, urlunparse

import requests
from google.api_core.exceptions import NotFound, ResourceExhausted
from langchain.schema import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
_llm_window_lock = threading.Lock()
_llm_call_timestamps: deque[float] = deque()
_local_models_lock = threading.Lock()
_local_models_cache: tuple[float, list[str]] = (0.0, [])


def _acquire_generation_slot() -> None:
    rpm = max(1, settings.llm_requests_per_minute)
    now = time.time()

    with _llm_window_lock:
        # Sliding 60-second window.
        while _llm_call_timestamps and (now - _llm_call_timestamps[0]) >= 60.0:
            _llm_call_timestamps.popleft()

        if len(_llm_call_timestamps) >= rpm:
            retry_after = max(1, int(60 - (now - _llm_call_timestamps[0])))
            raise RuntimeError(
                "Application rate limit reached for Gemini generation. "
                f"Retry after approximately {retry_after} seconds."
            )

        _llm_call_timestamps.append(now)


def _normalize_model_name(name: str) -> str:
    """Normalize 'models/foo' to 'foo' for stable comparisons."""
    if name.startswith("models/"):
        return name.split("/", 1)[1]
    return name


def _discover_generation_models(api_key: str) -> list[str]:
    """Return discovered model ids that support generateContent."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    discovered: list[str] = []
    for item in response.json().get("models", []):
        methods = item.get("supportedGenerationMethods", [])
        if "generateContent" in methods:
            name = item.get("name")
            if not name:
                continue
            discovered.append(_normalize_model_name(name))

    return list(dict.fromkeys(discovered))


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

    # Preserve strict behavior: return empty string if no usable answer is found.
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
            # Ignore malformed partial lines and keep consuming stream.
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
    """Generate an answer from local Ollama with at most two total calls.

    Returns:
        (answer_text, retry_count, retry_reason)
    """
    _validate_configured_local_model_exists()

    # Encourage a direct final answer and reduce token spend on reasoning.
    prompt_for_model = (
        "Provide only the final concise answer. "
        "Do not include reasoning traces or analysis tags. "
        "If the context is insufficient, output exactly: I don't know.\n\n"
        f"{prompt}"
    )

    default_budget = max(1, int(settings.llm_max_tokens or settings.local_llm_num_predict))
    base_num_predict = max(1, int(max_tokens_override or default_budget))
    retry_num_predict = max(1, int(settings.local_llm_retry_num_predict))
    max_attempts = max(1, int(settings.local_llm_max_attempts))
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
            raise RuntimeError(f"Local LLM request failed: {exc}")

        try:
            data = _parse_ollama_response(response, stream_enabled=stream_enabled)
        except Exception as exc:
            logger.error("Local LLM invalid JSON response: %s", exc)
            raise RuntimeError(f"Local LLM returned invalid JSON: {exc}")

        elapsed_s = time.perf_counter() - request_start
        return data, elapsed_s

    max_attempts = min(2, max_attempts)

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

        # If stream mode produced no chunks/text, salvage with one non-stream call
        # at the same token budget to avoid escalating into full retry immediately.
        if stream_enabled and response_empty and int(stream_chunk_count or 0) == 0:
            logger.warning(
                "Local stream produced no chunks. Falling back to non-stream for same attempt."
            )
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


def call_gemini(prompt: str) -> str:
    """Single-attempt Gemini invocation used only as fallback."""
    model_name = _normalize_model_name(settings.llm_model)
    _acquire_generation_slot()

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=settings.google_api_key,
        temperature=0,
        max_retries=0,
    )

    try:
        response: Any = llm.invoke([HumanMessage(content=prompt)])
        return str(response.content).strip()
    except ResourceExhausted as exc:
        raise RuntimeError(
            "Gemini API quota exceeded for model "
            f"'{model_name}'. No retries were attempted. Error: {exc}"
        )
    except NotFound as exc:
        try:
            discovered = _discover_generation_models(settings.google_api_key)
        except Exception:
            discovered = []
        raise RuntimeError(
            "Configured LLM model is unavailable: "
            f"'{model_name}'. "
            f"Discovered generate-capable models: {discovered}. "
            f"Original error: {exc}"
        )


def generate_answer(
    prompt: str,
    max_tokens_override: int | None = None,
) -> tuple[str, str, int, str]:
    """
    Generate an answer using local-first flow with Gemini fallback.

    Returns:
        (answer_text, model_used, retry_count, retry_reason)
    """
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
            logger.info("Fallback triggered: empty")
            if not settings.enable_gemini_fallback:
                raise RuntimeError("Local response was empty and Gemini fallback is disabled.")
            gemini_answer = call_gemini(prompt)
            return gemini_answer, "gemini", retry_count, retry_reason

        return cleaned, "local", retry_count, retry_reason

    except Exception as exc:
        logger.error("Local model failed: %s", str(exc))
        if not settings.enable_gemini_fallback:
            raise RuntimeError(
                "Local generation failed and Gemini fallback is disabled. "
                f"Error: {exc}"
            )
        gemini_answer = call_gemini(prompt)
        return gemini_answer, "gemini", 0, ""
