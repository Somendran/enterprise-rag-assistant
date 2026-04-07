"""
llm_service.py
──────────────
Responsibility: Provide a strict Gemini chat invocation path.

Design choices:
1. Use a single configured model (Gemini 2.5 Flash by default).
2. Avoid broad retry/fallback loops that increase latency and unpredictability.
3. Return precise, actionable runtime errors for model and quota issues.
"""

from typing import Any

import requests
from google.api_core.exceptions import NotFound, ResourceExhausted
from langchain.schema import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


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


def generate_answer(prompt: str) -> tuple[str, str]:
    """
    Generate an answer using the configured Gemini model.

    Returns:
        (answer_text, model_used)
    """
    model_name = _normalize_model_name(settings.llm_model)

    try:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=settings.google_api_key,
            temperature=0,
        )
        response: Any = llm.invoke([HumanMessage(content=prompt)])
        answer = str(response.content).strip()
        return answer, model_name
    except ResourceExhausted as exc:
        raise RuntimeError(
            "Gemini API quota exceeded for model "
            f"'{model_name}'. Check billing/limits and retry. Error: {exc}"
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
