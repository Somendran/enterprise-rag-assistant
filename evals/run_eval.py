"""Run a small RAG quality check against the backend API.

By default this validates the eval file only. Use --live to call the backend.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request


DEFAULT_EVAL_FILE = Path(__file__).with_name("questions.json")


@dataclass
class EvalResult:
    eval_id: str
    passed: bool
    message: str


def load_evals(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Eval file must contain a JSON array.")

    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Eval item {idx} must be an object.")
        if not item.get("id") or not item.get("question"):
            raise ValueError(f"Eval item {idx} must include id and question.")
        item.setdefault("expected_sources", [])
        item.setdefault("expected_keywords", [])
        item.setdefault("expected_answer_regex", [])
        item.setdefault("min_confidence", 0.0)
        item.setdefault("min_sources", 0)
    return payload


def call_query(api_url: str, api_key: str, question: str, timeout: int) -> dict[str, Any]:
    body = json.dumps({"question": question}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    req = request.Request(
        f"{api_url.rstrip('/')}/query",
        data=body,
        headers=headers,
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def source_matches(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    expected_document = str(expected.get("document", "")).lower()
    expected_page = expected.get("page")
    actual_document = str(actual.get("document", "")).lower()
    actual_page = actual.get("page")

    if expected_document and expected_document not in actual_document:
        return False
    if expected_page is not None and int(expected_page) != int(actual_page):
        return False
    return True


def score_eval(item: dict[str, Any], payload: dict[str, Any]) -> EvalResult:
    answer = str(payload.get("answer", "") or "")
    answer_lower = answer.lower()
    sources = payload.get("sources", []) or []

    missing_keywords = [
        keyword
        for keyword in item.get("expected_keywords", [])
        if str(keyword).lower() not in answer_lower
    ]
    missing_sources = [
        expected
        for expected in item.get("expected_sources", [])
        if not any(source_matches(expected, actual) for actual in sources)
    ]
    missing_patterns = [
        pattern
        for pattern in item.get("expected_answer_regex", [])
        if not re.search(str(pattern), answer, re.IGNORECASE | re.MULTILINE)
    ]
    confidence = payload.get("confidence_score")
    try:
        confidence_value = float(confidence)
    except (TypeError, ValueError):
        confidence_value = 0.0
    min_confidence = float(item.get("min_confidence", 0.0) or 0.0)
    min_sources = int(item.get("min_sources", 0) or 0)

    failures: list[str] = []
    if not answer.strip():
        failures.append("empty answer")
    if missing_keywords:
        failures.append(f"missing keywords: {missing_keywords}")
    if missing_sources:
        failures.append(f"missing sources: {missing_sources}")
    if missing_patterns:
        failures.append(f"missing regex matches: {missing_patterns}")
    if confidence_value < min_confidence:
        failures.append(f"confidence {confidence_value:.3f} below {min_confidence:.3f}")
    if len(sources) < min_sources:
        failures.append(f"sources {len(sources)} below {min_sources}")

    if failures:
        return EvalResult(str(item["id"]), False, "; ".join(failures))

    return EvalResult(
        str(item["id"]),
        True,
        f"answer_chars={len(answer)} sources={len(sources)} confidence={confidence_value:.3f}",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RAG eval checks.")
    parser.add_argument("--file", default=str(DEFAULT_EVAL_FILE), help="Path to eval JSON file.")
    parser.add_argument("--api-url", default="http://localhost:8000", help="Backend API base URL.")
    parser.add_argument("--api-key", default="", help="Optional backend API key.")
    parser.add_argument("--timeout", type=int, default=120, help="Per-query timeout in seconds.")
    parser.add_argument("--live", action="store_true", help="Call the backend /query endpoint.")
    args = parser.parse_args()

    evals = load_evals(Path(args.file))

    if not args.live:
        print(f"Loaded {len(evals)} evals from {args.file}.")
        print("Dry run only. Pass --live to call the backend /query endpoint.")
        return 0

    results: list[EvalResult] = []
    for item in evals:
        try:
            payload = call_query(args.api_url, args.api_key, str(item["question"]), args.timeout)
            result = score_eval(item, payload)
        except Exception as exc:
            result = EvalResult(str(item["id"]), False, str(exc))
        results.append(result)
        status = "PASS" if result.passed else "FAIL"
        print(f"{status} {result.eval_id}: {result.message}")

    failed = [result for result in results if not result.passed]
    print(f"\nSummary: {len(results) - len(failed)}/{len(results)} passed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
