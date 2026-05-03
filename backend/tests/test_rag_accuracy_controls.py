import unittest
import importlib.util
import sys
import types
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

vector_store_stub = sys.modules.get("app.services.vector_store")
if vector_store_stub is None:
    vector_store_stub = types.ModuleType("app.services.vector_store")
    sys.modules["app.services.vector_store"] = vector_store_stub
if not hasattr(vector_store_stub, "get_or_create_store"):
    vector_store_stub.get_or_create_store = lambda: None
if not hasattr(vector_store_stub, "get_knowledge_base_version"):
    vector_store_stub.get_knowledge_base_version = lambda: "test"

query_cache_stub = sys.modules.get("app.services.query_cache")
if query_cache_stub is None:
    query_cache_stub = types.ModuleType("app.services.query_cache")
    sys.modules["app.services.query_cache"] = query_cache_stub
if not hasattr(query_cache_stub, "build_cache_key"):
    query_cache_stub.build_cache_key = lambda **_kwargs: "test-cache-key"
if not hasattr(query_cache_stub, "build_prompt_fingerprint"):
    query_cache_stub.build_prompt_fingerprint = lambda _prompt: "fingerprint"
if not hasattr(query_cache_stub, "get_cached_result"):
    query_cache_stub.get_cached_result = lambda _key: None
if not hasattr(query_cache_stub, "set_cached_result"):
    query_cache_stub.set_cached_result = lambda **_kwargs: None

from app.services.retriever import classify_query
from evals.run_eval import score_eval


rag_pipeline_path = Path(__file__).resolve().parents[1] / "app" / "services" / "rag_pipeline.py"
rag_pipeline_spec = importlib.util.spec_from_file_location("test_real_rag_pipeline", rag_pipeline_path)
rag_pipeline_module = importlib.util.module_from_spec(rag_pipeline_spec)
assert rag_pipeline_spec and rag_pipeline_spec.loader
sys.modules["test_real_rag_pipeline"] = rag_pipeline_module
rag_pipeline_spec.loader.exec_module(rag_pipeline_module)
validate_source_markers = rag_pipeline_module.validate_source_markers


class QueryClassificationTests(unittest.TestCase):
    def test_summary_query_profile(self):
        self.assertEqual(classify_query("Summarize all indexed documents."), "summary")

    def test_comparison_query_profile(self):
        self.assertEqual(classify_query("Compare priority one and priority two targets."), "comparison")

    def test_lookup_query_profile(self):
        self.assertEqual(classify_query("How many days are allowed for receipts?"), "lookup")


class CitationMarkerValidationTests(unittest.TestCase):
    def test_invalid_source_marker_is_reported(self):
        invalid = validate_source_markers(
            "Use [Source 1] for this fact and [Source 9] for another.",
            {"Source 1", "Source 2"},
        )
        self.assertEqual(invalid, ["Source 9"])


class EvalScoringTests(unittest.TestCase):
    def test_negative_control_requires_refusal(self):
        item = {
            "id": "negative",
            "question": "What is the private phone number?",
            "expected_sources": [],
            "expected_keywords": [],
            "expected_answer_regex": ["i don't know|not enough"],
            "must_refuse": True,
            "min_confidence": 0.0,
            "min_sources": 0,
        }
        result = score_eval(item, {"answer": "I don't know.", "sources": [], "confidence_score": 0.0})
        self.assertTrue(result.passed)
        self.assertTrue(result.metrics["refused"])


if __name__ == "__main__":
    unittest.main()
