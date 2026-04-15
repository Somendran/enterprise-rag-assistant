import unittest
import sys
import types

from fastapi.testclient import TestClient

rag_pipeline_stub = types.ModuleType("app.services.rag_pipeline")
rag_pipeline_stub.run_rag_pipeline = lambda question, stream_callback=None: None
sys.modules.setdefault("app.services.rag_pipeline", rag_pipeline_stub)

vector_store_stub = types.ModuleType("app.services.vector_store")
vector_store_stub.load_store = lambda: None
vector_store_stub.add_documents = lambda chunks: None
vector_store_stub.is_document_indexed = lambda file_hash: False
vector_store_stub.get_indexed_document = lambda file_hash: None
vector_store_stub.list_indexed_documents = lambda: []
vector_store_stub.register_indexed_document = lambda **kwargs: None
vector_store_stub.delete_indexed_document = lambda file_hash: {
    "file_hash": file_hash,
    "filename": "handbook.pdf",
    "chunks_deleted": 0,
    "upload_path": "",
}
vector_store_stub.reset_vector_store = lambda: True
sys.modules.setdefault("app.services.vector_store", vector_store_stub)

document_loader_stub = types.ModuleType("app.services.document_loader")
document_loader_stub.load_pdf = lambda file_path: []
sys.modules.setdefault("app.services.document_loader", document_loader_stub)

text_splitter_stub = types.ModuleType("app.services.text_splitter")
text_splitter_stub.split_documents = lambda documents: []
text_splitter_stub.chunk_structured_blocks = lambda blocks: []
sys.modules.setdefault("app.services.text_splitter", text_splitter_stub)

doc_parser_stub = types.ModuleType("app.services.ingestion.doc_parser")
doc_parser_stub.parse_document = lambda file_path: []
sys.modules.setdefault("app.services.ingestion.doc_parser", doc_parser_stub)

vision_enricher_stub = types.ModuleType("app.services.ingestion.vision_enricher")
vision_enricher_stub.enrich_blocks_with_vision = lambda blocks: blocks
vision_enricher_stub.get_last_vision_calls_used = lambda: 0
sys.modules.setdefault("app.services.ingestion.vision_enricher", vision_enricher_stub)

query_cache_stub = types.ModuleType("app.services.query_cache")
query_cache_stub.clear_query_cache = lambda: None
sys.modules.setdefault("app.services.query_cache", query_cache_stub)

from app.api import query as query_module
from app.api import upload as upload_module
from app.config import Settings, settings
from app.main import app


class ApiIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.original_api_key = settings.app_api_key
        self.original_run_rag_pipeline = query_module.run_rag_pipeline
        self.original_get_indexed_document = upload_module.get_indexed_document
        self.original_delete_indexed_document = upload_module.delete_indexed_document
        self.original_delete_stored_upload = upload_module._delete_stored_upload

    def tearDown(self):
        settings.app_api_key = self.original_api_key
        query_module.run_rag_pipeline = self.original_run_rag_pipeline
        upload_module.get_indexed_document = self.original_get_indexed_document
        upload_module.delete_indexed_document = self.original_delete_indexed_document
        upload_module._delete_stored_upload = self.original_delete_stored_upload

    def test_health_is_public(self):
        settings.app_api_key = "secret"
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_protected_route_requires_api_key_when_configured(self):
        settings.app_api_key = "secret"
        response = self.client.get("/knowledge-base/files")
        self.assertEqual(response.status_code, 401)

    def test_protected_route_accepts_api_key_header(self):
        settings.app_api_key = "secret"
        response = self.client.get("/knowledge-base/files", headers={"X-API-Key": "secret"})
        self.assertEqual(response.status_code, 200)

    def test_reset_requires_auth_when_configured(self):
        settings.app_api_key = "secret"
        response = self.client.post("/knowledge-base/reset")
        self.assertEqual(response.status_code, 401)

    def test_delete_document_returns_404_for_missing_hash(self):
        settings.app_api_key = ""
        upload_module.get_indexed_document = lambda file_hash: None

        response = self.client.delete("/knowledge-base/files/missing")

        self.assertEqual(response.status_code, 404)

    def test_delete_document_removes_index_and_upload(self):
        settings.app_api_key = ""
        upload_module.get_indexed_document = lambda file_hash: {
            "file_hash": file_hash,
            "filename": "handbook.pdf",
            "upload_path": "",
        }
        upload_module.delete_indexed_document = lambda file_hash: {
            "file_hash": file_hash,
            "filename": "handbook.pdf",
            "chunks_deleted": 3,
            "upload_path": "",
        }
        upload_module._delete_stored_upload = lambda **kwargs: True

        response = self.client.delete("/knowledge-base/files/abc123")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["file_hash"], "abc123")
        self.assertEqual(payload["chunks_deleted"], 3)
        self.assertTrue(payload["upload_deleted"])

    def test_upload_rejects_bad_pdf_bytes_before_ingestion(self):
        settings.app_api_key = ""
        response = self.client.post(
            "/upload",
            files={"files": ("handbook.pdf", b"not a pdf", "application/pdf")},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["processed_files"], 0)
        self.assertEqual(payload["files"][0]["status"], "failed")
        self.assertIn("valid PDF", payload["files"][0]["message"])

    def test_query_empty_knowledge_base_returns_400(self):
        settings.app_api_key = ""

        def fake_run_rag_pipeline(question, stream_callback=None):
            raise RuntimeError("The knowledge base is empty. Please upload at least one PDF document first.")

        query_module.run_rag_pipeline = fake_run_rag_pipeline
        response = self.client.post("/query", json={"question": "What is the leave policy?"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("knowledge base is empty", response.json()["detail"])

    def test_streaming_query_emits_error_event(self):
        settings.app_api_key = ""

        def fake_run_rag_pipeline(question, stream_callback=None):
            raise RuntimeError("boom")

        query_module.run_rag_pipeline = fake_run_rag_pipeline
        response = self.client.post("/query/stream", json={"question": "What is the leave policy?"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("event: error", response.text)
        self.assertIn('"detail": "boom"', response.text)


class ProductionConfigTests(unittest.TestCase):
    def test_api_key_is_required_in_production(self):
        with self.assertRaises(ValueError):
            Settings(app_env="production", app_api_key="")

    def test_api_key_can_be_empty_in_development(self):
        settings_obj = Settings(app_env="development", app_api_key="")
        self.assertEqual(settings_obj.app_env, "development")


if __name__ == "__main__":
    unittest.main()
