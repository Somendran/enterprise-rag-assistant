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
vector_store_stub.cleanup_expired_demo_documents = lambda: 0
vector_store_stub.is_document_indexed = lambda file_hash: False
vector_store_stub.get_indexed_document = lambda file_hash: None
vector_store_stub.list_indexed_documents = lambda: []
vector_store_stub.list_document_chunks = lambda file_hash: []
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
doc_parser_stub.parse_document = lambda file_path, force_ocr=None: []
sys.modules.setdefault("app.services.ingestion.doc_parser", doc_parser_stub)

quality_stub = types.ModuleType("app.services.ingestion.quality")
quality_stub.assess_pdf_text_quality = lambda file_path: types.SimpleNamespace(
    total_pages=0,
    text_coverage_ratio=0.0,
    low_text_pages=0,
    ocr_recommended=False,
    warnings=[],
)
quality_stub.summarize_block_text_quality = lambda blocks: (0.0, 0)
sys.modules.setdefault("app.services.ingestion.quality", quality_stub)

vision_enricher_stub = types.ModuleType("app.services.ingestion.vision_enricher")
vision_enricher_stub.enrich_blocks_with_vision = lambda blocks: blocks
vision_enricher_stub.get_last_vision_calls_used = lambda: 0
sys.modules.setdefault("app.services.ingestion.vision_enricher", vision_enricher_stub)

query_cache_stub = types.ModuleType("app.services.query_cache")
query_cache_stub.clear_query_cache = lambda: None
sys.modules.setdefault("app.services.query_cache", query_cache_stub)

metadata_store_stub = types.ModuleType("app.services.metadata_store")
metadata_store_stub.record_feedback = lambda **kwargs: {
    "id": 1,
    "created_at": 123,
    "rating": kwargs.get("rating", ""),
    "reason": kwargs.get("reason", ""),
    "comment": kwargs.get("comment", ""),
}
metadata_store_stub.admin_summary = lambda: {
    "document_count": 0,
    "chunk_count": 0,
    "feedback_count": 0,
    "recent_feedback": [],
    "metadata_db_path": "memory",
}
metadata_store_stub.users_exist = lambda: False
metadata_store_stub.get_user_for_token = lambda token: None
metadata_store_stub.can_user_access_document = lambda user, document, write=False: True
metadata_store_stub.list_documents_for_user = lambda user: []
metadata_store_stub.allowed_file_hashes_for_user = lambda user: []
metadata_store_stub.check_rate_limit = lambda **kwargs: {"allowed": True, "remaining": 1, "limit": kwargs.get("limit", 1), "reset_at": 0}
metadata_store_stub.count_documents_for_owner = lambda *args, **kwargs: 0
metadata_store_stub.list_expired_demo_documents = lambda *args, **kwargs: []
metadata_store_stub.record_audit_event = lambda **kwargs: {}
metadata_store_stub.list_audit_events = lambda limit=50: []
metadata_store_stub.create_chat_session = lambda session_id, title, user_id="": {
    "id": session_id,
    "title": title,
    "created_at": 123,
    "updated_at": 123,
    "user_id": user_id,
}
metadata_store_stub.get_chat_session = lambda session_id: None
metadata_store_stub.list_chat_sessions = lambda **kwargs: []
metadata_store_stub.add_chat_message = lambda **kwargs: {
    "id": kwargs.get("message_id"),
    "session_id": kwargs.get("session_id"),
    "role": kwargs.get("role"),
    "content": kwargs.get("content"),
    "created_at": 123,
}
metadata_store_stub.list_chat_messages = lambda session_id: []
sys.modules.setdefault("app.services.metadata_store", metadata_store_stub)

from app.api import query as query_module
from app.api import upload as upload_module
from app.config import Settings, settings
from app.main import app


class ApiIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.original_api_key = settings.app_api_key
        self.original_enable_user_auth = settings.enable_user_auth
        self.original_public_demo_mode = settings.public_demo_mode
        self.original_run_rag_pipeline = query_module.run_rag_pipeline
        self.original_get_indexed_document = upload_module.get_indexed_document
        self.original_delete_indexed_document = upload_module.delete_indexed_document
        self.original_delete_stored_upload = upload_module._delete_stored_upload
        self.original_list_document_chunks = upload_module.list_document_chunks

        settings.enable_user_auth = False
        settings.public_demo_mode = False

    def tearDown(self):
        settings.app_api_key = self.original_api_key
        settings.enable_user_auth = self.original_enable_user_auth
        settings.public_demo_mode = self.original_public_demo_mode
        query_module.run_rag_pipeline = self.original_run_rag_pipeline
        upload_module.get_indexed_document = self.original_get_indexed_document
        upload_module.delete_indexed_document = self.original_delete_indexed_document
        upload_module._delete_stored_upload = self.original_delete_stored_upload
        upload_module.list_document_chunks = self.original_list_document_chunks

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

    def test_document_chunks_endpoint_returns_chunks(self):
        settings.app_api_key = ""
        upload_module.get_indexed_document = lambda file_hash: {
            "file_hash": file_hash,
            "filename": "handbook.pdf",
        }
        upload_module.list_document_chunks = lambda file_hash, **kwargs: [
            {
                "id": "chunk-1",
                "content": "Leave policy text",
                "page": 1,
                "section": "Leave Policy",
                "chunk_index": 0,
                "metadata": {"source": "handbook.pdf"},
            }
        ]

        response = self.client.get("/knowledge-base/files/abc123/chunks")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["filename"], "handbook.pdf")
        self.assertEqual(payload["chunks"][0]["section"], "Leave Policy")

    def test_feedback_endpoint_records_feedback(self):
        settings.app_api_key = ""
        response = self.client.post(
            "/feedback",
            json={
                "question": "What is the leave policy?",
                "answer": "Employees are eligible for leave.",
                "rating": "helpful",
                "sources": [],
            },
        )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["rating"], "helpful")

    def test_admin_overview_returns_summary(self):
        settings.app_api_key = ""
        response = self.client.get("/admin/overview")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["document_count"], 0)

    def test_public_demo_session_can_list_files_without_login(self):
        settings.enable_user_auth = True
        settings.public_demo_mode = True
        settings.app_api_key = "secret"

        response = self.client.get(
            "/knowledge-base/files",
            headers={"X-Demo-Session-Id": "demo-session-1234567890"},
        )

        self.assertEqual(response.status_code, 200)

    def test_public_demo_session_cannot_call_admin_overview(self):
        settings.enable_user_auth = True
        settings.public_demo_mode = True
        settings.app_api_key = "secret"

        response = self.client.get(
            "/admin/overview",
            headers={"X-Demo-Session-Id": "demo-session-1234567890"},
        )

        self.assertEqual(response.status_code, 403)

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

        def fake_run_rag_pipeline(question, stream_callback=None, **kwargs):
            raise RuntimeError("The knowledge base is empty. Please upload at least one PDF document first.")

        query_module.run_rag_pipeline = fake_run_rag_pipeline
        response = self.client.post("/query", json={"question": "What is the leave policy?"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("knowledge base is empty", response.json()["detail"])

    def test_streaming_query_emits_error_event(self):
        settings.app_api_key = ""

        def fake_run_rag_pipeline(question, stream_callback=None, **kwargs):
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
