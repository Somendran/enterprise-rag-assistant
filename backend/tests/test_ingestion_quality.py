import unittest
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.api import upload as upload_module
from app.config import settings

sys.modules.setdefault("pdfplumber", SimpleNamespace(open=lambda *_args, **_kwargs: None))
quality_path = Path(__file__).resolve().parents[1] / "app" / "services" / "ingestion" / "quality.py"
quality_spec = importlib.util.spec_from_file_location("test_real_ingestion_quality", quality_path)
quality_module = importlib.util.module_from_spec(quality_spec)
assert quality_spec and quality_spec.loader
sys.modules["test_real_ingestion_quality"] = quality_module
quality_spec.loader.exec_module(quality_module)
IngestionQuality = quality_module.IngestionQuality
assess_pdf_text_quality = quality_module.assess_pdf_text_quality


class _FakePdf:
    def __init__(self, texts):
        self.pages = [SimpleNamespace(extract_text=lambda text=text: text) for text in texts]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class IngestionQualityTests(unittest.TestCase):
    def test_scanned_detection_flags_mostly_low_text_pages(self):
        with patch.object(quality_module.pdfplumber, "open", return_value=_FakePdf(["", "x", "Policy text " * 30])):
            quality = assess_pdf_text_quality("scan.pdf")

        self.assertTrue(quality.ocr_recommended)
        self.assertEqual(quality.low_text_pages, 2)
        self.assertLess(quality.text_coverage_ratio, 0.5)
        self.assertTrue(quality.warnings)

    def test_scanned_detection_accepts_text_searchable_pdf(self):
        with patch.object(quality_module.pdfplumber, "open", return_value=_FakePdf(["Policy text " * 30, "Benefits " * 40])):
            quality = assess_pdf_text_quality("text.pdf")

        self.assertFalse(quality.ocr_recommended)
        self.assertEqual(quality.low_text_pages, 0)
        self.assertEqual(quality.text_coverage_ratio, 1.0)


class UploadIngestionRetryTests(unittest.TestCase):
    def setUp(self):
        self.original_enable_docling = settings.enable_docling
        self.original_enable_ocr = settings.enable_ocr
        self.original_enable_vision = settings.enable_vision_enrichment

    def tearDown(self):
        settings.enable_docling = self.original_enable_docling
        settings.enable_ocr = self.original_enable_ocr
        settings.enable_vision_enrichment = self.original_enable_vision

    def test_docling_retries_with_ocr_only_when_coverage_is_poor(self):
        settings.enable_docling = True
        settings.enable_ocr = False
        settings.enable_vision_enrichment = False
        calls = []

        def fake_parse(_path, force_ocr=None):
            calls.append(force_ocr)
            if force_ocr:
                return [{"type": "paragraph", "content": "OCR recovered policy text.", "page": 1, "source": "scan.pdf"}]
            return [{"type": "paragraph", "content": "", "page": 1, "source": "scan.pdf"}]

        def fake_summarize(blocks):
            return (1.0, 0) if blocks and blocks[0].get("content") else (0.0, 1)

        with (
            patch.object(upload_module, "assess_pdf_text_quality", return_value=IngestionQuality(
                total_pages=1,
                text_coverage_ratio=0.0,
                low_text_pages=1,
                ocr_recommended=True,
                warnings=["scanned"],
            )),
            patch.object(upload_module, "parse_document", side_effect=fake_parse),
            patch.object(upload_module, "summarize_block_text_quality", side_effect=fake_summarize),
            patch.object(upload_module, "chunk_structured_blocks", return_value=[
                SimpleNamespace(page_content="OCR recovered policy text.", metadata={})
            ]),
        ):
            prepared = upload_module._prepare_chunks_for_indexing(
                file_path=Path("scan.pdf"),
                filename="scan.pdf",
                file_hash="hash",
                document_id="doc",
            )

        self.assertEqual(calls, [False, True])
        self.assertEqual(prepared.parsing_method, "docling_ocr")
        self.assertTrue(prepared.ocr_applied)
        self.assertEqual(prepared.text_coverage_ratio, 1.0)

    def test_upload_result_includes_ingestion_quality_metadata(self):
        prepared = upload_module.PreparedChunks(
            chunks=[SimpleNamespace(page_content="Recovered text", metadata={})],
            parsing_method="docling_ocr",
            vision_calls_used=0,
            ocr_applied=True,
            text_coverage_ratio=0.5,
            low_text_pages=1,
            ingestion_warnings=["Document appears scanned."],
        )

        with (
            patch.object(upload_module, "is_document_indexed", return_value=False),
            patch.object(upload_module, "_prepare_chunks_for_indexing", return_value=prepared),
            patch.object(upload_module, "add_documents", return_value=None),
            patch.object(upload_module, "register_indexed_document", return_value=None),
        ):
            response = upload_module._ingest_saved_files([
                {
                    "filename": "scan.pdf",
                    "file_hash": "hash",
                    "document_id": "doc",
                    "file_path": "scan.pdf",
                }
            ])

        item = response.files[0]
        self.assertEqual(item.parsing_method, "docling_ocr")
        self.assertTrue(item.ocr_applied)
        self.assertEqual(item.text_coverage_ratio, 0.5)
        self.assertEqual(item.low_text_pages, 1)
        self.assertEqual(item.ingestion_warnings, ["Document appears scanned."])

    def test_empty_scanned_document_fails_with_clear_message(self):
        settings.enable_docling = False
        with (
            patch.object(upload_module, "assess_pdf_text_quality", return_value=IngestionQuality(
                total_pages=1,
                text_coverage_ratio=0.0,
                low_text_pages=1,
                ocr_recommended=True,
                warnings=["scanned"],
            )),
            patch.object(upload_module, "load_pdf", return_value=[]),
            patch.object(upload_module, "split_documents", return_value=[]),
        ):
            with self.assertRaises(ValueError) as raised:
                upload_module._prepare_chunks_for_indexing(
                    file_path=Path("scan.pdf"),
                    filename="scan.pdf",
                    file_hash="hash",
                    document_id="doc",
                )

        self.assertIn("appears to be scanned", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
