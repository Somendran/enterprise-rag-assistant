import asyncio
import unittest

from fastapi import HTTPException

from app.api.security import require_api_key
from app.api.upload_validation import count_pdf_pages, safe_pdf_filename, validate_pdf_upload
from app.config import settings


class UploadValidationTests(unittest.TestCase):
    def test_safe_pdf_filename_accepts_simple_pdf_name(self):
        self.assertEqual(safe_pdf_filename("handbook.pdf"), "handbook.pdf")

    def test_safe_pdf_filename_rejects_paths(self):
        self.assertIsNone(safe_pdf_filename("../handbook.pdf"))
        self.assertIsNone(safe_pdf_filename("..\\handbook.pdf"))

    def test_validate_pdf_upload_rejects_non_pdf_magic(self):
        error = validate_pdf_upload(
            filename="handbook.pdf",
            content_type="application/pdf",
            content=b"not a pdf",
            max_upload_size_bytes=1024,
            max_upload_size_mb=1,
        )
        self.assertEqual(error, "Uploaded file does not look like a valid PDF.")

    def test_validate_pdf_upload_accepts_pdf_magic(self):
        error = validate_pdf_upload(
            filename="handbook.pdf",
            content_type="application/pdf",
            content=b"%PDF-1.7\n",
            max_upload_size_bytes=1024,
            max_upload_size_mb=1,
        )
        self.assertIsNone(error)

    def test_count_pdf_pages_rejects_invalid_pdf(self):
        with self.assertRaises(ValueError):
            count_pdf_pages(b"%PDF-1.7\nnot really a pdf")


class ApiKeySecurityTests(unittest.TestCase):
    def setUp(self):
        self.original_api_key = settings.app_api_key

    def tearDown(self):
        settings.app_api_key = self.original_api_key

    def test_auth_is_disabled_when_no_key_configured(self):
        settings.app_api_key = ""
        asyncio.run(require_api_key())

    def test_auth_rejects_invalid_key_when_configured(self):
        settings.app_api_key = "secret"
        with self.assertRaises(HTTPException) as raised:
            asyncio.run(require_api_key(x_api_key="wrong"))
        self.assertEqual(raised.exception.status_code, 401)

    def test_auth_accepts_bearer_token(self):
        settings.app_api_key = "secret"
        asyncio.run(require_api_key(authorization="Bearer secret"))


if __name__ == "__main__":
    unittest.main()
