import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.config import settings
from app.services import embedding_service


class EmbeddingDeviceResolutionTests(unittest.TestCase):
    def setUp(self):
        self.original_embedding_device = settings.embedding_device

    def tearDown(self):
        settings.embedding_device = self.original_embedding_device

    def test_cpu_is_default(self):
        settings.embedding_device = "cpu"
        self.assertEqual(embedding_service._resolve_embedding_device(), "cpu")

    def test_cuda_is_used_when_available(self):
        settings.embedding_device = "cuda"
        with patch.object(embedding_service, "_cuda_is_available", return_value=True):
            self.assertEqual(embedding_service._resolve_embedding_device(), "cuda")

    def test_cuda_index_is_preserved_when_available(self):
        settings.embedding_device = "cuda:0"
        with patch.object(embedding_service, "_cuda_is_available", return_value=True):
            self.assertEqual(embedding_service._resolve_embedding_device(), "cuda:0")

    def test_cuda_falls_back_to_cpu_when_unavailable(self):
        settings.embedding_device = "cuda"
        with patch.object(embedding_service, "_cuda_is_available", return_value=False):
            self.assertEqual(embedding_service._resolve_embedding_device(), "cpu")

    def test_auto_uses_cuda_when_available(self):
        settings.embedding_device = "auto"
        with patch.object(embedding_service, "_cuda_is_available", return_value=True):
            self.assertEqual(embedding_service._resolve_embedding_device(), "cuda")

    def test_auto_uses_cpu_when_cuda_is_unavailable(self):
        settings.embedding_device = "auto"
        with patch.object(embedding_service, "_cuda_is_available", return_value=False):
            self.assertEqual(embedding_service._resolve_embedding_device(), "cpu")

    def test_gpu_alias_maps_to_cuda(self):
        settings.embedding_device = "gpu"
        with patch.object(embedding_service, "_cuda_is_available", return_value=True):
            self.assertEqual(embedding_service._resolve_embedding_device(), "cuda")


class CudaAvailabilityTests(unittest.TestCase):
    def test_cuda_availability_uses_torch(self):
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
        with patch.dict("sys.modules", {"torch": fake_torch}):
            self.assertTrue(embedding_service._cuda_is_available())


if __name__ == "__main__":
    unittest.main()
