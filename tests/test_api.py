"""Integration tests for FastAPI Backend endpoints /health and /analyze."""

import io
import unittest
from fastapi.testclient import TestClient
from PIL import Image

from backend.main import app


class TestBackendAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        # Enter lifespan context to initialize model once
        cls.client_context = cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_context.__exit__(None, None, None)

    def _create_test_image(self, width=256, height=256, color=(34, 139, 34)):
        """Helper to create an in-memory PNG image."""
        img = Image.new("RGB", (width, height), color=color)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return buf

    def test_health_endpoint(self):
        """Test GET /health returns ok and model loaded status."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["model_loaded"])
        self.assertTrue(data["has_adapter"])
        self.assertIn("pipeline_features", data)

    def test_analyze_endpoint_success(self):
        """Test POST /analyze with valid image, ROI coordinates and query."""
        img_buf = self._create_test_image(200, 200, color=(30, 144, 255))
        files = {"file": ("satellite_test.png", img_buf, "image/png")}
        data = {
            "x": 20.0,
            "y": 20.0,
            "width": 100.0,
            "height": 100.0,
            "query": "What land cover is visible?",
        }
        response = self.client.post("/analyze", files=files, data=data)
        self.assertEqual(response.status_code, 200)
        res = response.json()

        # Check required fields
        self.assertIn("caption", res)
        self.assertIsInstance(res["caption"], str)
        self.assertGreater(len(res["caption"]), 0)
        self.assertIn("confidence", res)
        self.assertIsInstance(res["confidence"], float)
        self.assertIn("low_confidence_flag", res)
        self.assertIsInstance(res["low_confidence_flag"], bool)

        # Check GeoJSON
        self.assertIn("geojson", res)
        geojson = res["geojson"]
        self.assertEqual(geojson["type"], "Feature")
        self.assertEqual(geojson["geometry"]["type"], "Polygon")
        self.assertEqual(len(geojson["geometry"]["coordinates"][0]), 5)

        # Check Audit
        self.assertIn("audit", res)
        audit = res["audit"]
        self.assertEqual(audit["query"], "What land cover is visible?")
        self.assertEqual(audit["roi"]["width"], 100.0)
        self.assertIn("latency_sec", audit)

    def test_analyze_endpoint_json_roi(self):
        """Test POST /analyze using JSON roi string."""
        img_buf = self._create_test_image(200, 200)
        files = {"file": ("satellite_test.png", img_buf, "image/png")}
        data = {
            "roi": '{"x": 10, "y": 15, "width": 80, "height": 80}',
            "query": "Describe the region",
        }
        response = self.client.post("/analyze", files=files, data=data)
        self.assertEqual(response.status_code, 200)
        res = response.json()
        self.assertEqual(res["audit"]["roi"]["x"], 10.0)
        self.assertEqual(res["audit"]["roi"]["y"], 15.0)

    def test_analyze_invalid_image(self):
        """Test POST /analyze with corrupted/non-image bytes returns 400."""
        corrupt_buf = io.BytesIO(b"not an image file content")
        files = {"file": ("bad.png", corrupt_buf, "image/png")}
        data = {"x": 0, "y": 0, "width": 50, "height": 50}
        response = self.client.post("/analyze", files=files, data=data)
        self.assertEqual(response.status_code, 400)
        self.assertIn("Cannot decode image", response.json()["detail"])

    def test_analyze_missing_roi(self):
        """Test POST /analyze with missing ROI returns 422."""
        img_buf = self._create_test_image(100, 100)
        files = {"file": ("test.png", img_buf, "image/png")}
        response = self.client.post("/analyze", files=files, data={})
        self.assertEqual(response.status_code, 422)

    def test_analyze_invalid_roi_dimensions(self):
        """Test POST /analyze with zero or negative width/height returns 422."""
        img_buf = self._create_test_image(100, 100)
        files = {"file": ("test.png", img_buf, "image/png")}
        data = {"x": 10, "y": 10, "width": -5, "height": 50}
        response = self.client.post("/analyze", files=files, data=data)
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
