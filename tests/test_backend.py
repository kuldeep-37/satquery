"""Unit tests for SatQuery AI backend module.

All tests run without downloading or loading heavy model weights,
using mocked / stubbed models and token outputs.
"""

import json
import unittest
from unittest.mock import MagicMock, patch
from PIL import Image
import torch

from app.backend import (
    ComparisonResult,
    InferenceResult,
    SatQueryModel,
    export_geojson,
)


class TestInferenceResult(unittest.TestCase):
    """Test InferenceResult dataclass behavior and dictionary conversion."""

    def test_inference_result_attributes(self):
        res = InferenceResult(
            caption="satellite view of dense residential area",
            confidence=0.88,
            low_confidence=False,
            latency_sec=0.125,
            model_variant="lora",
            prompt="What is this?",
        )
        self.assertEqual(res.caption, "satellite view of dense residential area")
        self.assertAlmostEqual(res.confidence, 0.88)
        self.assertFalse(res.low_confidence)
        self.assertAlmostEqual(res.latency_sec, 0.125)
        self.assertEqual(res.model_variant, "lora")
        self.assertEqual(res.prompt, "What is this?")

    def test_to_dict(self):
        res = InferenceResult(
            caption="runway and airplane",
            confidence=0.45,
            low_confidence=True,
            latency_sec=0.089,
            model_variant="base",
        )
        data = res.to_dict()
        self.assertIsInstance(data, dict)
        self.assertEqual(data["caption"], "runway and airplane")
        self.assertEqual(data["confidence"], 0.45)
        self.assertTrue(data["low_confidence"])
        self.assertEqual(data["model_variant"], "base")
        self.assertIsNone(data["prompt"])


class TestComparisonResult(unittest.TestCase):
    """Test ComparisonResult unpacking and indexing."""

    def setUp(self):
        self.base_res = InferenceResult(
            caption="base caption",
            confidence=0.60,
            low_confidence=False,
            latency_sec=0.10,
            model_variant="base",
        )
        self.lora_res = InferenceResult(
            caption="lora caption",
            confidence=0.92,
            low_confidence=False,
            latency_sec=0.11,
            model_variant="lora",
        )

    def test_unpacking(self):
        comp = ComparisonResult(base=self.base_res, lora=self.lora_res)
        b, l = comp
        self.assertEqual(b.caption, "base caption")
        self.assertEqual(l.caption, "lora caption")

    def test_indexing(self):
        comp = ComparisonResult(base=self.base_res, lora=self.lora_res)
        self.assertEqual(comp[0].caption, "base caption")
        self.assertEqual(comp["base"].caption, "base caption")
        self.assertEqual(comp[1].caption, "lora caption")
        self.assertEqual(comp["lora"].caption, "lora caption")

    def test_none_lora(self):
        comp = ComparisonResult(base=self.base_res, lora=None)
        b, l = comp
        self.assertIsNotNone(b)
        self.assertIsNone(l)
        data = comp.to_dict()
        self.assertIsNone(data["lora"])


class TestExportGeoJSON(unittest.TestCase):
    """Test export_geojson formatting, validity, and coordinate handling."""

    def setUp(self):
        self.result = InferenceResult(
            caption="agricultural field with circular irrigation",
            confidence=0.8523,
            low_confidence=False,
            latency_sec=0.142,
            model_variant="lora",
            prompt="What crop pattern is this?",
        )

    def test_export_geojson_with_lat_lon(self):
        json_str = export_geojson(
            image_name="scene_001.tif",
            result=self.result,
            lat=37.7749,
            lon=-122.4194,
        )
        data = json.loads(json_str)

        self.assertEqual(data["type"], "Feature")
        self.assertIsNotNone(data["geometry"])
        self.assertEqual(data["geometry"]["type"], "Point")
        # In GeoJSON, coordinates are [longitude, latitude]
        self.assertEqual(data["geometry"]["coordinates"], [-122.4194, 37.7749])

        props = data["properties"]
        self.assertEqual(props["image_name"], "scene_001.tif")
        self.assertEqual(props["caption"], self.result.caption)
        self.assertAlmostEqual(props["confidence"], 0.8523)
        self.assertFalse(props["low_confidence"])
        self.assertEqual(props["model_variant"], "lora")
        self.assertEqual(props["prompt"], "What crop pattern is this?")

    def test_export_geojson_without_lat_lon(self):
        json_str = export_geojson(
            image_name="scene_002.png",
            result=self.result,
            lat=None,
            lon=None,
        )
        data = json.loads(json_str)

        self.assertEqual(data["type"], "Feature")
        self.assertIsNone(data["geometry"])
        self.assertEqual(data["properties"]["image_name"], "scene_002.png")
        self.assertEqual(data["properties"]["caption"], self.result.caption)

    def test_export_geojson_invalid_coordinates_fallback(self):
        json_str = export_geojson(
            image_name="scene_003.jpg",
            result=self.result,
            lat="invalid_lat",
            lon=12.34,
        )
        data = json.loads(json_str)
        self.assertIsNone(data["geometry"])


class TestSatQueryModel(unittest.TestCase):
    """Test SatQueryModel logic with mocked processor and generation outputs."""

    def setUp(self):
        self.dummy_image = Image.new("RGB", (64, 64), color="blue")

    def test_adapter_property_and_init(self):
        model_no_adapter = SatQueryModel(adapter_path=None, confidence_threshold=0.6)
        self.assertFalse(model_no_adapter.has_adapter)
        self.assertEqual(model_no_adapter.confidence_threshold, 0.6)

        model_empty_adapter = SatQueryModel(adapter_path="   ")
        self.assertFalse(model_empty_adapter.has_adapter)

        model_with_adapter = SatQueryModel(adapter_path="./saved_adapter")
        self.assertTrue(model_with_adapter.has_adapter)

    def _setup_mock_model(self, model: SatQueryModel, mock_caption: str, token_probs: list):
        """Helper to attach mock processor and generate output to a SatQueryModel."""
        # Mock processor
        mock_processor = MagicMock()
        mock_processor.return_value = {
            "pixel_values": torch.zeros((1, 3, 224, 224), dtype=torch.float32),
            "input_ids": torch.zeros((1, 5), dtype=torch.long),
        }
        mock_processor.decode.return_value = mock_caption
        model._processor = mock_processor

        # Mock base and lora models
        mock_base = MagicMock()
        model._base_model = mock_base
        if model.adapter_path:
            mock_lora = MagicMock()
            mock_lora.disable_adapter.return_value = MagicMock(
                __enter__=MagicMock(), __exit__=MagicMock()
            )
            model._lora_model = mock_lora
        else:
            model._lora_model = None

        model._is_loaded = True

        # Construct synthetic scores that produce given token_probs via softmax max
        # If logits has large value at index 0 and 0 elsewhere, softmax max will approximate prob
        synthetic_scores = []
        for p in token_probs:
            # We can create a tensor of logits where torch.softmax.max() is exactly p
            # log(p / (1-p)) against 0 for binary case or similar
            # Simpler: just mock torch.softmax or construct logits
            score_tensor = torch.tensor([[p * 100.0, 0.0]], dtype=torch.float32)
            synthetic_scores.append(score_tensor)

        mock_output = MagicMock()
        mock_output.sequences = torch.tensor([[101, 102]])
        mock_output.scores = synthetic_scores

        model._generate = MagicMock(return_value=mock_output)
        return model

    def test_infer_high_confidence(self):
        model = SatQueryModel(adapter_path="./adapter", confidence_threshold=0.55)
        self._setup_mock_model(
            model,
            mock_caption="dense forest canopy",
            token_probs=[0.8, 0.9, 0.85],
        )

        res = model.infer(self.dummy_image, prompt="What is here?", use_lora=True)

        self.assertIsInstance(res, InferenceResult)
        self.assertEqual(res.caption, "dense forest canopy")
        self.assertAlmostEqual(res.confidence, 1.0, places=1)  # Softmax max of 80 vs 0 is ~1.0
        self.assertFalse(res.low_confidence)
        self.assertEqual(res.model_variant, "lora")
        self.assertEqual(res.prompt, "What is here?")
        self.assertGreater(res.latency_sec, 0.0)

    def test_infer_low_confidence_flag(self):
        model = SatQueryModel(adapter_path=None, confidence_threshold=0.70)
        # Setup uniform logits so softmax max is small (e.g. 1/10 = 0.1)
        mock_processor = MagicMock()
        mock_processor.return_value = {
            "pixel_values": torch.zeros((1, 3, 224, 224)),
        }
        mock_processor.decode.return_value = "unclear terrain"
        model._processor = mock_processor
        model._base_model = MagicMock()
        model._is_loaded = True

        # 5 uniform classes -> max softmax is 0.20
        uniform_scores = [torch.zeros((1, 5)) for _ in range(3)]
        mock_output = MagicMock()
        mock_output.sequences = torch.tensor([[1, 2, 3]])
        mock_output.scores = uniform_scores
        model._generate = MagicMock(return_value=mock_output)

        res = model.infer(self.dummy_image, use_lora=False)
        self.assertAlmostEqual(res.confidence, 0.20, places=2)
        self.assertTrue(res.low_confidence)
        self.assertEqual(res.model_variant, "base")

    def test_compare_before_after_with_adapter(self):
        model = SatQueryModel(adapter_path="./adapter", confidence_threshold=0.50)
        self._setup_mock_model(
            model,
            mock_caption="harbor with shipping containers",
            token_probs=[0.9, 0.9],
        )

        comparison = model.compare_before_after(self.dummy_image, prompt="Describe the port")
        self.assertIsInstance(comparison, ComparisonResult)
        self.assertIsNotNone(comparison.base)
        self.assertIsNotNone(comparison.lora)
        self.assertEqual(comparison.base.model_variant, "base")
        self.assertEqual(comparison.lora.model_variant, "lora")

    def test_compare_before_after_without_adapter(self):
        model = SatQueryModel(adapter_path=None, confidence_threshold=0.50)
        self._setup_mock_model(
            model,
            mock_caption="generic coastline",
            token_probs=[0.7],
        )

        comparison = model.compare_before_after(self.dummy_image)
        self.assertIsNotNone(comparison.base)
        self.assertIsNone(comparison.lora)
        self.assertEqual(comparison.base.model_variant, "base")


if __name__ == "__main__":
    unittest.main()
