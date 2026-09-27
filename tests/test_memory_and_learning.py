"""Unit & Integration tests for QA Memory storage and Continuous Learning Pipeline."""

import io
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from PIL import Image
import torch

from backend.qa_memory import (
    init_memory_db,
    save_qa_record,
    get_all_records,
    get_memory_stats,
    get_relevant_memory,
    mark_as_trained,
    clear_memory,
)
from backend.continuous_learner import (
    MemoryQADataset,
    prepare_training_samples,
    train_lora_on_memory,
)
from backend.main import app


class TestQAMemoryStore(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "test_memory.db")
        self.crops_dir = os.path.join(self.test_dir, "test_crops")
        init_memory_db(self.db_path, self.crops_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_save_and_retrieve_record(self):
        img = Image.new("RGB", (64, 64), color="blue")
        record = save_qa_record(
            cropped_image=img,
            question="What is the water condition?",
            answer="Clear river water flowing through agricultural valley.",
            short_caption="river and agricultural parcel",
            roi={"x": 10, "y": 10, "width": 50, "height": 50},
            confidence=0.89,
            intent="water_hydrology",
            language="en",
            image_filename="test_sat.png",
            db_path=self.db_path,
            crops_dir=self.crops_dir,
        )

        self.assertEqual(record["id"], 1)
        self.assertEqual(record["question"], "What is the water condition?")
        self.assertTrue(os.path.exists(record["crop_path"]))

        stats = get_memory_stats(self.db_path)
        self.assertEqual(stats["total_records"], 1)
        self.assertEqual(stats["trained_records"], 0)
        self.assertEqual(stats["pending_training"], 1)

        records = get_all_records(self.db_path)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["short_caption"], "river and agricultural parcel")

    def test_relevant_memory_retrieval(self):
        img = Image.new("RGB", (64, 64), color="green")
        save_qa_record(
            cropped_image=img,
            question="Assess wildfire hazard and aridity.",
            answer="Dense forest canopy with moderate moisture.",
            short_caption="forest canopy",
            roi={"x": 0, "y": 0, "width": 64, "height": 64},
            confidence=0.91,
            intent="wildfire_burn_risk",
            language="en",
            db_path=self.db_path,
            crops_dir=self.crops_dir,
        )

        matches = get_relevant_memory("wildfire risk and burn danger", db_path=self.db_path)
        self.assertEqual(len(matches), 1)
        self.assertIn("wildfire", matches[0]["question"].lower())

    def test_mark_as_trained_and_clear(self):
        img = Image.new("RGB", (64, 64), color="red")
        rec = save_qa_record(
            cropped_image=img,
            question="Urban buildings and roads?",
            answer="Dense residential settlement.",
            short_caption="residential buildings",
            roi={"x": 5, "y": 5, "width": 40, "height": 40},
            confidence=0.82,
            intent="urban_infrastructure",
            language="en",
            db_path=self.db_path,
            crops_dir=self.crops_dir,
        )

        mark_as_trained([rec["id"]], db_path=self.db_path)
        stats = get_memory_stats(self.db_path)
        self.assertEqual(stats["trained_records"], 1)
        self.assertEqual(stats["pending_training"], 0)

        clear_memory(self.db_path, self.crops_dir)
        stats_cleared = get_memory_stats(self.db_path)
        self.assertEqual(stats_cleared["total_records"], 0)


class TestMemoryDatasetAndLearner(unittest.TestCase):
    def test_prepare_samples(self):
        records = [
            {
                "id": 101,
                "crop_path": "nonexistent.png",
                "question": "What is here?",
                "short_caption": "forest",
            }
        ]
        samples = prepare_training_samples(records)
        self.assertEqual(len(samples), 0)


class TestAPIEndpointsMemory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.client_context = cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_context.__exit__(None, None, None)

    def test_get_memory_endpoint(self):
        res = self.client.get("/memory")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("stats", data)
        self.assertIn("records", data)
        self.assertIsInstance(data["records"], list)


if __name__ == "__main__":
    unittest.main()
