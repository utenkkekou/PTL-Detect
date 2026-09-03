from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from PIL import Image

from tools.review.serve_review import AppState, ConfigStore


class ReviewServerConfigTests(unittest.TestCase):
    def test_api_key_is_persisted_but_never_returned(self):
        with tempfile.TemporaryDirectory() as temp:
            config_path = Path(temp) / "config.json"
            store = ConfigStore(config_path)
            response = store.update({
                "base_url": "https://example.test/v1",
                "api_key": "test-secret-value",
                "model": "vision-model",
            })
            self.assertNotIn("api_key", response)
            self.assertTrue(response["has_api_key"])
            self.assertIn("test-secret-value", config_path.read_text(encoding="utf-8"))

            preserved = store.update({"api_key": "", "model": "vision-model-2"})
            self.assertTrue(preserved["has_api_key"])
            self.assertEqual(store.values["api_key"], "test-secret-value")

    def test_app_creates_manual_review_task_from_folder(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image_dir = root / "images"
            output_dir = root / "task"
            image_dir.mkdir()
            Image.new("RGB", (32, 24), "black").save(image_dir / "one.jpg")

            store = ConfigStore(root / "config.json")
            store.update({
                "input_dir": str(image_dir),
                "output_dir": str(output_dir),
            })
            app = AppState(store)
            total = app.start_preannotation(use_api=False)
            self.assertEqual(total, 1)

            deadline = time.time() + 5
            while app.job["status"] == "running" and time.time() < deadline:
                time.sleep(0.02)
            self.assertEqual(app.job["status"], "done", app.job)
            self.assertIsNotNone(app.review)
            self.assertEqual(len(app.review.manifest), 1)


if __name__ == "__main__":
    unittest.main()
