from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.api_pre_annotate import (
    ApiSettings,
    build_manifest,
    discover_images,
    parse_api_boxes,
)


class ApiPreAnnotateTests(unittest.TestCase):
    def test_parse_boxes_accepts_0_to_1000_coordinates(self):
        boxes = parse_api_boxes(json.dumps({
            "boxes": [{
                "label": "pedestrian_green",
                "bbox": [100, 200, 300, 600],
                "confidence": 0.91,
            }],
        }))
        self.assertEqual(boxes[0]["cls"], 1)
        self.assertAlmostEqual(boxes[0]["cx"], 0.2)
        self.assertAlmostEqual(boxes[0]["h"], 0.4)

    def test_parse_boxes_ignores_unknown_labels_and_invalid_boxes(self):
        boxes = parse_api_boxes({
            "boxes": [
                {"label": "tree", "bbox": [0, 0, 100, 100]},
                {"label": "vehicle_signal", "bbox": [500, 500, 500, 700]},
                {"label": "人行红灯", "bbox": [0.1, 0.2, 0.3, 0.4]},
            ],
        })
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0]["cls"], 0)

    def test_manual_manifest_needs_no_api_or_model(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            images = root / "images"
            nested = images / "nested"
            nested.mkdir(parents=True)
            Image.new("RGB", (40, 30), "white").save(nested / "sample.png")
            (nested / "ignore.txt").write_text("not an image", encoding="utf-8")

            found = discover_images(images)
            self.assertEqual([path.name for path in found], ["sample.png"])

            output = root / "output"
            manifest_path = build_manifest(
                images,
                output,
                ApiSettings("", "", ""),
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(len(manifest), 1)
            self.assertEqual(manifest[0]["boxes"], [])
            self.assertEqual(manifest[0]["source"], "manual")
            self.assertTrue((output / manifest[0]["preview"]).exists())


if __name__ == "__main__":
    unittest.main()
