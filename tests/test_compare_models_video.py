from __future__ import annotations

import unittest
from pathlib import Path

from scripts.compare_models_video import (
    CLEAN4_NAMES,
    LEGACY3_NAMES,
    identify_schema,
    normalize_names,
    parse_model_spec,
    parse_model_specs,
    planned_contact_indices,
    summarize_prediction_frames,
)


class CompareModelsVideoTests(unittest.TestCase):
    def test_model_specs_and_three_four_class_schemas_stay_distinct(self):
        self.assertEqual(
            parse_model_spec("old3=models/old.pt"),
            ("old3", Path("models/old.pt")),
        )
        self.assertEqual(identify_schema(dict(enumerate(LEGACY3_NAMES)))[1], "legacy3")
        self.assertEqual(identify_schema(CLEAN4_NAMES)[1], "clean4")

        with self.assertRaises(ValueError):
            parse_model_specs(["same=a.pt", "same=b.pt"])
        with self.assertRaises(ValueError):
            parse_model_specs(["only=a.pt"])
        with self.assertRaises(ValueError):
            identify_schema(["red", "green", "unknown"])

    def test_class_ids_must_be_contiguous_and_names_unique(self):
        with self.assertRaises(ValueError):
            normalize_names({0: "red", 2: "unknown"})
        with self.assertRaises(ValueError):
            normalize_names({0: "same", 1: "same"})

    def test_stability_summary_reports_runs_gaps_and_strict_flicker(self):
        # class 0 presence: off,on,off,on,on,off
        frames = [
            {"detections": [], "inference_ms": 1, "wall_ms": 3},
            {
                "detections": [{"class_id": 0, "confidence": 0.6}],
                "inference_ms": 2,
                "wall_ms": 4,
            },
            {"detections": [], "inference_ms": 3, "wall_ms": 5},
            {
                "detections": [
                    {"class_id": 0, "confidence": 0.8},
                    {"class_id": 1, "confidence": 0.7},
                ],
                "inference_ms": 4,
                "wall_ms": 6,
            },
            {
                "detections": [{"class_id": 0, "confidence": 0.9}],
                "inference_ms": 5,
                "wall_ms": 7,
            },
            {"detections": [], "inference_ms": 6, "wall_ms": 8},
        ]
        summary = summarize_prediction_frames(frames, LEGACY3_NAMES, 0.5)
        red = summary["per_class"]["0"]

        self.assertEqual(summary["sampled_frames"], 6)
        self.assertEqual(summary["detections"]["total"], 4)
        self.assertAlmostEqual(red["frame_coverage"]["rate"], 0.5)
        self.assertEqual(
            red["continuity"]["longest_presence_run_sampled_frames"], 2
        )
        self.assertEqual(
            red["continuity"]["longest_absence_gap_sampled_frames"], 1
        )
        self.assertEqual(red["continuity"]["state_transition_count"], 4)
        self.assertEqual(red["continuity"]["one_sample_presence_spikes"], 1)
        self.assertEqual(red["continuity"]["one_sample_absence_dropouts"], 1)
        self.assertEqual(red["continuity"]["flicker_event_count"], 2)
        self.assertAlmostEqual(
            red["continuity"]["longest_presence_run_approx_seconds"], 1.0
        )
        self.assertEqual(
            summary["per_class"]["2"]["continuity"][
                "longest_absence_gap_sampled_frames"
            ],
            6,
        )

    def test_contact_indices_are_even_and_bounded(self):
        # 100 source frames / stride 10 -> 10 processed frames.
        self.assertEqual(
            planned_contact_indices(100, 10, 0, 3),
            {0, 4, 9},
        )
        self.assertEqual(
            planned_contact_indices(100, 10, 5, 8),
            {0, 1, 2, 3, 4},
        )
        self.assertEqual(planned_contact_indices(100, 10, 0, 1), {0})
        self.assertEqual(planned_contact_indices(0, 3, 0, 2), {0, 1})


if __name__ == "__main__":
    unittest.main()
