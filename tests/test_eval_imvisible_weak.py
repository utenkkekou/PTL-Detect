from __future__ import annotations

import unittest

from scripts.eval_imvisible_weak import (
    deduplicate_by_sha,
    summarize_predictions,
)


class ImVisibleWeakEvaluationTests(unittest.TestCase):
    def test_image_level_summary_and_none_ignores_vehicle_class(self):
        records = [
            {"file": "none_vehicle.jpg", "mode": "4", "predicted_classes": [3]},
            {"file": "none_green.jpg", "mode": "4", "predicted_classes": [1, 3]},
            {"file": "red_hit.jpg", "mode": "0", "predicted_classes": [0, 3]},
            {"file": "red_wrong_ped.jpg", "mode": "0", "predicted_classes": [2]},
            {"file": "red_miss.jpg", "mode": "0", "predicted_classes": []},
            {"file": "green_hit.jpg", "mode": "1", "predicted_classes": [1]},
            {"file": "green_miss.jpg", "mode": "1", "predicted_classes": [3]},
            {"file": "count_hit.jpg", "mode": "2", "predicted_classes": [1, 3]},
            {"file": "count_wrong.jpg", "mode": "2", "predicted_classes": [0]},
            {"file": "blank.jpg", "mode": "3", "predicted_classes": [2]},
        ]

        summary = summarize_predictions(records)

        none = summary["none"]
        self.assertEqual(none["total_images"], 2)
        self.assertEqual(none["pedestrian_false_positive_images"], 1)
        self.assertEqual(none["pedestrian_false_positive_rate"], 0.5)
        self.assertEqual(none["vehicle_signal_image_hits_ignored"], 2)

        red = summary["positive_modes"]["red"]
        self.assertEqual(red["same_color_hit_images"], 1)
        self.assertEqual(red["any_pedestrian_hit_images"], 2)
        self.assertAlmostEqual(red["same_color_hit_rate"], 1 / 3)
        self.assertAlmostEqual(red["any_pedestrian_hit_rate"], 2 / 3)

        green = summary["positive_modes"]["green"]
        self.assertEqual(green["same_color_hit_rate"], 0.5)
        self.assertEqual(green["any_pedestrian_hit_rate"], 0.5)

        countdown = summary["positive_modes"]["countdown_green"]
        self.assertEqual(countdown["same_color_hit_rate"], 0.5)
        self.assertEqual(countdown["any_pedestrian_hit_rate"], 1.0)
        self.assertEqual(
            summary["countdown_blank_unscored"]["total_images"], 1
        )

    def test_sha_dedup_is_deterministic_and_rejects_mode_conflict(self):
        records = [
            {"file": "first.jpg", "mode": "0", "sha256": "aa"},
            {"file": "same.jpg", "mode": "0", "sha256": "aa"},
            {"file": "other.jpg", "mode": "1", "sha256": "bb"},
        ]
        unique, groups = deduplicate_by_sha(records)
        self.assertEqual(
            [record["file"] for record in unique],
            ["first.jpg", "other.jpg"],
        )
        self.assertEqual(groups[0]["kept"], "first.jpg")
        self.assertEqual(groups[0]["removed"], ["same.jpg"])

        with self.assertRaises(ValueError):
            deduplicate_by_sha(
                [
                    {"file": "red.jpg", "mode": "0", "sha256": "same"},
                    {"file": "green.jpg", "mode": "1", "sha256": "same"},
                ]
            )


if __name__ == "__main__":
    unittest.main()
