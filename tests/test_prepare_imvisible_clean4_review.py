from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from scripts.prepare_imvisible_clean4_review import (
    MODE_TARGET,
    assign_scene_ids,
    ensure_fresh_output,
    exclude_testing_scenes,
    has_cross_class_overlap,
    merge_untrusted_candidates,
    select_with_none_quota,
)


def record(
    filename: str,
    split: str,
    mode: int,
    sha: str,
    *,
    gps=None,
    contributor="john",
    minute=0,
):
    return {
        "file": filename,
        "split": split,
        "mode": mode,
        "sha256": sha,
        "gps": gps,
        "contributor": contributor,
        "captured_at": datetime(2020, 1, 1) + timedelta(minutes=minute),
    }


class PrepareImVisibleClean4ReviewTests(unittest.TestCase):
    def test_countdown_blank_has_no_safe_class_target(self):
        self.assertEqual(MODE_TARGET[3], -1)

    def test_testing_member_excludes_whole_transitive_scene(self):
        rows = [
            record("train-near.jpg", "training", 0, "a", gps=(31.0, 121.0)),
            record(
                "test-near.jpg",
                "testing",
                0,
                "b",
                gps=(31.0002, 121.0),
            ),
            record(
                "train-far.jpg",
                "training",
                1,
                "c",
                gps=(31.01, 121.0),
            ),
        ]
        grouped = assign_scene_ids(rows)
        eligible, report = exclude_testing_scenes(grouped)
        self.assertEqual([item["file"] for item in eligible], ["train-far.jpg"])
        self.assertEqual(report["excluded_non_testing_image_count"], 1)

    def test_no_gps_training_frame_links_to_timed_testing_gps_neighbor(self):
        rows = [
            record(
                "train-no-gps.jpg",
                "training",
                0,
                "a",
                gps=None,
                contributor="sam",
                minute=0,
            ),
            record(
                "test-with-gps.jpg",
                "testing",
                0,
                "b",
                gps=(31.0, 121.0),
                contributor="sam",
                minute=9,
            ),
            record(
                "train-other.jpg",
                "training",
                1,
                "c",
                gps=None,
                contributor="john",
                minute=9,
            ),
        ]
        eligible, _ = exclude_testing_scenes(assign_scene_ids(rows))
        self.assertEqual([item["file"] for item in eligible], ["train-other.jpg"])

    def test_legacy_candidate_merge_deduplicates_same_class_only(self):
        a = {"cls": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}
        near = {"cls": 0, "cx": 0.51, "cy": 0.5, "w": 0.2, "h": 0.2}
        conflict = {"cls": 3, "cx": 0.51, "cy": 0.5, "w": 0.2, "h": 0.2}
        merged = merge_untrusted_candidates(
            [("seed_progress", [a]), ("reann_progress", [near, conflict])]
        )
        self.assertEqual(len(merged), 2)
        self.assertEqual(
            merged[0]["candidate_sources"],
            ["seed_progress", "reann_progress"],
        )
        self.assertEqual(merged[1]["cls"], 3)
        self.assertTrue(all(box["untrusted_seed"] for box in merged))
        self.assertTrue(has_cross_class_overlap(merged))

    def test_fail_closed_and_none_scene_diversity(self):
        with self.assertRaises(ValueError):
            assign_scene_ids(
                [
                    record("red.jpg", "training", 0, "same"),
                    record("green.jpg", "training", 1, "same"),
                ]
            )
        eligible = [
            {
                "file": f"none-{scene}-{index}.jpg",
                "mode": 4,
                "scene_id": scene,
            }
            for scene in ("a", "b", "c")
            for index in range(2)
        ]
        selected, report = select_with_none_quota(eligible, none_limit=2)
        self.assertEqual(len(selected), 2)
        self.assertEqual(len({item["scene_id"] for item in selected}), 2)
        self.assertEqual(report["quota_excluded_none"], 4)
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory)
            with self.assertRaises(FileExistsError):
                ensure_fresh_output(existing)


if __name__ == "__main__":
    unittest.main()
