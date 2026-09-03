import copy
import unittest

from scripts.eval_video_roi_reference import (
    audit_region,
    detection_in_roi,
    evaluate,
    validate_prediction_binding,
)


def _det(class_id, confidence=0.8, center=(15, 15)):
    x, y = center
    return {
        "class_id": class_id,
        "confidence": confidence,
        "xyxy_pixels": [x - 2, y - 2, x + 2, y + 2],
    }


class EvalVideoRoiReferenceTests(unittest.TestCase):
    def test_detection_center_controls_roi_membership(self):
        roi = [10, 10, 20, 20]
        self.assertTrue(detection_in_roi(_det(0, center=(10, 20)), roi))
        self.assertFalse(detection_in_roi(_det(0, center=(9, 20)), roi))

    def test_region_separates_clean_hit_conflict_wrong_and_miss(self):
        region = {
            "roi_xyxy_pixels": [0, 0, 30, 30],
            "expected_segments": [[0, 3, 0]],
        }
        frames = [
            {"detections": [_det(0)]},
            {"detections": [_det(0), _det(3)]},
            {"detections": [_det(1)]},
            {"detections": []},
        ]
        report = audit_region(frames, region, 0.25)
        outcomes = report["outcomes"]
        self.assertEqual(outcomes["expected_class_hit"]["frames"], 2)
        self.assertEqual(outcomes["clean_expected_class_only"]["frames"], 1)
        self.assertEqual(outcomes["expected_plus_conflicting_class"]["frames"], 1)
        self.assertEqual(outcomes["wrong_class_only"]["frames"], 1)
        self.assertEqual(outcomes["missed"]["frames"], 1)

    def test_threshold_filters_low_confidence_green_gate(self):
        region = {
            "roi_xyxy_pixels": [0, 0, 30, 30],
            "expected_segments": [[0, 0, 0], [1, 1, 1]],
        }
        frames = [
            {"detections": [_det(1, confidence=0.6)]},
            {"detections": [_det(1, confidence=0.75)]},
        ]
        low = audit_region(frames, region, 0.25)["green_gate_observation"]
        high = audit_region(frames, region, 0.70)["green_gate_observation"]
        self.assertEqual(low["green_on_non_green_frames"], 1)
        self.assertEqual(
            low["green_on_non_green_by_expected_class"], {"0": 1}
        )
        self.assertEqual(high["green_on_non_green_frames"], 0)
        self.assertEqual(high["green_hit_frames"], 1)

    def test_ignored_segment_is_not_scored_as_a_miss(self):
        region = {
            "roi_xyxy_pixels": [0, 0, 30, 30],
            "expected_segments": [[0, 0, 3], [1, 1, None]],
        }
        report = audit_region(
            [{"detections": [_det(3)]}, {"detections": []}],
            region,
            0.25,
        )
        self.assertEqual(report["source_frames"], 2)
        self.assertEqual(report["frames"], 1)
        self.assertEqual(report["ignored_frames"], 1)
        self.assertEqual(report["outcomes"]["missed"]["frames"], 0)

    def test_evaluate_requires_contiguous_stride_one_frames(self):
        reference = {
            "schema_version": 1,
            "source_frame_count": 1,
            "frame_width": 30,
            "frame_height": 30,
            "class_names": [
                "pedestrian_signal_red",
                "pedestrian_signal_green",
                "pedestrian_signal_unknown",
                "vehicle_signal",
            ],
            "regions": [
                {
                    "id": "ped",
                    "roi_xyxy_pixels": [0, 0, 30, 30],
                    "expected_segments": [[0, 0, 0]],
                }
            ],
        }
        frames = [
            {
                "source_frame_index": 2,
                "models": {
                    "m": {"schema": "clean4", "detections": []}
                },
            }
        ]
        with self.assertRaisesRegex(ValueError, "stride=1"):
            evaluate(frames, reference, [0.25])

    def test_reference_rejects_duplicate_regions_and_threshold_keys(self):
        reference = {
            "schema_version": 1,
            "source_frame_count": 1,
            "frame_width": 30,
            "frame_height": 30,
            "class_names": [
                "pedestrian_signal_red",
                "pedestrian_signal_green",
                "pedestrian_signal_unknown",
                "vehicle_signal",
            ],
            "regions": [
                {
                    "id": "same",
                    "roi_xyxy_pixels": [0, 0, 30, 30],
                    "expected_segments": [[0, 0, 0]],
                },
                {
                    "id": "same",
                    "roi_xyxy_pixels": [0, 0, 30, 30],
                    "expected_segments": [[0, 0, 0]],
                },
            ],
        }
        frames = [
            {
                "source_frame_index": 0,
                "models": {"m": {"schema": "clean4", "detections": []}},
            }
        ]
        with self.assertRaisesRegex(ValueError, "unique"):
            evaluate(frames, reference, [0.25])
        reference["regions"] = reference["regions"][:1]
        with self.assertRaisesRegex(ValueError, "collide"):
            evaluate(frames, reference, [0.251, 0.254])

    def test_prediction_binding_checks_hash_conf_and_stride(self):
        summary = {
            "video": {
                "sha256": "a" * 64,
                "sampled_frames_processed": 2,
            },
            "settings": {"stride": 1, "conf": 0.25},
            "artifacts": {"per_frame_json_sha256": "b" * 64},
            "model_order": ["m"],
            "models": {
                "m": {
                    "path": "m.pt",
                    "sha256": "c" * 64,
                    "schema": "clean4",
                }
            },
        }
        validate_prediction_binding(
            summary=summary,
            frames_sha256="b" * 64,
            video_sha256="a" * 64,
            frame_count=2,
            thresholds=[0.25, 0.70],
            models=["m"],
            model_schemas={"m": "clean4"},
        )
        with self.assertRaisesRegex(ValueError, "cannot be lower"):
            validate_prediction_binding(
                summary=summary,
                frames_sha256="b" * 64,
                video_sha256="a" * 64,
                frame_count=2,
                thresholds=[0.20],
                models=["m"],
                model_schemas={"m": "clean4"},
            )
        invalid_conf = copy.deepcopy(summary)
        invalid_conf["settings"]["conf"] = -0.1
        with self.assertRaisesRegex(ValueError, "generation conf"):
            validate_prediction_binding(
                summary=invalid_conf,
                frames_sha256="b" * 64,
                video_sha256="a" * 64,
                frame_count=2,
                thresholds=[0.25],
                models=["m"],
                model_schemas={"m": "clean4"},
            )

    def test_reference_rejects_infinite_or_out_of_frame_roi(self):
        reference = {
            "schema_version": 1,
            "source_frame_count": 1,
            "frame_width": 30,
            "frame_height": 30,
            "class_names": [
                "pedestrian_signal_red",
                "pedestrian_signal_green",
                "pedestrian_signal_unknown",
                "vehicle_signal",
            ],
            "regions": [
                {
                    "id": "bad",
                    "roi_xyxy_pixels": [0, 0, float("inf"), 20],
                    "expected_segments": [[0, 0, 0]],
                }
            ],
        }
        frames = [
            {
                "source_frame_index": 0,
                "models": {"m": {"schema": "clean4", "detections": []}},
            }
        ]
        with self.assertRaisesRegex(ValueError, "invalid ROI"):
            evaluate(frames, reference, [0.25])
        reference["regions"][0]["roi_xyxy_pixels"] = [0, 0, 31, 20]
        with self.assertRaisesRegex(ValueError, "invalid ROI"):
            evaluate(frames, reference, [0.25])


if __name__ == "__main__":
    unittest.main()
