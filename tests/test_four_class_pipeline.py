from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from scripts import export_coreml
from scripts.prepare_four_class_review import reviewed_seed
from scripts.review_subset import (
    ATTESTATION,
    apply_suggestions,
    build_subset,
    main as review_subset_main,
    merge_subset,
)
from scripts.split_dataset import assign_splits, extract_features
from scripts.video_to_dataset import (
    _force_temporal_sample,
    _normalized_box,
    _validate_unique_video_stems,
    load_excluded_videos,
    location_of,
    main as video_to_dataset_main,
    parse_scene_aliases,
)
from tools.review.serve_review import State


EXPECTED = {
    0: "pedestrian_signal_red",
    1: "pedestrian_signal_green",
    2: "pedestrian_signal_unknown",
    3: "vehicle_signal",
}


class FourClassContractTests(unittest.TestCase):
    def test_coreml_contract_is_exactly_four_classes(self):
        self.assertEqual(export_coreml.EXPECTED_NAMES, EXPECTED)

    def test_review_ui_puts_signal_type_before_color(self):
        html = Path("tools/review/index.html").read_text(encoding="utf-8")
        self.assertIn("先分灯种类，再看红绿/明暗", html)
        self.assertIn("仅人行灯状态不明", html)
        self.assertIn("车行/自行车信号灯（任意状态）", html)
        self.assertIn('answer !== "人行"', html)
        self.assertIn("requires_change", html)
        self.assertIn("vehicle_only", html)
        self.assertIn("WEAK_HINTS", html)
        self.assertIn("右侧弱提示", html)
        self.assertIn("预标框照常显示", html)
        self.assertIn("隐藏候选框", html)

    def test_reviewed_seed_keeps_human_boxes_and_rejects_class3_seed(self):
        manifest = [{"id": "a", "boxes": [{"cls": 1}]}]
        progress = {
            "a": {
                "status": "reviewed",
                "boxes": [{"cls": 2, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1}],
            }
        }
        seeded = reviewed_seed(manifest, progress)
        self.assertEqual(seeded[0][1][0]["cls"], 2)
        progress["a"]["boxes"][0]["cls"] = 3
        with self.assertRaises(ValueError):
            reviewed_seed(manifest, progress)

    def test_review_stats_ignore_progress_ids_outside_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = root / "manifest.json"
            progress = root / "progress.json"
            manifest.write_text(json.dumps([
                {"id": "a", "boxes": [{"cls": 3}]},
                {"id": "b", "boxes": []},
            ]), encoding="utf-8")
            progress.write_text(json.dumps({
                "a": {"status": "reviewed", "boxes": [{"cls": 3}]},
                "other-task": {"status": "reviewed", "boxes": [{"cls": 0}]},
            }), encoding="utf-8")
            stats = State(manifest, root, progress).stats()
            self.assertEqual(stats["total"], 2)
            self.assertEqual(stats["reviewed"], 1)
            self.assertEqual(stats["cls_vehicle"], 1)

    def test_t10_filename_preserves_numbered_scene_without_index(self):
        a = extract_features(Path("frames/公司楼下2/公司楼下2__00001.jpg"))
        b = extract_features(Path("frames/公司楼下3/公司楼下3__00002.jpg"))
        self.assertEqual(a["scene_hint"], "公司楼下2")
        self.assertEqual(b["scene_hint"], "公司楼下3")

    def test_t10_scene_index_can_merge_known_video_segments(self):
        hints = {
            "公司楼下2__00001": "公司楼下",
            "公司楼下3__00002": "公司楼下",
        }
        a = extract_features(
            Path("frames/公司楼下2/公司楼下2__00001.jpg"), hints)
        b = extract_features(
            Path("frames/公司楼下3/公司楼下3__00002.jpg"), hints)
        self.assertEqual(a["scene_hint"], "公司楼下")
        self.assertEqual(b["scene_hint"], "公司楼下")

    def test_incremental_video_scene_names_are_not_merged_by_default(self):
        self.assertEqual(location_of("未知路口2"), "未知路口2")
        self.assertEqual(
            location_of("公司楼下3", merge_numbered_suffix=True), "公司楼下")
        self.assertEqual(
            parse_scene_aliases(["充电站左=充电站", "充电站右=充电站"]),
            {"充电站左": "充电站", "充电站右": "充电站"},
        )
        with self.assertRaises(ValueError):
            parse_scene_aliases(["充电站左=充电站", "充电站左=别处"])

    def test_multiple_exclude_indices_union_with_explicit_duplicates(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            first = root / "first.json"
            second = root / "second.json"
            first.write_text(
                json.dumps([{"video": "首批A"}, {"video": "首批B"}]),
                encoding="utf-8",
            )
            second.write_text(
                json.dumps([{"video": "二批A"}]), encoding="utf-8")
            excluded = load_excluded_videos(
                [str(first), str(second)], ["未知路口4", "未知路口5"])
            self.assertEqual(
                excluded,
                {"首批A", "首批B", "二批A", "未知路口4", "未知路口5"},
            )

    def test_frames_only_refuses_to_overwrite_existing_index(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            (out / "_frames_index.json").write_text("[]", encoding="utf-8")
            with patch(
                "sys.argv",
                ["video_to_dataset.py", "--out", str(out), "--frames-only"],
            ):
                with self.assertRaises(FileExistsError):
                    video_to_dataset_main()
            with patch(
                "sys.argv",
                [
                    "video_to_dataset.py",
                    "--out",
                    str(out),
                    "--scene-alias",
                    "视频A=路口A",
                ],
            ):
                with self.assertRaises(ValueError):
                    video_to_dataset_main()

    def test_normalized_box_rounding_stays_inside_image(self):
        box = _normalized_box(3, 0, 0.169406, 0.079403, 0.191426, 0.8522)
        self.assertGreaterEqual(box["cx"] - box["w"] / 2, 0)
        self.assertGreaterEqual(box["cy"] - box["h"] / 2, 0)
        self.assertLessEqual(box["cx"] + box["w"] / 2, 1)
        self.assertLessEqual(box["cy"] + box["h"] / 2, 1)
        with self.assertRaises(ValueError):
            _normalized_box(3, float("nan"), 0, 1, 1, 0.5)

    def test_dedup_forces_periodic_and_tail_samples(self):
        self.assertTrue(_force_temporal_sample(1, None, 100, 2, 10))
        self.assertFalse(_force_temporal_sample(19, 1, 100, 2, 10))
        self.assertTrue(_force_temporal_sample(21, 1, 100, 2, 10))
        self.assertFalse(_force_temporal_sample(100, 98, 100, 2, 10))
        self.assertTrue(_force_temporal_sample(100, 96, 100, 2, 10))
        with self.assertRaises(ValueError):
            _validate_unique_video_stems(
                [Path("Same.mp4"), Path("same.mov")]
            )

    def test_small_scene_split_hits_global_optimum(self):
        sizes = [126, 115, 53, 27, 23, 15, 13, 2]
        scene_of = [scene for scene, size in enumerate(sizes) for _ in range(size)]
        _, counts = assign_splits(scene_of)
        self.assertEqual(counts, {"train": 264, "val": 55, "test": 55})

    def test_many_scene_multistart_avoids_greedy_local_minimum(self):
        sizes = [100, 80, 60, 40, 30, 20, 15, 12, 10, 8, 6, 4, 2]
        scene_of = [scene for scene, size in enumerate(sizes) for _ in range(size)]
        _, counts = assign_splits(scene_of)
        self.assertEqual(counts, {"train": 271, "val": 58, "test": 58})

    def test_class_aware_split_covers_rare_class_in_all_splits(self):
        sizes = [2, 23, 115, 27, 15, 126, 13, 53]
        per_scene = [
            Counter({1: 2}),
            Counter({0: 19, 1: 3, 2: 2, 3: 42}),
            Counter({0: 77, 1: 30, 2: 14, 3: 97}),
            Counter({0: 9, 1: 19, 3: 44}),
            Counter({0: 2, 1: 13}),
            Counter({0: 44, 1: 78, 2: 31, 3: 82}),
            Counter({0: 5, 1: 8, 2: 3, 3: 8}),
            Counter({0: 46, 1: 4, 3: 68}),
        ]
        scene_of = []
        image_counts = []
        for scene, size in enumerate(sizes):
            scene_of.extend([scene] * size)
            image_counts.append(per_scene[scene])
            image_counts.extend(Counter() for _ in range(size - 1))
        mapping, _ = assign_splits(scene_of, class_counts=image_counts)
        covered = {split: set() for split in ("train", "val", "test")}
        for scene, counts in enumerate(per_scene):
            covered[mapping[scene]].update(counts)
        for split in covered:
            self.assertEqual(covered[split], {0, 1, 2, 3})

    def test_review_subset_seeds_current_boxes_and_merges_only_when_complete(self):
        manifest = [
            {"id": "a", "boxes": [{"cls": 0}], "sort_key": 0.5},
            {"id": "b", "boxes": [], "sort_key": 0.2},
        ]
        progress = {
            "a": {
                "status": "reviewed",
                "boxes": [
                    {"cls": 3, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2},
                    {"cls": 3, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1},
                ],
            },
            "b": {"status": "reviewed", "boxes": []},
        }
        subset = build_subset(
            manifest, progress, explicit_ids=set(), ios_threshold=0.8)
        self.assertEqual([item["id"] for item in subset], ["a"])
        self.assertEqual(subset[0]["boxes"], progress["a"]["boxes"])
        by_class = build_subset(
            manifest, progress, explicit_ids=set(), class_ids={3})
        self.assertEqual([item["id"] for item in by_class], ["a"])
        by_prefix = build_subset(
            manifest, progress, explicit_ids=set(), id_prefixes=("b",))
        self.assertEqual([item["id"] for item in by_prefix], ["b"])
        partial = {"a": progress["a"]}
        with self.assertRaises(ValueError):
            build_subset(
                manifest, partial, explicit_ids=set(), id_prefixes=("b",))
        seeded_pending = build_subset(
            manifest,
            partial,
            explicit_ids=set(),
            id_prefixes=("b",),
            seed_unreviewed_from_manifest=True,
        )
        self.assertEqual([item["id"] for item in seeded_pending], ["b"])
        self.assertEqual(seeded_pending[0]["boxes"], manifest[1]["boxes"])
        self.assertIn("使用 manifest", seeded_pending[0]["review_note"])
        with self.assertRaises(ValueError):
            merge_subset(progress, subset, {})
        correction = {
            "a": {
                "status": "reviewed",
                "boxes": [progress["a"]["boxes"][0]],
            }
        }
        merged = merge_subset(progress, subset, correction)
        self.assertEqual(len(merged["a"]["boxes"]), 1)
        self.assertTrue(merged["a"]["changed"])
        self.assertEqual(merged["b"], progress["b"])
        no_op = {
            "a": {
                "status": "reviewed",
                "boxes": list(reversed(progress["a"]["boxes"])),
            }
        }
        merged_no_op = merge_subset(progress, subset, no_op)
        self.assertFalse(merged_no_op["a"]["changed"])

    def test_review_subset_applies_guarded_suggestions(self):
        subset = [
            {
                "id": "scene__00001",
                "boxes": [
                    {"cls": 2, "cx": 0.1, "cy": 0.2, "w": 0.3, "h": 0.4},
                    {"cls": 1, "cx": 0.5, "cy": 0.6, "w": 0.1, "h": 0.2},
                ],
                "review_note": "待复核",
            },
            {
                "id": "other__00001",
                "boxes": [
                    {"cls": 0, "cx": 0.7, "cy": 0.8, "w": 0.1, "h": 0.1},
                ],
            },
        ]
        suggestions = {
            "prefix_reclass": [
                {
                    "prefix": "scene__",
                    "from": [2],
                    "to": 3,
                    "expected_changes": 1,
                    "note": "暗灯头按 vehicle 预改",
                    "vehicle_only": True,
                }
            ],
            "box_delete": [
                {
                    "id": "scene__00001",
                    "index": 1,
                    "from": 1,
                    "cx": 0.5,
                    "cy": 0.6,
                    "w": 0.1,
                    "h": 0.2,
                    "note": "数字倒计时，预删",
                }
            ],
            "box_reclass": [
                {
                    "id": "other__00001",
                    "index": 0,
                    "from": 0,
                    "to": 3,
                    "cx": 0.7,
                    "cy": 0.8,
                    "w": 0.1,
                    "h": 0.1,
                    "note": "车行圆灯，预改",
                }
            ],
            "box_add": [
                {
                    "id": "scene__00001",
                    "cls": 0,
                    "cx": 0.8,
                    "cy": 0.1,
                    "w": 0.1,
                    "h": 0.1,
                    "conf": 0.9,
                    "expected_n_boxes": 2,
                    "note": "补回完整人行红灯",
                }
            ],
            "requires_change": ["scene__00001"],
        }
        result = apply_suggestions(subset, suggestions)
        self.assertEqual([box["cls"] for box in result[0]["boxes"]], [3, 0])
        self.assertEqual(result[0]["n_boxes"], 2)
        self.assertEqual(result[0]["boxes"][1]["conf"], 0.9)
        self.assertTrue(result[0]["vehicle_only"])
        self.assertIn("暗灯头", result[0]["review_note"])
        self.assertIn("倒计时", result[0]["review_note"])
        self.assertIn("补回完整人行红灯", result[0]["review_note"])
        self.assertEqual(result[1]["boxes"][0]["cls"], 3)
        self.assertTrue(result[0]["requires_change"])
        self.assertEqual(subset[0]["boxes"][0]["cls"], 2)

        unlocked = apply_suggestions(
            result,
            {
                "prefix_reclass": [
                    {
                        "prefix": "scene__",
                        "from": [2],
                        "to": 3,
                        "expected_changes": 0,
                        "vehicle_only": False,
                        "note": "证据不足，撤销硬锁后重新逐框判断",
                    }
                ]
            },
        )
        self.assertNotIn("vehicle_only", unlocked[0])
        self.assertIn("撤销硬锁", unlocked[0]["review_note"])

        bad = dict(suggestions)
        bad["box_delete"] = [dict(suggestions["box_delete"][0], cx=0.51)]
        with self.assertRaises(ValueError):
            apply_suggestions(subset, bad)
        unguarded = dict(suggestions)
        unguarded["box_delete"] = [
            {"id": "scene__00001", "index": 1, "from": 1}
        ]
        with self.assertRaises(ValueError):
            apply_suggestions(subset, unguarded)
        nonfinite = dict(suggestions)
        nonfinite["box_delete"] = [
            dict(suggestions["box_delete"][0], cx=float("nan"))
        ]
        with self.assertRaises(ValueError):
            apply_suggestions(subset, nonfinite)
        bool_class = dict(suggestions)
        bool_class["box_delete"] = [
            dict(suggestions["box_delete"][0], **{"from": True})
        ]
        with self.assertRaises(ValueError):
            apply_suggestions(subset, bool_class)
        zero_change = dict(suggestions)
        zero_change["prefix_reclass"] = [
            dict(suggestions["prefix_reclass"][0])
        ]
        zero_change["prefix_reclass"][0]["from"] = [0]
        with self.assertRaises(ValueError):
            apply_suggestions(subset, zero_change)
        unknown_required = dict(suggestions)
        unknown_required["requires_change"] = ["missing"]
        with self.assertRaises(ValueError):
            apply_suggestions(subset, unknown_required)

        bad_add_count = dict(suggestions)
        bad_add_count["box_add"] = [
            dict(suggestions["box_add"][0], expected_n_boxes=1)
        ]
        with self.assertRaises(ValueError):
            apply_suggestions(subset, bad_add_count)
        bad_add_bounds = dict(suggestions)
        bad_add_bounds["box_add"] = [
            dict(suggestions["box_add"][0], cx=0.99, w=0.1)
        ]
        with self.assertRaises(ValueError):
            apply_suggestions(subset, bad_add_bounds)
        bad_add_type = dict(suggestions)
        bad_add_type["box_add"] = [
            dict(suggestions["box_add"][0], cls=True)
        ]
        with self.assertRaises(ValueError):
            apply_suggestions(subset, bad_add_type)
        duplicate_add = dict(suggestions)
        duplicate_add["box_add"] = [
            suggestions["box_add"][0],
            dict(suggestions["box_add"][0]),
        ]
        with self.assertRaises(ValueError):
            apply_suggestions(subset, duplicate_add)

    def test_review_subset_prepare_selects_box_add_id(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = root / "manifest.json"
            progress = root / "progress.json"
            suggestions = root / "suggestions.json"
            out_manifest = root / "subset.json"
            out_progress = root / "subset_progress.json"
            manifest.write_text(
                json.dumps([
                    {"id": "a", "boxes": [], "sort_key": 0.2},
                    {"id": "b", "boxes": [], "sort_key": 0.1},
                ]),
                encoding="utf-8",
            )
            progress.write_text(
                json.dumps({
                    "a": {"status": "reviewed", "boxes": []},
                    "b": {"status": "reviewed", "boxes": []},
                }),
                encoding="utf-8",
            )
            suggestions.write_text(
                json.dumps({
                    "box_add": [{
                        "id": "b",
                        "cls": 1,
                        "cx": 0.5,
                        "cy": 0.5,
                        "w": 0.2,
                        "h": 0.2,
                        "expected_n_boxes": 0,
                    }]
                }),
                encoding="utf-8",
            )
            with patch("sys.argv", [
                "review_subset.py",
                "prepare",
                "--manifest", str(manifest),
                "--progress", str(progress),
                "--suggestions", str(suggestions),
                "--out-manifest", str(out_manifest),
                "--out-progress", str(out_progress),
            ]):
                self.assertEqual(review_subset_main(), 0)
            prepared = json.loads(out_manifest.read_text(encoding="utf-8"))
            self.assertEqual([item["id"] for item in prepared], ["b"])
            self.assertEqual(prepared[0]["n_boxes"], 1)

    def test_review_subset_attest_is_hash_guarded_and_fail_closed(self):
        def digest(path: Path) -> str:
            return hashlib.sha256(path.read_bytes()).hexdigest().upper()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = root / "subset.json"
            source = root / "source.json"
            output = root / "attested.json"
            source_box = {
                "cls": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2,
            }
            changed_box = dict(source_box, cls=1)
            manifest.write_text(
                json.dumps([
                    {"id": "a", "boxes": [source_box]},
                    {"id": "b", "boxes": [changed_box]},
                ]),
                encoding="utf-8",
            )
            source.write_text(
                json.dumps({
                    "a": {"status": "reviewed", "boxes": [source_box]},
                    "b": {"status": "reviewed", "boxes": [source_box]},
                }),
                encoding="utf-8",
            )

            def run(out: Path, attestation: str = ATTESTATION,
                    expected_manifest: str | None = None) -> int:
                with patch("sys.argv", [
                    "review_subset.py",
                    "attest",
                    "--subset-manifest", str(manifest),
                    "--source-progress", str(source),
                    "--out-progress", str(out),
                    "--reviewer", "qa-agent",
                    "--expected-manifest-sha256",
                    expected_manifest or digest(manifest),
                    "--expected-source-sha256", digest(source),
                    "--attestation", attestation,
                ]):
                    return review_subset_main()

            self.assertEqual(run(output), 0)
            attested = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(attested["a"]["changed"])
            self.assertTrue(attested["b"]["changed"])
            self.assertEqual(attested["b"]["by"], "qa-agent")
            self.assertEqual(run(output), 2)

            wrong_phrase = root / "wrong_phrase.json"
            self.assertEqual(run(wrong_phrase, "looks-good"), 2)
            self.assertFalse(wrong_phrase.exists())
            wrong_hash = root / "wrong_hash.json"
            self.assertEqual(run(wrong_hash, expected_manifest="0" * 64), 2)
            self.assertFalse(wrong_hash.exists())

            manifest.write_text(
                json.dumps([
                    {"id": "a", "boxes": [source_box]},
                    {"id": "a", "boxes": [source_box]},
                ]),
                encoding="utf-8",
            )
            duplicate = root / "duplicate.json"
            self.assertEqual(run(duplicate), 2)
            self.assertFalse(duplicate.exists())

            manifest.write_text(
                json.dumps([{
                    "id": "a",
                    "boxes": [dict(source_box, cx=float("nan"))],
                }]),
                encoding="utf-8",
            )
            invalid_box = root / "invalid_box.json"
            self.assertEqual(run(invalid_box), 2)
            self.assertFalse(invalid_box.exists())


if __name__ == "__main__":
    unittest.main()
