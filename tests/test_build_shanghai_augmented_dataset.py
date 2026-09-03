import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.build_shanghai_augmented_dataset import build_dataset


NAMES = [
    "pedestrian_signal_red",
    "pedestrian_signal_green",
    "pedestrian_signal_unknown",
    "vehicle_signal",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class BuildShanghaiAugmentedDatasetTests(unittest.TestCase):
    def _fixture(self, root: Path):
        base = root / "base"
        for split in ("train", "val", "test"):
            image_dir = base / "images" / split
            label_dir = base / "labels" / split
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            Image.new("RGB", (12, 10), (20 + len(split), 30, 40)).save(
                image_dir / f"{split}.png"
            )
            (label_dir / f"{split}.txt").write_text("", encoding="utf-8")
        (base / "data.yaml").write_text(
            "path: fixture\n"
            "train: images/train\nval: images/val\ntest: images/test\n"
            "nc: 4\nnames:\n"
            + "".join(f"  {index}: {name}\n" for index, name in enumerate(NAMES)),
            encoding="utf-8",
        )

        raw = root / "raw" / "part1"
        raw.mkdir(parents=True)
        source = raw / "sam_IMG_0001.JPG"
        Image.new("RGB", (20, 16), (200, 10, 10)).save(source)
        manifest = [
            {
                "id": "sam_IMG_0001",
                "file": source.name,
                "width": 20,
                "height": 16,
                "scene_id": "scene_1",
                "source_split": "training",
                "source_sha256": _sha256(source),
            }
        ]
        progress = {
            "sam_IMG_0001": {
                "status": "reviewed",
                "boxes": [{"cls": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.25}],
                "by": "fixture-reviewer",
                "attestation": "independent-visual-qa-complete",
            }
        }
        manifest_path = root / "manifest.json"
        progress_path = root / "progress.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        progress_path.write_text(json.dumps(progress), encoding="utf-8")
        return base, raw.parent, manifest_path, progress_path

    def test_adds_reviewed_images_to_train_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base, raw, manifest, progress = self._fixture(root)
            output = root / "out"
            original_val_hash = _sha256(base / "images" / "val" / "val.png")
            report = build_dataset(
                base=base,
                manifest_path=manifest,
                progress_path=progress,
                raw_images=raw,
                output=output,
                expected_manifest_sha256=_sha256(manifest),
            )
            self.assertEqual(report["base_counts"], {"train": 1, "val": 1, "test": 1})
            self.assertEqual(report["final_counts"], {"train": 2, "val": 1, "test": 1})
            self.assertEqual(report["shanghai_class_counts"][NAMES[0]], 1)
            self.assertTrue((output / "images" / "train" / "imv__sam_IMG_0001.jpg").is_file())
            self.assertTrue((output / "labels" / "train" / "imv__sam_IMG_0001.txt").is_file())
            self.assertFalse((output / "images" / "val" / "imv__sam_IMG_0001.jpg").exists())
            self.assertEqual(
                _sha256(output / "images" / "val" / "val.png"), original_val_hash
            )

    def test_rejects_unreviewed_item_without_creating_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base, raw, manifest, progress = self._fixture(root)
            saved = json.loads(progress.read_text(encoding="utf-8"))
            saved["sam_IMG_0001"]["status"] = "pending"
            progress.write_text(json.dumps(saved), encoding="utf-8")
            output = root / "out"
            with self.assertRaisesRegex(ValueError, "not an exact completed set"):
                build_dataset(
                    base=base,
                    manifest_path=manifest,
                    progress_path=progress,
                    raw_images=raw,
                    output=output,
                    expected_manifest_sha256=_sha256(manifest),
                )
            self.assertFalse(output.exists())

    def test_rejects_source_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base, raw, manifest, progress = self._fixture(root)
            saved = json.loads(manifest.read_text(encoding="utf-8"))
            saved[0]["source_sha256"] = "0" * 64
            manifest.write_text(json.dumps(saved), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected exactly one source"):
                build_dataset(
                    base=base,
                    manifest_path=manifest,
                    progress_path=progress,
                    raw_images=raw,
                    output=root / "out",
                    expected_manifest_sha256=_sha256(manifest),
                )

    def test_rejects_testing_source_and_unattested_progress(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base, raw, manifest, progress = self._fixture(root)
            saved_manifest = json.loads(manifest.read_text(encoding="utf-8"))
            saved_manifest[0]["source_split"] = "testing"
            manifest.write_text(json.dumps(saved_manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source_split"):
                build_dataset(
                    base=base,
                    manifest_path=manifest,
                    progress_path=progress,
                    raw_images=raw,
                    output=root / "testing-out",
                    expected_manifest_sha256=_sha256(manifest),
                )

            saved_manifest[0]["source_split"] = "training"
            manifest.write_text(json.dumps(saved_manifest), encoding="utf-8")
            saved_progress = json.loads(progress.read_text(encoding="utf-8"))
            del saved_progress["sam_IMG_0001"]["attestation"]
            progress.write_text(json.dumps(saved_progress), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "attestation"):
                build_dataset(
                    base=base,
                    manifest_path=manifest,
                    progress_path=progress,
                    raw_images=raw,
                    output=root / "unattested-out",
                    expected_manifest_sha256=_sha256(manifest),
                )

            saved_progress["sam_IMG_0001"]["attestation"] = (
                "independent-visual-qa-complete"
            )
            saved_progress["sam_IMG_0001"]["by"] = {}
            progress.write_text(json.dumps(saved_progress), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "attestation"):
                build_dataset(
                    base=base,
                    manifest_path=manifest,
                    progress_path=progress,
                    raw_images=raw,
                    output=root / "nonstring-reviewer-out",
                    expected_manifest_sha256=_sha256(manifest),
                )

    def test_rejects_fractional_class_and_manifest_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base, raw, manifest, progress = self._fixture(root)
            with self.assertRaisesRegex(ValueError, "approved selection"):
                build_dataset(
                    base=base,
                    manifest_path=manifest,
                    progress_path=progress,
                    raw_images=raw,
                    output=root / "hash-out",
                    expected_manifest_sha256="0" * 64,
                )

            saved = json.loads(progress.read_text(encoding="utf-8"))
            saved["sam_IMG_0001"]["boxes"][0]["cls"] = 0.9
            progress.write_text(json.dumps(saved), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid box"):
                build_dataset(
                    base=base,
                    manifest_path=manifest,
                    progress_path=progress,
                    raw_images=raw,
                    output=root / "class-out",
                    expected_manifest_sha256=_sha256(manifest),
                )


if __name__ == "__main__":
    unittest.main()
