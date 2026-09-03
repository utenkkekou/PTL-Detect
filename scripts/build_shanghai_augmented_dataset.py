#!/usr/bin/env python3
"""Build clean4 train-only augmentation from an approved, scene-isolated manifest.

The base dataset's train/val/test files are copied (or hard-linked) unchanged.
Every Shanghai item must be reviewed, its source SHA256 and dimensions must
match the manifest, carry an independent-QA attestation, and originate outside
official testing.  The exact approved manifest SHA binds the upstream testing-
scene isolation audit.  Every selected item is added only to ``train``; locked
base validation and test files remain directly comparable with earlier models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from ptl.constants import ATTESTATION_KEY as ATTESTATION
from ptl.constants import EXPECTED_NAMES

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
APPROVED_MANIFEST_SHA256 = (
    "A33B796018428471FB6A5B3F52159C1C2812EE440C39CA7CBEAE617B50C8887E"
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ordered_names(data_yaml: Path) -> list[str]:
    import yaml

    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = config.get("names", [])
    if isinstance(names, dict):
        names = [names[key] for key in sorted(names, key=lambda value: int(value))]
    return list(names)


def _validate_box(item_id: str, index: int, box: dict[str, Any]) -> tuple[int, float, float, float, float]:
    try:
        raw_cls = box["cls"]
        if isinstance(raw_cls, bool) or not isinstance(raw_cls, int):
            raise TypeError("class id must be an integer, not a coercible value")
        cls = raw_cls
        cx, cy, width, height = (
            float(box["cx"]),
            float(box["cy"]),
            float(box["w"]),
            float(box["h"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{item_id}: invalid box #{index}: {box!r}") from exc

    if cls not in range(len(EXPECTED_NAMES)):
        raise ValueError(f"{item_id}: class {cls} is outside 0..3")
    values = (cx, cy, width, height)
    if not all(value == value and abs(value) != float("inf") for value in values):
        raise ValueError(f"{item_id}: non-finite box #{index}: {box!r}")
    epsilon = 1e-6
    if (
        width <= 0
        or height <= 0
        or cx - width / 2 < -epsilon
        or cy - height / 2 < -epsilon
        or cx + width / 2 > 1 + epsilon
        or cy + height / 2 > 1 + epsilon
    ):
        raise ValueError(f"{item_id}: out-of-bounds box #{index}: {box!r}")
    return cls, cx, cy, width, height


def _copy_or_link(source: Path, destination: Path, hardlink: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if hardlink:
        os.link(source, destination)
    else:
        shutil.copy2(source, destination)


def _find_review_sources(
    manifest: list[dict[str, Any]], raw_images: Path
) -> dict[str, Path]:
    wanted_names = {str(item["file"]).casefold() for item in manifest}
    candidates: dict[str, list[Path]] = defaultdict(list)
    for path in raw_images.rglob("*"):
        if path.is_file() and path.name.casefold() in wanted_names:
            candidates[path.name.casefold()].append(path)

    resolved: dict[str, Path] = {}
    for item in manifest:
        item_id = str(item["id"])
        expected_hash = str(item["source_sha256"]).lower()
        matches = candidates.get(str(item["file"]).casefold(), [])
        if not matches:
            raise FileNotFoundError(f"{item_id}: source image not found: {item['file']}")
        hash_matches = [path for path in matches if sha256_file(path) == expected_hash]
        if len(hash_matches) != 1:
            raise ValueError(
                f"{item_id}: expected exactly one source with SHA256 "
                f"{expected_hash}, found {len(hash_matches)}"
            )
        source = hash_matches[0]
        with Image.open(source) as image:
            actual_size = image.size
        expected_size = (int(item["width"]), int(item["height"]))
        if actual_size != expected_size:
            raise ValueError(
                f"{item_id}: dimensions {actual_size} do not match {expected_size}"
            )
        resolved[item_id] = source
    return resolved


def _copy_base_dataset(
    base: Path, stage: Path, hardlink: bool
) -> tuple[dict[str, int], set[str], set[str]]:
    split_counts: dict[str, int] = {}
    hashes_by_split: dict[str, set[str]] = {}
    all_hashes: set[str] = set()

    for split in ("train", "val", "test"):
        image_dir = base / "images" / split
        label_dir = base / "labels" / split
        images = sorted(
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            raise ValueError(f"base split has no images: {image_dir}")
        stems: set[str] = set()
        split_hashes: set[str] = set()
        for source_image in images:
            normalized_stem = source_image.stem.casefold()
            if normalized_stem in stems:
                raise ValueError(f"duplicate image stem in {split}: {source_image.stem}")
            stems.add(normalized_stem)
            source_label = label_dir / f"{source_image.stem}.txt"
            if not source_label.is_file():
                raise FileNotFoundError(f"missing base label: {source_label}")
            image_hash = sha256_file(source_image)
            if image_hash in all_hashes:
                raise ValueError(
                    f"exact image duplicate within/across base splits: {source_image}"
                )
            all_hashes.add(image_hash)
            split_hashes.add(image_hash)
            _copy_or_link(
                source_image,
                stage / "images" / split / source_image.name,
                hardlink,
            )
            _copy_or_link(
                source_label,
                stage / "labels" / split / source_label.name,
                hardlink,
            )
        split_counts[split] = len(images)
        hashes_by_split[split] = split_hashes

    if hashes_by_split["val"] & hashes_by_split["test"]:
        raise AssertionError("base val/test exact-hash overlap")
    base_train_stems = {
        path.stem.casefold()
        for path in (stage / "images" / "train").iterdir()
        if path.is_file()
    }
    return split_counts, all_hashes, base_train_stems


def build_dataset(
    *,
    base: Path,
    manifest_path: Path,
    progress_path: Path,
    raw_images: Path,
    output: Path,
    expected_manifest_sha256: str,
    hardlink: bool = False,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"output already exists; refusing to overwrite: {output}")
    data_yaml = base / "data.yaml"
    if _ordered_names(data_yaml) != EXPECTED_NAMES:
        raise ValueError(f"base data.yaml does not use the clean4 class contract: {data_yaml}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    expected_manifest_sha256 = expected_manifest_sha256.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256):
        raise ValueError("expected_manifest_sha256 must be exactly 64 hex characters")
    actual_manifest_sha256 = sha256_file(manifest_path)
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise ValueError(
            "manifest SHA256 does not match the independently approved selection: "
            f"{actual_manifest_sha256} != {expected_manifest_sha256}"
        )
    if not isinstance(manifest, list) or not isinstance(progress, dict):
        raise ValueError("manifest must be a list and progress must be an object")
    ids = [str(item["id"]) for item in manifest]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate IDs in review manifest")
    if len({item_id.casefold() for item_id in ids}) != len(ids):
        raise ValueError("case-insensitive duplicate IDs in review manifest")
    for item in manifest:
        item_id = str(item["id"])
        filename = str(item.get("file", ""))
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", item_id):
            raise ValueError(f"unsafe manifest id: {item_id!r}")
        if not filename or Path(filename).name != filename:
            raise ValueError(f"{item_id}: unsafe source filename: {filename!r}")
        source_split = str(item.get("source_split", "")).casefold()
        if source_split not in {"training", "validation"}:
            raise ValueError(
                f"{item_id}: source_split must be training/validation, got "
                f"{item.get('source_split')!r}"
            )
    missing = [
        item_id
        for item_id in ids
        if progress.get(item_id, {}).get("status") != "reviewed"
        or "boxes" not in progress.get(item_id, {})
    ]
    extras = sorted(set(progress) - set(ids))
    if missing or extras:
        raise ValueError(
            f"review is not an exact completed set: missing={len(missing)}, extras={len(extras)}"
        )
    unattested = [
        item_id
        for item_id in ids
        if progress[item_id].get("attestation") != ATTESTATION
        or not isinstance(progress[item_id].get("by"), str)
        or not progress[item_id]["by"].strip()
    ]
    if unattested:
        raise ValueError(
            "review entries lack independent-QA attestation/reviewer: "
            f"{unattested[:10]}"
        )

    reviewed_boxes: dict[str, list[tuple[int, float, float, float, float]]] = {}
    class_counts: Counter[int] = Counter()
    empty_images = 0
    for item in manifest:
        item_id = str(item["id"])
        boxes = [
            _validate_box(item_id, index, box)
            for index, box in enumerate(progress[item_id]["boxes"])
        ]
        if len(boxes) != len(set(boxes)):
            raise ValueError(f"{item_id}: exact duplicate reviewed boxes")
        if not boxes:
            empty_images += 1
        class_counts.update(box[0] for box in boxes)
        reviewed_boxes[item_id] = boxes

    sources = _find_review_sources(manifest, raw_images)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f"{output.name}.tmp-", dir=output.parent))
    try:
        base_counts, base_hashes, used_train_stems = _copy_base_dataset(
            base, stage, hardlink
        )
        shanghai_hashes: set[str] = set()
        for item in manifest:
            item_id = str(item["id"])
            source = sources[item_id]
            source_hash = str(item["source_sha256"]).lower()
            if source_hash in shanghai_hashes:
                raise ValueError(f"duplicate Shanghai source SHA256: {item_id}")
            if source_hash in base_hashes:
                raise ValueError(f"Shanghai image duplicates the locked base dataset: {item_id}")
            shanghai_hashes.add(source_hash)

            destination_stem = f"imv__{item_id}"
            normalized_destination_stem = destination_stem.casefold()
            if normalized_destination_stem in used_train_stems:
                raise ValueError(
                    f"Shanghai destination stem collides with train data: "
                    f"{destination_stem}"
                )
            used_train_stems.add(normalized_destination_stem)
            destination_image = (
                stage / "images" / "train" / f"{destination_stem}{source.suffix.lower()}"
            )
            _copy_or_link(source, destination_image, hardlink)
            lines = [
                f"{cls} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}"
                for cls, cx, cy, width, height in reviewed_boxes[item_id]
            ]
            destination_label = stage / "labels" / "train" / f"{destination_stem}.txt"
            destination_label.parent.mkdir(parents=True, exist_ok=True)
            destination_label.write_text(
                "\n".join(lines) + ("\n" if lines else ""),
                encoding="utf-8",
            )

        final_counts = dict(base_counts)
        final_counts["train"] += len(manifest)
        yaml_lines = [
            "# Generated by scripts/build_shanghai_augmented_dataset.py",
            f"path: {output.resolve().as_posix()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "nc: 4",
            "names:",
            *[f"  {index}: {name}" for index, name in enumerate(EXPECTED_NAMES)],
            "",
        ]
        (stage / "data.yaml").write_text("\n".join(yaml_lines), encoding="utf-8")
        report: dict[str, Any] = {
            "schema_version": 1,
            "base_dataset": str(base.resolve()),
            "base_counts": base_counts,
            "final_counts": final_counts,
            "shanghai_split": "train",
            "shanghai_images": len(manifest),
            "shanghai_scenes": len({str(item["scene_id"]) for item in manifest}),
            "shanghai_empty_images": empty_images,
            "shanghai_boxes": sum(class_counts.values()),
            "shanghai_class_counts": {
                EXPECTED_NAMES[index]: class_counts[index]
                for index in range(len(EXPECTED_NAMES))
            },
            "manifest_sha256": actual_manifest_sha256,
            "approved_manifest_sha256": expected_manifest_sha256,
            "progress_sha256": sha256_file(progress_path),
            "review_attestation": ATTESTATION,
            "source_split_counts": dict(
                sorted(Counter(str(item["source_split"]) for item in manifest).items())
            ),
            "exact_sha_overlap_with_base": 0,
            "base_val_test_modified": False,
            "hardlinked": hardlink,
        }
        (stage / "augmentation_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        stage.replace(output)
        return report
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="data/yolo_clean4_video_v2")
    parser.add_argument(
        "--manifest",
        default="data/interim/imvisible_clean4_review_v1/review_manifest_clean4.json",
    )
    parser.add_argument(
        "--progress",
        default=(
            "data/interim/imvisible_clean4_review_v1/"
            "review_progress_clean4_final.json"
        ),
    )
    parser.add_argument(
        "--expected-manifest-sha256",
        default=APPROVED_MANIFEST_SHA256,
        help="binds the independently audited testing-scene-isolated manifest",
    )
    parser.add_argument("--raw-images", default="data/raw/imvisible/images")
    parser.add_argument("--out", default="data/yolo_clean4_video_shanghai_v1")
    parser.add_argument(
        "--hardlink",
        action="store_true",
        help="hard-link immutable images/base labels instead of copying them",
    )
    args = parser.parse_args()
    report = build_dataset(
        base=Path(args.base),
        manifest_path=Path(args.manifest),
        progress_path=Path(args.progress),
        raw_images=Path(args.raw_images),
        output=Path(args.out),
        expected_manifest_sha256=args.expected_manifest_sha256,
        hardlink=args.hardlink,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
