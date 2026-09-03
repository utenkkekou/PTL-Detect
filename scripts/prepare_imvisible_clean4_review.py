#!/usr/bin/env python3
"""Prepare a leak-safe ImVisible batch for a fresh four-class manual review.

The official ImVisible annotations are image-level hints, not bounding boxes.
The official testing split is sealed: all 5,059 official images are grouped
into physical/temporal scenes first, then every scene containing a testing
image is excluded in full.

Legacy three-class boxes and optional predictions from the current four-class
model are only *untrusted review seeds*.  Nothing from this script is training
ground truth until the new, initially empty progress file is manually reviewed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from PIL import ExifTags, Image, ImageFile

REPO = Path(__file__).resolve().parents[1]
EXPECTED_NAMES = [
    "pedestrian_signal_red",
    "pedestrian_signal_green",
    "pedestrian_signal_unknown",
    "vehicle_signal",
]
MODE_TARGET = {0: 0, 1: 1, 2: 1, 3: -1, 4: -1}
MODE_NAME = {
    0: "Red",
    1: "Green",
    2: "Countdown Green",
    3: "Countdown Blank",
    4: "None",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
REVIEW_NOTE = (
    "所有框均为不可信候选，必须逐灯头目视复核；车行灯/自行车灯一律标为 "
    "class 3（红绿颜色不改变 class 3）；倒计时数字不是信号灯框，必须删除；"
    "每个独立灯头分别复核、补漏、删错框并修正人行红/绿/未知（class 0/1/2）。"
    "官方 mode 只作整图提示/target，不是框级真值；Countdown Blank（mode 3）"
    "无法安全映射为 unknown，绝不能据此赋 class 2。"
)

ImageFile.LOAD_TRUNCATED_IMAGES = True


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contributor_of(filename: str) -> str:
    match = re.match(r"^([^_]+)_", Path(filename).stem)
    return match.group(1).casefold() if match else "anonymous"


def _degrees(value: Iterable[Any]) -> float:
    parts = [float(part) for part in value]
    if len(parts) != 3:
        raise ValueError(f"invalid GPS coordinate: {value!r}")
    return parts[0] + parts[1] / 60 + parts[2] / 3600


def read_image_metadata(path: Path) -> dict[str, Any]:
    try:
        with Image.open(path) as image:
            width, height = image.size
            exif = image.getexif()
            exif_ifd = exif.get_ifd(ExifTags.IFD.Exif)
            raw_time = (
                exif_ifd.get(36867)
                or exif_ifd.get(36868)
                or exif.get(306)
            )
            captured_at = None
            if raw_time:
                captured_at = datetime.strptime(
                    str(raw_time), "%Y:%m:%d %H:%M:%S"
                )
            gps = None
            gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
            if gps_ifd.get(2) and gps_ifd.get(4):
                lat = _degrees(gps_ifd[2])
                lon = _degrees(gps_ifd[4])
                if str(gps_ifd.get(1, "N")).upper() == "S":
                    lat = -lat
                if str(gps_ifd.get(3, "E")).upper() == "W":
                    lon = -lon
                gps = (lat, lon)
            return {
                "width": width,
                "height": height,
                "captured_at": captured_at,
                "gps": gps,
            }
    except Exception as exc:
        raise ValueError(f"cannot read image metadata: {path}: {exc}") from exc


def haversine_meters(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    value = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 6371000.0 * 2 * math.asin(math.sqrt(value))


def assign_scene_ids(
    records: list[dict[str, Any]],
    gps_radius_m: float = 40.0,
    no_gps_minutes: float = 10.0,
) -> list[dict[str, Any]]:
    """Return copied records with deterministic transitive scene IDs."""
    if gps_radius_m <= 0 or no_gps_minutes <= 0:
        raise ValueError("scene thresholds must be positive")
    copied = [dict(record) for record in records]
    union = UnionFind(len(copied))

    by_sha: dict[str, int] = {}
    for index, record in enumerate(copied):
        sha = str(record.get("sha256", "")).lower()
        if not sha:
            raise ValueError(f"missing SHA256: {record.get('file')}")
        previous = by_sha.get(sha)
        if previous is not None:
            if int(copied[previous]["mode"]) != int(record["mode"]):
                raise ValueError(
                    "same SHA has conflicting official modes: "
                    f"{copied[previous]['file']} / {record['file']}"
                )
            union.union(previous, index)
        else:
            by_sha[sha] = index

    gps_rows = sorted(
        (
            (index, record["gps"])
            for index, record in enumerate(copied)
            if record.get("gps") is not None
        ),
        key=lambda pair: pair[1][0],
    )
    max_latitude_delta = gps_radius_m / 110500.0
    for position, (left_index, left_gps) in enumerate(gps_rows):
        for right_index, right_gps in gps_rows[position + 1 :]:
            if right_gps[0] - left_gps[0] > max_latitude_delta:
                break
            if haversine_meters(left_gps, right_gps) <= gps_radius_m:
                union.union(left_index, right_index)

    by_contributor: dict[str, list[tuple[datetime, int]]] = defaultdict(list)
    for index, record in enumerate(copied):
        if record.get("captured_at") is not None:
            by_contributor[str(record["contributor"])].append(
                (record["captured_at"], index)
            )
    max_seconds = no_gps_minutes * 60
    for entries in by_contributor.values():
        entries.sort()
        for (left_time, left), (right_time, right) in zip(entries, entries[1:]):
            # GPS-GPS pairs are linked only by physical proximity above.  A
            # missing-GPS frame may, however, inherit the same scene from its
            # close temporal neighbor (including one that has GPS).
            if (
                copied[left].get("gps") is None
                or copied[right].get("gps") is None
            ) and (right_time - left_time).total_seconds() <= max_seconds:
                union.union(left, right)

    members: dict[int, list[int]] = defaultdict(list)
    for index in range(len(copied)):
        members[union.find(index)].append(index)
    ordered_roots = sorted(
        members,
        key=lambda root: min(
            str(copied[index]["file"]).casefold() for index in members[root]
        ),
    )
    scene_name = {
        root: f"imv_scene_{number:04d}"
        for number, root in enumerate(ordered_roots, start=1)
    }
    for index, record in enumerate(copied):
        record["scene_id"] = scene_name[union.find(index)]
    return copied


def exclude_testing_scenes(
    records: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = [dict(record) for record in records]
    testing_scenes = {
        record["scene_id"]
        for record in records
        if record.get("split") == "testing"
    }
    eligible = [
        record for record in records if record["scene_id"] not in testing_scenes
    ]
    if any(record.get("split") == "testing" for record in eligible):
        raise AssertionError("official testing leakage after scene exclusion")
    excluded = [
        record for record in records if record["scene_id"] in testing_scenes
    ]
    return eligible, {
        "testing_scene_count": len(testing_scenes),
        "excluded_image_count": len(excluded),
        "excluded_non_testing_image_count": sum(
            record.get("split") != "testing" for record in excluded
        ),
    }


def select_with_none_quota(
    eligible: Iterable[dict[str, Any]], none_limit: int = 20
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if none_limit < 0:
        raise ValueError("none_limit must be non-negative")
    eligible = [dict(record) for record in eligible]
    positives = [record for record in eligible if int(record["mode"]) != 4]
    none_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in eligible:
        if int(record["mode"]) == 4:
            none_by_scene[record["scene_id"]].append(record)
    for values in none_by_scene.values():
        values.sort(key=lambda record: str(record["file"]).casefold())

    chosen_none: list[dict[str, Any]] = []
    scene_names = sorted(none_by_scene)
    depth = 0
    while len(chosen_none) < none_limit:
        added = False
        for scene in scene_names:
            values = none_by_scene[scene]
            if depth < len(values):
                chosen_none.append(values[depth])
                added = True
                if len(chosen_none) == none_limit:
                    break
        if not added:
            break
        depth += 1

    selected = positives + chosen_none
    selected.sort(
        key=lambda record: (
            int(record["mode"]) == 4,
            int(record["mode"]),
            record["scene_id"],
            str(record["file"]).casefold(),
        )
    )
    total_none = sum(len(values) for values in none_by_scene.values())
    return selected, {
        "eligible_none": total_none,
        "selected_none": len(chosen_none),
        "quota_excluded_none": total_none - len(chosen_none),
        "selected_none_scenes": len(
            {record["scene_id"] for record in chosen_none}
        ),
    }


def box_iou(left: dict[str, Any], right: dict[str, Any]) -> float:
    def corners(box: dict[str, Any]) -> tuple[float, float, float, float]:
        return (
            float(box["cx"]) - float(box["w"]) / 2,
            float(box["cy"]) - float(box["h"]) / 2,
            float(box["cx"]) + float(box["w"]) / 2,
            float(box["cy"]) + float(box["h"]) / 2,
        )

    ax1, ay1, ax2, ay2 = corners(left)
    bx1, by1, bx2, by2 = corners(right)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1)
    return intersection / (union - intersection) if union > intersection else 0.0


def _validated_candidate(box: dict[str, Any], source: str) -> dict[str, Any]:
    required = {"cls", "cx", "cy", "w", "h"}
    if not isinstance(box, dict) or not required <= set(box):
        raise ValueError(f"malformed legacy box from {source}: {box!r}")
    cls = box["cls"]
    if not isinstance(cls, int) or isinstance(cls, bool) or cls not in range(4):
        raise ValueError(f"invalid class from {source}: {cls!r}")
    values = {key: float(box[key]) for key in ("cx", "cy", "w", "h")}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError(f"non-finite box from {source}")
    x1 = values["cx"] - values["w"] / 2
    y1 = values["cy"] - values["h"] / 2
    x2 = values["cx"] + values["w"] / 2
    y2 = values["cy"] + values["h"] / 2
    # Historical six-decimal exports can exceed an edge by half a rounding
    # unit. Canonicalize that tiny error, while still failing closed on real
    # invalid geometry.
    tolerance = 1e-5
    if (
        values["w"] <= 0
        or values["h"] <= 0
        or x1 < -tolerance
        or x2 > 1 + tolerance
        or y1 < -tolerance
        or y2 > 1 + tolerance
    ):
        raise ValueError(f"out-of-image box from {source}: {box!r}")
    x1, y1, x2, y2 = (
        max(0.0, x1),
        max(0.0, y1),
        min(1.0, x2),
        min(1.0, y2),
    )
    values = {
        "cx": (x1 + x2) / 2,
        "cy": (y1 + y2) / 2,
        "w": x2 - x1,
        "h": y2 - y1,
    }
    result = {"cls": cls, **{key: round(value, 6) for key, value in values.items()}}
    if "conf" in box:
        confidence = float(box["conf"])
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError(f"invalid confidence from {source}")
        result["conf"] = round(confidence, 4)
    result["candidate_sources"] = [source]
    result["untrusted_seed"] = True
    return result


def merge_untrusted_candidates(
    candidates: Iterable[tuple[str, Iterable[dict[str, Any]]]],
    iou_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    if not 0 < iou_threshold <= 1:
        raise ValueError("IoU threshold must be in (0, 1]")
    merged: list[dict[str, Any]] = []
    for source, boxes in candidates:
        for raw_box in boxes:
            box = _validated_candidate(raw_box, source)
            duplicate = next(
                (
                    kept
                    for kept in merged
                    if kept["cls"] == box["cls"]
                    and box_iou(kept, box) >= iou_threshold
                ),
                None,
            )
            if duplicate is None:
                merged.append(box)
            elif source not in duplicate["candidate_sources"]:
                duplicate["candidate_sources"].append(source)
    return merged


def has_cross_class_overlap(
    boxes: Iterable[dict[str, Any]], iou_threshold: float = 0.5
) -> bool:
    boxes = list(boxes)
    return any(
        left["cls"] != right["cls"] and box_iou(left, right) >= iou_threshold
        for index, left in enumerate(boxes)
        for right in boxes[index + 1 :]
    )


def ensure_fresh_output(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite output: {path}")


def _image_index(images_dir: Path) -> dict[str, Path]:
    if not images_dir.is_dir():
        raise ValueError(f"images directory does not exist: {images_dir}")
    index: dict[str, Path] = {}
    for path in images_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        key = path.name.casefold()
        if key in index:
            raise ValueError(
                f"ambiguous image basename: {index[key]} / {path}"
            )
        index[key] = path
    return index


def load_official_records(
    annotations_dir: Path,
    images_dir: Path,
    expected_total: int = 5059,
    expected_testing: int = 739,
) -> list[dict[str, Any]]:
    index = _image_index(images_dir)
    records: list[dict[str, Any]] = []
    seen_files: set[str] = set()
    for split in ("training", "validation", "testing"):
        csv_path = annotations_dir / f"{split}_file.csv"
        if not csv_path.is_file():
            raise ValueError(f"missing official CSV: {csv_path}")
        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or not {"file", "mode"} <= set(
                reader.fieldnames
            ):
                raise ValueError(f"official CSV schema mismatch: {csv_path}")
            for row in reader:
                filename = row["file"].strip()
                key = filename.casefold()
                if key in seen_files:
                    raise ValueError(f"duplicate official filename: {filename}")
                seen_files.add(key)
                try:
                    mode = int(row["mode"])
                except Exception as exc:
                    raise ValueError(
                        f"invalid official mode: {filename}"
                    ) from exc
                if mode not in MODE_TARGET:
                    raise ValueError(f"invalid official mode={mode}: {filename}")
                path = index.get(key)
                if path is None:
                    raise ValueError(f"official image is missing: {filename}")
                metadata = read_image_metadata(path)
                records.append(
                    {
                        "file": filename,
                        "id": path.stem,
                        "path": path,
                        "split": split,
                        "mode": mode,
                        "contributor": contributor_of(filename),
                        "sha256": sha256_file(path),
                        **metadata,
                    }
                )
    if len(records) != expected_total:
        raise ValueError(
            f"official row count mismatch: {len(records)} != {expected_total}"
        )
    testing_count = sum(record["split"] == "testing" for record in records)
    if testing_count != expected_testing:
        raise ValueError(
            f"official testing count mismatch: {testing_count} != {expected_testing}"
        )
    if len({record["id"].casefold() for record in records}) != len(records):
        raise ValueError("official image stems are not unique")
    return records


def load_legacy_progress(path: Path, source: str) -> dict[str, list[dict[str, Any]]]:
    if not path.is_file():
        raise ValueError(f"missing legacy progress: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"legacy progress must be an object: {path}")
    output: dict[str, list[dict[str, Any]]] = {}
    for image_id, entry in payload.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("boxes"), list):
            raise ValueError(f"malformed legacy progress entry: {source}:{image_id}")
        output[str(image_id)] = [
            _validated_candidate(box, source) for box in entry["boxes"]
        ]
    return output


def predict_class3(
    records: Iterable[dict[str, Any]],
    model_path: Path,
    confidence: float,
    image_size: int,
) -> dict[str, list[dict[str, Any]]]:
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    names = model.names
    normalized = (
        [names[key] for key in sorted(names, key=int)]
        if isinstance(names, dict)
        else list(names)
    )
    if normalized != EXPECTED_NAMES:
        raise ValueError(
            f"model is not the exact clean4 contract: {normalized!r}"
        )
    output: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        results = model.predict(
            str(record["path"]),
            classes=[3],
            conf=confidence,
            imgsz=image_size,
            verbose=False,
        )
        if not results:
            raise RuntimeError(f"model returned no result: {record['file']}")
        boxes: list[dict[str, Any]] = []
        for detected in results[0].boxes:
            x1, y1, x2, y2 = detected.xyxyn[0].tolist()
            boxes.append(
                {
                    "cls": 3,
                    "cx": (x1 + x2) / 2,
                    "cy": (y1 + y2) / 2,
                    "w": x2 - x1,
                    "h": y2 - y1,
                    "conf": float(detected.conf[0]),
                }
            )
        output[record["id"]] = boxes
    return output


def _render_report(
    all_records: list[dict[str, Any]],
    eligible: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    exclusion: dict[str, Any],
    quota: dict[str, int],
    manifest: list[dict[str, Any]],
    model_path: Path | None,
    model_confidence: float,
    image_size: int,
    dedup_iou: float,
) -> str:
    def counts(records: Iterable[dict[str, Any]]) -> dict[int, int]:
        counter = Counter(int(record["mode"]) for record in records)
        return {mode: counter[mode] for mode in range(5)}

    candidate_classes = Counter(
        box["cls"] for item in manifest for box in item["boxes"]
    )
    lines = [
        "# ImVisible clean4 fresh-review selection report",
        "",
        "Official testing is sealed. Scene grouping was completed across all "
        f"{len(all_records)} official rows before exclusion.",
        "",
        f"- All official modes: `{counts(all_records)}`",
        f"- Testing-containing scenes: `{exclusion['testing_scene_count']}`",
        f"- Images excluded with those scenes: "
        f"`{exclusion['excluded_image_count']}` "
        f"(including `{exclusion['excluded_non_testing_image_count']}` "
        "non-testing neighbors)",
        f"- Eligible after scene isolation: `{len(eligible)}`; "
        f"modes `{counts(eligible)}`",
        f"- Selected for fresh review: `{len(selected)}`; "
        f"modes `{counts(selected)}`",
        f"- Mode 4 quota: selected `{quota['selected_none']}` from "
        f"`{quota['eligible_none']}`; excluded by quota "
        f"`{quota['quota_excluded_none']}`; selected scenes "
        f"`{quota['selected_none_scenes']}`",
        f"- Untrusted candidate boxes by class: "
        f"`{dict(sorted(candidate_classes.items()))}`",
        f"- Optional clean4 model: `{model_path if model_path else 'not used'}`",
        (
            f"- Model SHA256: `{sha256_file(model_path)}`"
            if model_path
            else "- Model SHA256: `not applicable`"
        ),
        f"- Candidate settings: conf `{model_confidence}`, imgsz "
        f"`{image_size}`, same-class dedup IoU `{dedup_iou}`",
        "",
        "Scene rules: exact SHA256 linkage; EXIF GPS distance <= 40 m; "
        "for images without GPS, same-contributor adjacent EXIF time <= 10 min. "
        "All rules are transitive.",
        "",
        "Legacy boxes and model boxes are candidates only. Official mode is an "
        "image-level hint only. Countdown Blank (mode 3) has target=-1 and "
        "must not be mapped to class 2/unknown. The progress file is "
        "deliberately empty.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotations-dir",
        default=str(REPO / "data/raw/imvisible/Annotations"),
    )
    parser.add_argument(
        "--images-dir", default=str(REPO / "data/raw/imvisible/images")
    )
    parser.add_argument(
        "--seed-progress",
        default=str(REPO / "data/interim/ptl/seed_progress.json"),
    )
    parser.add_argument(
        "--reann-progress",
        default=str(REPO / "data/interim/ptl/reann_progress.json"),
    )
    parser.add_argument(
        "--out",
        default=str(REPO / "data/interim/imvisible_clean4_review_v1"),
    )
    parser.add_argument("--none-limit", type=int, default=20)
    parser.add_argument("--gps-radius-m", type=float, default=40.0)
    parser.add_argument("--no-gps-minutes", type=float, default=10.0)
    parser.add_argument("--dedup-iou", type=float, default=0.5)
    parser.add_argument(
        "--model",
        default=None,
        help="optional exact four-class model; only class-3 candidates are added",
    )
    parser.add_argument("--model-conf", type=float, default=0.20)
    parser.add_argument("--imgsz", type=int, default=960)
    args = parser.parse_args()

    out_dir = Path(args.out).resolve()
    ensure_fresh_output(out_dir)

    records = load_official_records(
        Path(args.annotations_dir), Path(args.images_dir)
    )
    with_scenes = assign_scene_ids(
        records, args.gps_radius_m, args.no_gps_minutes
    )
    eligible, exclusion = exclude_testing_scenes(with_scenes)
    selected, quota = select_with_none_quota(eligible, args.none_limit)

    seed = load_legacy_progress(Path(args.seed_progress), "seed_progress")
    reann = load_legacy_progress(Path(args.reann_progress), "reann_progress")
    model_path = Path(args.model).resolve() if args.model else None
    model_boxes = (
        predict_class3(selected, model_path, args.model_conf, args.imgsz)
        if model_path
        else {}
    )

    manifest: list[dict[str, Any]] = []
    for record in selected:
        image_id = record["id"]
        candidates: list[tuple[str, Iterable[dict[str, Any]]]] = [
            ("seed_progress", seed.get(image_id, [])),
            ("reann_progress", reann.get(image_id, [])),
        ]
        if model_path:
            candidates.append(("clean4_model_class3", model_boxes[image_id]))
        boxes = merge_untrusted_candidates(candidates, args.dedup_iou)
        requires_change = has_cross_class_overlap(boxes, args.dedup_iou)
        manifest.append(
            {
                "id": image_id,
                "file": record["file"],
                "preview": f"preview/{image_id}.jpg",
                "width": record["width"],
                "height": record["height"],
                "mode": record["mode"],
                "official_mode_name": MODE_NAME[record["mode"]],
                "target_class": MODE_TARGET[record["mode"]],
                "official_target_is_image_hint_only": True,
                "scene_id": record["scene_id"],
                "source_split": record["split"],
                "source_sha256": record["sha256"],
                "boxes": boxes,
                "n_boxes": len(boxes),
                "multi": len(boxes) > 1,
                "requires_change": requires_change,
                "sort_key": 0.0 if not boxes else min(
                    (float(box.get("conf", 0.0)) for box in boxes),
                    default=0.0,
                ),
                "review_note": REVIEW_NOTE,
            }
        )

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{out_dir.name}.", dir=out_dir.parent)
    )
    try:
        preview_dir = stage / "preview"
        preview_dir.mkdir()
        for record in selected:
            with Image.open(record["path"]) as image:
                preview = image.convert("RGB")
                preview.thumbnail((1600, 1600))
                preview.save(
                    preview_dir / f"{record['id']}.jpg", quality=90
                )
        (stage / "review_manifest_clean4.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        (stage / "review_progress_clean4.json").write_text(
            "{}\n", encoding="utf-8"
        )
        (stage / "selection_report.md").write_text(
            _render_report(
                with_scenes,
                eligible,
                selected,
                exclusion,
                quota,
                manifest,
                model_path,
                args.model_conf,
                args.imgsz,
                args.dedup_iou,
            ),
            encoding="utf-8",
        )
        ensure_fresh_output(out_dir)
        stage.replace(out_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise

    print(f"[OUT] {out_dir}")
    print(
        f"[SELECT] {len(selected)} images; "
        f"mode counts={dict(sorted(Counter(r['mode'] for r in selected).items()))}"
    )
    print(
        "[SEALED] official testing scenes excluded; "
        f"quota-excluded mode4={quota['quota_excluded_none']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
