#!/usr/bin/env python3
"""Validate, sanitize, and freeze a completed clean4 review progress file."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from ptl.constants import ATTESTATION_KEY as ATTESTATION
from ptl.constants import EXPECTED_NAMES as CLASS_NAMES

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _semantic_key(box: dict[str, Any]) -> tuple[int, float, float, float, float]:
    raw_cls = box["cls"]
    if isinstance(raw_cls, bool) or not isinstance(raw_cls, int):
        raise TypeError("class id must be an integer, not a coercible value")
    return (
        raw_cls,
        round(float(box["cx"]), 6),
        round(float(box["cy"]), 6),
        round(float(box["w"]), 6),
        round(float(box["h"]), 6),
    )


def _validate_box(item_id: str, index: int, box: dict[str, Any]) -> dict[str, Any]:
    try:
        cls, cx, cy, width, height = _semantic_key(box)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{item_id}: invalid box #{index}: {box!r}") from exc
    values = (cx, cy, width, height)
    if cls not in range(4):
        raise ValueError(f"{item_id}: class outside 0..3 in box #{index}")
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{item_id}: non-finite geometry in box #{index}")
    epsilon = 1e-6
    if (
        width <= 0
        or height <= 0
        or cx - width / 2 < -epsilon
        or cy - height / 2 < -epsilon
        or cx + width / 2 > 1 + epsilon
        or cy + height / 2 > 1 + epsilon
    ):
        raise ValueError(f"{item_id}: out-of-bounds geometry in box #{index}")
    return {"cls": cls, "cx": cx, "cy": cy, "w": width, "h": height}


def overlap_over_smaller(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_x1, left_y1 = left["cx"] - left["w"] / 2, left["cy"] - left["h"] / 2
    left_x2, left_y2 = left["cx"] + left["w"] / 2, left["cy"] + left["h"] / 2
    right_x1, right_y1 = (
        right["cx"] - right["w"] / 2,
        right["cy"] - right["h"] / 2,
    )
    right_x2, right_y2 = (
        right["cx"] + right["w"] / 2,
        right["cy"] + right["h"] / 2,
    )
    intersection = max(0.0, min(left_x2, right_x2) - max(left_x1, right_x1)) * max(
        0.0, min(left_y2, right_y2) - max(left_y1, right_y1)
    )
    return intersection / max(
        1e-12, min(left["w"] * left["h"], right["w"] * right["h"])
    )


def freeze(
    *,
    manifest: list[dict[str, Any]],
    progress: dict[str, Any],
    reviewer: str,
    attestation: str,
    overlap_threshold: float = 0.5,
    allowed_overlap_ids: set[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if attestation != ATTESTATION:
        raise ValueError(f"attestation must be exactly {ATTESTATION}")
    if not reviewer.strip():
        raise ValueError("reviewer must be non-empty")
    if not 0 < overlap_threshold <= 1:
        raise ValueError("overlap_threshold must be in (0, 1]")
    allowed_overlap_ids = allowed_overlap_ids or set()

    ids = [str(item["id"]) for item in manifest]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate IDs in manifest")
    missing = [
        item_id
        for item_id in ids
        if progress.get(item_id, {}).get("status") != "reviewed"
        or not isinstance(progress.get(item_id, {}).get("boxes"), list)
    ]
    extras = sorted(set(progress) - set(ids))
    if missing or extras:
        raise ValueError(
            f"review set mismatch: incomplete={len(missing)}, extras={len(extras)}"
        )

    frozen: dict[str, Any] = {}
    class_counts: Counter[int] = Counter()
    empty_images = 0
    changed_images = 0
    overlap_warnings: list[dict[str, Any]] = []
    requires_change_unchanged: list[str] = []

    for item in manifest:
        item_id = str(item["id"])
        boxes = [
            _validate_box(item_id, index, box)
            for index, box in enumerate(progress[item_id]["boxes"])
        ]
        keys = [_semantic_key(box) for box in boxes]
        if len(keys) != len(set(keys)):
            raise ValueError(f"{item_id}: exact duplicate boxes")
        for left_index, left in enumerate(boxes):
            for right_index, right in enumerate(boxes[left_index + 1 :], left_index + 1):
                score = overlap_over_smaller(left, right)
                if score >= overlap_threshold:
                    overlap_warnings.append(
                        {
                            "id": item_id,
                            "left_index": left_index,
                            "right_index": right_index,
                            "left_class": left["cls"],
                            "right_class": right["cls"],
                            "intersection_over_smaller": round(score, 6),
                        }
                    )
        source_keys = sorted(_semantic_key(box) for box in item.get("boxes", []))
        is_changed = sorted(keys) != source_keys
        if item.get("requires_change") and not is_changed:
            requires_change_unchanged.append(item_id)
        if is_changed:
            changed_images += 1
        if not boxes:
            empty_images += 1
        class_counts.update(box["cls"] for box in boxes)
        frozen[item_id] = {
            "status": "reviewed",
            "boxes": boxes,
            "changed": is_changed,
            "by": reviewer,
            "attestation": attestation,
        }

    unexpected_overlaps = [
        warning
        for warning in overlap_warnings
        if warning["id"] not in allowed_overlap_ids
    ]
    if unexpected_overlaps:
        raise ValueError(
            f"{len(unexpected_overlaps)} strong overlaps remain; "
            f"first={unexpected_overlaps[0]}"
        )
    if requires_change_unchanged:
        raise ValueError(
            "requires_change items still match untrusted seeds: "
            f"{requires_change_unchanged[:10]}"
        )

    report = {
        "schema_version": 1,
        "attestation": attestation,
        "reviewer": reviewer,
        "images": len(ids),
        "changed_images": changed_images,
        "empty_images": empty_images,
        "boxes": sum(class_counts.values()),
        "class_counts": {
            CLASS_NAMES[class_id]: class_counts[class_id] for class_id in range(4)
        },
        "invalid_boxes": 0,
        "exact_duplicate_boxes": 0,
        "strong_overlap_threshold": overlap_threshold,
        "strong_overlap_warnings": overlap_warnings,
        "allowed_overlap_ids": sorted(allowed_overlap_ids),
        "requires_change_unchanged": [],
        "box_fields": ["cls", "cx", "cy", "w", "h"],
    }
    return frozen, report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--progress", required=True)
    parser.add_argument("--out-progress", required=True)
    parser.add_argument("--out-report", required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--attestation", required=True)
    parser.add_argument("--overlap-threshold", type=float, default=0.5)
    parser.add_argument("--allow-overlap-id", action="append", default=[])
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    progress_path = Path(args.progress)
    out_progress = Path(args.out_progress)
    out_report = Path(args.out_report)
    if out_progress.resolve() == out_report.resolve():
        raise ValueError("out-progress and out-report must be different files")
    for path in (out_progress, out_report):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    frozen, report = freeze(
        manifest=manifest,
        progress=progress,
        reviewer=args.reviewer,
        attestation=args.attestation,
        overlap_threshold=args.overlap_threshold,
        allowed_overlap_ids=set(args.allow_overlap_id),
    )
    report["source_manifest_sha256"] = sha256_file(manifest_path)
    report["source_progress_sha256"] = sha256_file(progress_path)
    out_progress.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_progress.write_text(
        json.dumps(frozen, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    report["frozen_progress_sha256"] = sha256_file(out_progress)
    out_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[FROZEN] {out_progress} sha256={report['frozen_progress_sha256']}")
    print(f"[REPORT] {out_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
