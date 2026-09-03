#!/usr/bin/env python3
"""Compare per-frame predictions with a small, scene-specific ROI reference.

This is intentionally not a detector benchmark.  A reference describes broad
regions containing known signal heads and an expected class timeline.  The
script reports frame agreement, misses, wrong-only predictions, and conflicting
duplicate classes inside those regions.  It does not estimate mAP, precision,
recall, localization quality, or performance on other scenes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from ptl.constants import EXPECTED_NAMES

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


DISCLAIMER = (
    "Scene-specific ROI audit only. Broad manually verified regions and a "
    "manually inspected state timeline were used, not exhaustive frame-level "
    "bounding-box ground truth. Agreement rates are not mAP, precision, recall, "
    "general accuracy, or a safety certification."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_reference(reference: dict[str, Any], frame_count: int) -> None:
    if reference.get("schema_version") != 1:
        raise ValueError("unsupported reference schema_version")
    if int(reference.get("source_frame_count", -1)) != frame_count:
        raise ValueError(
            "reference source_frame_count does not match prediction frames: "
            f"{reference.get('source_frame_count')} != {frame_count}"
        )
    names = reference.get("class_names")
    if names != EXPECTED_NAMES:
        raise ValueError(f"unexpected reference class contract: {names}")
    frame_width = reference.get("frame_width")
    frame_height = reference.get("frame_height")
    if (
        isinstance(frame_width, bool)
        or isinstance(frame_height, bool)
        or not isinstance(frame_width, int)
        or not isinstance(frame_height, int)
        or frame_width <= 0
        or frame_height <= 0
    ):
        raise ValueError("reference frame_width/frame_height must be positive integers")
    regions = reference.get("regions")
    if not isinstance(regions, list) or not regions:
        raise ValueError("reference regions must be a non-empty list")
    region_ids = [str(region.get("id", "")).strip() for region in regions]
    if any(not region_id for region_id in region_ids):
        raise ValueError("every reference region must have a non-empty id")
    if len(region_ids) != len(set(region_ids)):
        raise ValueError("reference region ids must be unique")
    for region in regions:
        roi = region.get("roi_xyxy_pixels")
        if (
            not isinstance(roi, list)
            or len(roi) != 4
            or not all(isinstance(value, (int, float)) for value in roi)
            or any(isinstance(value, bool) for value in roi)
            or not all(math.isfinite(value) for value in roi)
            or not (
                0 <= roi[0] < roi[2] <= frame_width
                and 0 <= roi[1] < roi[3] <= frame_height
            )
        ):
            raise ValueError(f"invalid ROI for {region.get('id')}: {roi}")
        segments = region.get("expected_segments")
        if not isinstance(segments, list) or not segments:
            raise ValueError(f"missing expected_segments for {region.get('id')}")
        covered = [False] * frame_count
        for segment in segments:
            if (
                not isinstance(segment, list)
                or len(segment) != 3
                or not all(isinstance(value, int) for value in segment[:2])
                or any(isinstance(value, bool) for value in segment[:2])
            ):
                raise ValueError(f"invalid segment for {region.get('id')}: {segment}")
            start, end, class_id = segment
            if (
                start < 0
                or end < start
                or end >= frame_count
                or (
                    class_id is not None
                    and (
                        isinstance(class_id, bool)
                        or not isinstance(class_id, int)
                        or class_id not in range(4)
                    )
                )
            ):
                raise ValueError(f"out-of-range segment for {region.get('id')}: {segment}")
            for frame_index in range(start, end + 1):
                if covered[frame_index]:
                    raise ValueError(
                        f"overlapping segments for {region.get('id')} at {frame_index}"
                    )
                covered[frame_index] = True
        if not all(covered):
            raise ValueError(f"segments do not cover every frame for {region.get('id')}")


def expected_by_frame(
    region: dict[str, Any], frame_count: int
) -> list[int | None]:
    expected: list[int | None] = [None] * frame_count
    for start, end, class_id in region["expected_segments"]:
        expected[start : end + 1] = [class_id] * (end - start + 1)
    return expected


def detection_in_roi(detection: dict[str, Any], roi: list[float]) -> bool:
    x1, y1, x2, y2 = (float(value) for value in detection["xyxy_pixels"])
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return roi[0] <= center_x <= roi[2] and roi[1] <= center_y <= roi[3]


def _rate(count: int, total: int) -> float | None:
    return count / total if total else None


def audit_region(
    model_frames: list[dict[str, Any]],
    region: dict[str, Any],
    threshold: float,
) -> dict[str, Any]:
    frame_count = len(model_frames)
    expected = expected_by_frame(region, frame_count)
    expected_counts = Counter(
        class_id for class_id in expected if class_id is not None
    )
    scored_frames = sum(class_id is not None for class_id in expected)
    ignored_frames = frame_count - scored_frames
    predicted_class_frames: Counter[int] = Counter()
    expected_hit = clean_hit = wrong_only = conflict = missed = any_detection = 0
    non_green_frames = green_expected_frames = 0
    false_green_frames = green_hit_frames = 0
    expected_hit_by_class: Counter[int] = Counter()
    conflict_by_expected_class: Counter[int] = Counter()
    false_green_by_expected_class: Counter[int] = Counter()

    for frame_index, model_frame in enumerate(model_frames):
        target = expected[frame_index]
        if target is None:
            continue
        detections = [
            detection
            for detection in model_frame.get("detections", [])
            if float(detection["confidence"]) >= threshold
            and detection_in_roi(detection, region["roi_xyxy_pixels"])
        ]
        predicted = {int(detection["class_id"]) for detection in detections}
        for class_id in predicted:
            predicted_class_frames[class_id] += 1
        if detections:
            any_detection += 1
        if target == 1:
            green_expected_frames += 1
            if 1 in predicted:
                green_hit_frames += 1
        else:
            non_green_frames += 1
            if 1 in predicted:
                false_green_frames += 1
                false_green_by_expected_class[target] += 1

        if not detections:
            missed += 1
        elif target in predicted:
            expected_hit += 1
            expected_hit_by_class[target] += 1
            if predicted == {target}:
                clean_hit += 1
            else:
                conflict += 1
                conflict_by_expected_class[target] += 1
        else:
            wrong_only += 1

    return {
        "threshold": threshold,
        "source_frames": frame_count,
        "frames": scored_frames,
        "ignored_frames": ignored_frames,
        "expected_class_frames": {
            str(class_id): expected_counts[class_id] for class_id in range(4)
        },
        "predicted_class_frame_coverage": {
            str(class_id): {
                "frames": predicted_class_frames[class_id],
                "rate": _rate(predicted_class_frames[class_id], scored_frames),
            }
            for class_id in range(4)
        },
        "outcomes": {
            "any_detection": {
                "frames": any_detection,
                "rate": _rate(any_detection, scored_frames),
            },
            "expected_class_hit": {
                "frames": expected_hit,
                "rate": _rate(expected_hit, scored_frames),
            },
            "clean_expected_class_only": {
                "frames": clean_hit,
                "rate": _rate(clean_hit, scored_frames),
            },
            "expected_plus_conflicting_class": {
                "frames": conflict,
                "rate": _rate(conflict, scored_frames),
            },
            "wrong_class_only": {
                "frames": wrong_only,
                "rate": _rate(wrong_only, scored_frames),
            },
            "missed": {
                "frames": missed,
                "rate": _rate(missed, scored_frames),
            },
        },
        "expected_hit_by_class": {
            str(class_id): {
                "hit_frames": expected_hit_by_class[class_id],
                "expected_frames": expected_counts[class_id],
                "agreement_rate": _rate(
                    expected_hit_by_class[class_id], expected_counts[class_id]
                ),
                "conflict_frames": conflict_by_expected_class[class_id],
            }
            for class_id in range(4)
            if expected_counts[class_id]
        },
        "green_gate_observation": {
            "green_expected_frames": green_expected_frames,
            "green_hit_frames": green_hit_frames,
            "green_hit_rate": _rate(green_hit_frames, green_expected_frames),
            "non_green_expected_frames": non_green_frames,
            "green_on_non_green_frames": false_green_frames,
            "green_on_non_green_rate": _rate(false_green_frames, non_green_frames),
            "green_on_non_green_by_expected_class": {
                str(class_id): false_green_by_expected_class[class_id]
                for class_id in (0, 2, 3)
                if expected_counts[class_id]
            },
        },
    }


def evaluate(
    frames: list[dict[str, Any]],
    reference: dict[str, Any],
    thresholds: list[float],
) -> dict[str, Any]:
    _validate_reference(reference, len(frames))
    if not frames:
        raise ValueError("prediction frames are empty")
    models = list(frames[0].get("models", {}))
    if not models:
        raise ValueError("no models in prediction frames")
    if any(list(frame.get("models", {})) != models for frame in frames):
        raise ValueError("model order/set changes between prediction frames")
    first_schemas = {
        model: str(frames[0]["models"][model].get("schema", "unknown"))
        for model in models
    }
    if any(
        str(frame["models"][model].get("schema", "unknown"))
        != first_schemas[model]
        for frame in frames
        for model in models
    ):
        raise ValueError("model schema changes between prediction frames")
    for expected_index, frame in enumerate(frames):
        if int(frame.get("source_frame_index", -1)) != expected_index:
            raise ValueError(
                "ROI reference requires stride=1 contiguous source frames; "
                f"record {expected_index} has source_frame_index="
                f"{frame.get('source_frame_index')}"
            )
    if not thresholds or any(
        not math.isfinite(value) or value < 0 or value > 1 for value in thresholds
    ):
        raise ValueError("thresholds must be finite values in [0, 1]")
    threshold_keys = [f"{threshold:.2f}" for threshold in thresholds]
    if len(threshold_keys) != len(set(threshold_keys)):
        raise ValueError("thresholds collide after two-decimal report formatting")

    output: dict[str, Any] = {
        "schema_version": 1,
        "evaluation_kind": "scene_specific_roi_reference_audit",
        "disclaimer": DISCLAIMER,
        "formal_detector_metrics_allowed": False,
        "reference_note": reference.get("note", ""),
        "source_video_sha256": reference.get("source_video_sha256"),
        "source_frame_count": len(frames),
        "thresholds": thresholds,
        "models": {},
    }
    for model in models:
        schema = str(frames[0]["models"][model].get("schema", "unknown"))
        model_frames = [
            frame["models"][model]
            for frame in frames
        ]
        output["models"][model] = {
            "schema": schema,
            "regions": {
                region["id"]: {
                    "description": region.get("description", ""),
                    "expected_object_type": region.get("expected_object_type", ""),
                    "roi_xyxy_pixels": region["roi_xyxy_pixels"],
                    "thresholds": {
                        f"{threshold:.2f}": audit_region(
                            model_frames, region, threshold
                        )
                        for threshold in thresholds
                    },
                }
                for region in reference["regions"]
            },
        }
    return output


def validate_prediction_binding(
    *,
    summary: dict[str, Any],
    frames_sha256: str,
    video_sha256: str,
    frame_count: int,
    thresholds: list[float],
    models: list[str],
    model_schemas: dict[str, str],
) -> None:
    """Fail closed unless prediction records are bound to this video/settings."""
    summary_video = summary.get("video", {})
    recorded_video_hash = str(summary_video.get("sha256", "")).lower()
    if recorded_video_hash != video_sha256.lower():
        raise ValueError(
            "prediction summary video SHA256 mismatch or missing: "
            f"{recorded_video_hash or '<missing>'} != {video_sha256.lower()}"
        )
    recorded_frames_hash = str(
        summary.get("artifacts", {}).get("per_frame_json_sha256", "")
    ).lower()
    if recorded_frames_hash != frames_sha256.lower():
        raise ValueError(
            "prediction frames SHA256 mismatch or missing from summary: "
            f"{recorded_frames_hash or '<missing>'} != {frames_sha256.lower()}"
        )
    if int(summary_video.get("sampled_frames_processed", -1)) != frame_count:
        raise ValueError("prediction summary frame count does not match frames.json")
    settings = summary.get("settings", {})
    if int(settings.get("stride", -1)) != 1:
        raise ValueError("ROI audit requires predictions generated with stride=1")
    generation_conf = float(settings.get("conf", float("nan")))
    if not math.isfinite(generation_conf) or not 0 <= generation_conf <= 1:
        raise ValueError("prediction summary generation conf is missing or invalid")
    if any(threshold < generation_conf for threshold in thresholds):
        raise ValueError(
            "ROI threshold cannot be lower than prediction generation conf "
            f"({generation_conf})"
        )
    if summary.get("model_order") != models:
        raise ValueError("prediction summary model order does not match frames.json")
    summary_models = summary.get("models", {})
    for label in models:
        model_info = summary_models.get(label, {})
        model_hash = str(model_info.get("sha256", "")).lower()
        if not re.fullmatch(r"[0-9a-f]{64}", model_hash):
            raise ValueError(f"prediction summary model SHA256 missing: {label}")
        recorded_schema = str(model_info.get("schema", ""))
        if recorded_schema not in {"legacy3", "clean4"}:
            raise ValueError(f"prediction summary model schema invalid: {label}")
        if recorded_schema != model_schemas.get(label):
            raise ValueError(
                f"prediction summary model schema does not match frames.json: {label}"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True, help="compare_models_video.py frames.json")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--video", required=True, help="source video, verified by SHA256")
    parser.add_argument(
        "--summary",
        default=None,
        help="compare_models_video.py summary.json; defaults beside frames.json",
    )
    parser.add_argument(
        "--allow-unbound-frames",
        action="store_true",
        help="legacy escape hatch: allow old frames.json without SHA-bound summary",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--threshold",
        action="append",
        type=float,
        default=[],
        help="repeatable confidence threshold; defaults to 0.25 and 0.70",
    )
    args = parser.parse_args()

    frames_path = Path(args.frames)
    reference_path = Path(args.reference)
    video_path = Path(args.video)
    output_path = Path(args.out)
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    actual_video_hash = sha256_file(video_path)
    expected_video_hash = str(reference.get("source_video_sha256", "")).lower()
    if actual_video_hash != expected_video_hash:
        raise ValueError(
            f"source video SHA256 mismatch: {actual_video_hash} != {expected_video_hash}"
        )
    frames = json.loads(frames_path.read_text(encoding="utf-8"))
    if not isinstance(frames, list) or not frames:
        raise ValueError("prediction frames must be a non-empty list")
    thresholds = args.threshold or [0.25, 0.70]
    frames_hash = sha256_file(frames_path)
    summary_path = (
        Path(args.summary) if args.summary else frames_path.with_name("summary.json")
    )
    if not args.allow_unbound_frames:
        if not summary_path.is_file():
            raise FileNotFoundError(
                f"prediction summary is required for SHA binding: {summary_path}"
            )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        validate_prediction_binding(
            summary=summary,
            frames_sha256=frames_hash,
            video_sha256=actual_video_hash,
            frame_count=len(frames),
            thresholds=thresholds,
            models=list(frames[0].get("models", {})),
            model_schemas={
                label: str(model_frame.get("schema", "unknown"))
                for label, model_frame in frames[0].get("models", {}).items()
            },
        )
    report = evaluate(frames, reference, thresholds)
    report["prediction_binding_verified"] = not args.allow_unbound_frames
    report["prediction_summary"] = (
        str(summary_path) if not args.allow_unbound_frames else None
    )
    report["prediction_summary_sha256"] = (
        sha256_file(summary_path) if not args.allow_unbound_frames else None
    )
    report["prediction_models"] = (
        {
            label: {
                key: summary["models"][label][key]
                for key in ("path", "sha256", "schema")
            }
            for label in summary["model_order"]
        }
        if not args.allow_unbound_frames
        else None
    )
    report["prediction_frames_sha256"] = frames_hash
    report["reference_sha256"] = sha256_file(reference_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[OUT] {output_path}")
    print(f"[NOTICE] {DISCLAIMER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
