#!/usr/bin/env python3
"""在同一视频、同一推理设置下比较多个 YOLO 检测模型的输出行为。

本工具没有使用逐框真值（GT），所以只报告检测覆盖、置信度、连续性、抖动和
推理耗时；这些指标不能解释为 precision、recall、mAP 或安全准确率。

典型用法（旧三类 vs 新四类）:

  python scripts/compare_models_video.py ^
      --video data/raw/mine/example.mp4 ^
      --model old3=models/PedestrianSignalIcon_v1_yolo11n.pt ^
      --model new4=models/PedestrianSignal4_video_v1_yolo11n.pt ^
      --stride 3 --conf 0.25 --imgsz 960 ^
      --out runs/video_compare/example

输出:
  frames.json                 每个抽样帧、每个模型的完整预测
  summary.json                无 GT 的稳定性/耗时汇总
  <label>_annotated.mp4       每个模型独立画框视频
  comparison_tiled.mp4        同帧横向拼接视频
  comparison_contact.jpg      若干等距抽样帧的接触表
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from ptl.constants import EXPECTED_NAMES as CLEAN4_NAMES
from ptl.constants import LEGACY3_NAMES

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


SCHEMAS = {
    tuple(LEGACY3_NAMES): "legacy3",
    tuple(CLEAN4_NAMES): "clean4",
}

# OpenCV 使用 BGR。
CLASS_COLORS = {
    0: (30, 30, 240),
    1: (40, 210, 40),
    2: (20, 170, 245),
    3: (235, 145, 40),
}

NO_GT_DISCLAIMER = (
    "No ground-truth boxes were used. Prediction coverage, confidence, "
    "continuity, flicker, and timing are descriptive only and must not be "
    "reported as accuracy, precision, recall, mAP, or a safety result."
)


def normalize_names(raw: Any) -> list[str]:
    """规范化 Ultralytics ``model.names``，同时验证 class id 连续且名称唯一。"""
    if isinstance(raw, dict):
        try:
            pairs = sorted((int(key), str(value)) for key, value in raw.items())
        except (TypeError, ValueError) as exc:
            raise ValueError(f"模型 class id 不能转换为整数: {raw!r}") from exc
        ids = [key for key, _ in pairs]
        if ids != list(range(len(ids))):
            raise ValueError(f"模型 class id 必须从 0 连续编号，实际为 {ids}")
        names = [value for _, value in pairs]
    elif isinstance(raw, (list, tuple)):
        names = [str(value) for value in raw]
    else:
        raise ValueError(f"无法识别模型 names: {raw!r}")

    if not names or any(not name.strip() for name in names):
        raise ValueError("模型 class name 不能为空")
    if len(set(names)) != len(names):
        raise ValueError(f"模型 class name 必须唯一，实际为 {names}")
    return names


def identify_schema(raw_names: Any) -> tuple[list[str], str]:
    """只接受项目已定义的旧三类或当前四类契约，避免静默错位比较。"""
    names = normalize_names(raw_names)
    schema = SCHEMAS.get(tuple(names))
    if schema is None:
        raise ValueError(
            "不支持的类别契约。仅接受旧三类 "
            f"{LEGACY3_NAMES} 或当前四类 {CLEAN4_NAMES}；实际为 {names}"
        )
    return names, schema


def parse_model_spec(value: str) -> tuple[str, Path]:
    """解析 ``LABEL=PATH``；LABEL 同时用于安全的输出文件名。"""
    label, separator, raw_path = value.partition("=")
    label = label.strip()
    raw_path = raw_path.strip()
    if not separator or not label or not raw_path:
        raise ValueError(f"--model 必须是 LABEL=PATH，实际为 {value!r}")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
        raise ValueError(
            f"模型标签 {label!r} 只能包含英文字母、数字、点、下划线和连字符"
        )
    return label, Path(raw_path)


def parse_model_specs(values: Iterable[str]) -> list[tuple[str, Path]]:
    specs = [parse_model_spec(value) for value in values]
    labels = [label for label, _ in specs]
    if len(set(labels)) != len(labels):
        duplicates = sorted(label for label in set(labels) if labels.count(label) > 1)
        raise ValueError(f"--model 标签重复: {duplicates}")
    if len(specs) < 2:
        raise ValueError("至少提供两个 --model LABEL=PATH 才能进行对比")
    return specs


def percentile(values: Iterable[float], fraction: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def describe_values(values: Iterable[float]) -> dict[str, float | int | None]:
    data = [float(value) for value in values]
    if not data:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(data),
        "mean": statistics.fmean(data),
        "median": statistics.median(data),
        "p90": percentile(data, 0.90),
        "p95": percentile(data, 0.95),
        "min": min(data),
        "max": max(data),
    }


def longest_run(sequence: Iterable[bool], value: bool) -> int:
    best = current = 0
    for item in sequence:
        if bool(item) is value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def internal_singleton_count(sequence: Iterable[bool], value: bool) -> int:
    """统计严格夹在相反状态之间的单采样尖峰或单采样掉线。"""
    values = [bool(item) for item in sequence]
    return sum(
        values[index] is value
        and values[index - 1] is not value
        and values[index + 1] is not value
        for index in range(1, len(values) - 1)
    )


def planned_contact_indices(
    source_frame_count: int,
    stride: int,
    max_processed_frames: int,
    contact_frames: int,
) -> set[int]:
    """返回处理帧序号（不是源视频帧号）的等距接触表位置。"""
    if contact_frames <= 0:
        return set()
    if source_frame_count <= 0:
        return set(range(contact_frames))
    total = (source_frame_count + stride - 1) // stride
    if max_processed_frames > 0:
        total = min(total, max_processed_frames)
    if total <= 0:
        return set()
    if contact_frames == 1:
        return {0}
    if total <= contact_frames:
        return set(range(total))
    return {
        round(index * (total - 1) / (contact_frames - 1))
        for index in range(contact_frames)
    }


def summarize_prediction_frames(
    frames: list[dict[str, Any]],
    class_names: list[str],
    sample_period_seconds: float,
) -> dict[str, Any]:
    """汇总无 GT 的预测稳定性指标，供主流程和单元测试共同使用。"""
    frame_count = len(frames)
    detection_counts = [len(frame.get("detections", [])) for frame in frames]
    inference_ms = [float(frame["inference_ms"]) for frame in frames]
    wall_ms = [float(frame["wall_ms"]) for frame in frames]

    present_sets: list[tuple[int, ...]] = []
    dominant: list[int | None] = []
    for frame in frames:
        detections = frame.get("detections", [])
        present_sets.append(
            tuple(sorted({int(det["class_id"]) for det in detections}))
        )
        dominant.append(
            int(max(detections, key=lambda det: float(det["confidence"]))["class_id"])
            if detections
            else None
        )

    adjacency_count = max(0, frame_count - 1)
    set_transitions = sum(
        present_sets[index] != present_sets[index - 1]
        for index in range(1, frame_count)
    )
    dominant_transitions = sum(
        dominant[index] != dominant[index - 1]
        for index in range(1, frame_count)
    )

    class_metrics: dict[str, Any] = {}
    for class_id, class_name in enumerate(class_names):
        presence = [
            any(
                int(det["class_id"]) == class_id
                for det in frame.get("detections", [])
            )
            for frame in frames
        ]
        confidences = [
            float(det["confidence"])
            for frame in frames
            for det in frame.get("detections", [])
            if int(det["class_id"]) == class_id
        ]
        detections = len(confidences)
        covered = sum(presence)
        state_transitions = sum(
            presence[index] != presence[index - 1]
            for index in range(1, frame_count)
        )
        longest_presence = longest_run(presence, True)
        longest_gap = longest_run(presence, False)
        one_sample_spikes = internal_singleton_count(presence, True)
        one_sample_dropouts = internal_singleton_count(presence, False)
        class_metrics[str(class_id)] = {
            "class_id": class_id,
            "class_name": class_name,
            "frame_coverage": {
                "frames": covered,
                "rate": covered / frame_count if frame_count else None,
            },
            "detections": detections,
            "detections_per_frame": detections / frame_count if frame_count else None,
            "confidence": describe_values(confidences),
            "continuity": {
                "longest_presence_run_sampled_frames": longest_presence,
                "longest_presence_run_approx_seconds": (
                    longest_presence * sample_period_seconds
                ),
                "longest_absence_gap_sampled_frames": longest_gap,
                "longest_absence_gap_approx_seconds": (
                    longest_gap * sample_period_seconds
                ),
                "state_transition_count": state_transitions,
                "state_transition_rate_per_adjacency": (
                    state_transitions / adjacency_count
                    if adjacency_count
                    else None
                ),
                "one_sample_presence_spikes": one_sample_spikes,
                "one_sample_absence_dropouts": one_sample_dropouts,
                "flicker_event_count": one_sample_spikes + one_sample_dropouts,
            },
        }

    any_presence = [count > 0 for count in detection_counts]
    return {
        "sampled_frames": frame_count,
        "sample_period_approx_seconds": sample_period_seconds,
        "any_detection_frame_coverage": {
            "frames": sum(any_presence),
            "rate": sum(any_presence) / frame_count if frame_count else None,
        },
        "detections": {
            "total": sum(detection_counts),
            "per_frame": (
                sum(detection_counts) / frame_count if frame_count else None
            ),
            "count_per_frame_distribution": describe_values(detection_counts),
        },
        "confidence_all_detections": describe_values(
            det["confidence"]
            for frame in frames
            for det in frame.get("detections", [])
        ),
        "empty_frame_longest_gap": {
            "sampled_frames": longest_run(any_presence, False),
            "approx_seconds": (
                longest_run(any_presence, False) * sample_period_seconds
            ),
        },
        "prediction_set_transition": {
            "count": set_transitions,
            "rate_per_adjacency": (
                set_transitions / adjacency_count if adjacency_count else None
            ),
        },
        "dominant_class_transition": {
            "count": dominant_transitions,
            "rate_per_adjacency": (
                dominant_transitions / adjacency_count
                if adjacency_count
                else None
            ),
        },
        "timing_ms": {
            "model_reported_inference": describe_values(inference_ms),
            "wall_predict_call": describe_values(wall_ms),
        },
        "per_class": class_metrics,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_detections(result: Any, class_names: list[str]) -> list[dict[str, Any]]:
    """把 Ultralytics Result 转成稳定、可审计的 JSON 结构。"""
    height, width = (int(value) for value in result.orig_shape)
    detections: list[dict[str, Any]] = []
    for box in result.boxes:
        class_id = int(box.cls[0])
        if class_id < 0 or class_id >= len(class_names):
            raise ValueError(
                f"模型输出越界 class id {class_id}，类别数为 {len(class_names)}"
            )
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
        detections.append(
            {
                "class_id": class_id,
                "class_name": class_names[class_id],
                "confidence": confidence,
                "xyxy_pixels": [x1, y1, x2, y2],
                "xyxyn": [
                    x1 / width,
                    y1 / height,
                    x2 / width,
                    y2 / height,
                ],
            }
        )
    detections.sort(key=lambda item: item["confidence"], reverse=True)
    return detections


def annotate_frame(
    frame: Any,
    detections: list[dict[str, Any]],
    label: str,
    schema: str,
    source_frame_index: int,
) -> Any:
    import cv2

    annotated = frame.copy()
    height, width = annotated.shape[:2]
    thickness = max(2, round(max(height, width) / 700))
    font_scale = max(0.45, min(1.1, max(height, width) / 1300))
    header = f"{label} [{schema}]  source_frame={source_frame_index}"
    cv2.rectangle(annotated, (0, 0), (width, 34), (20, 20, 20), -1)
    cv2.putText(
        annotated,
        header,
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    for detection in detections:
        class_id = int(detection["class_id"])
        color = CLASS_COLORS.get(class_id, (210, 210, 210))
        x1, y1, x2, y2 = (
            round(value) for value in detection["xyxy_pixels"]
        )
        x1 = min(max(x1, 0), width - 1)
        x2 = min(max(x2, 0), width - 1)
        y1 = min(max(y1, 0), height - 1)
        y2 = min(max(y2, 0), height - 1)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        text = (
            f"{detection['class_name']} "
            f"{float(detection['confidence']):.2f}"
        )
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        text_y = max(36 + text_height, y1)
        cv2.rectangle(
            annotated,
            (x1, text_y - text_height - baseline - 3),
            (min(width - 1, x1 + text_width + 4), text_y + baseline),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            text,
            (x1 + 2, text_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (10, 10, 10),
            max(1, thickness - 1),
            cv2.LINE_AA,
        )
    return annotated


def resize_tile(frame: Any, tile_width: int) -> Any:
    import cv2

    height, width = frame.shape[:2]
    tile_height = max(1, round(height * tile_width / width))
    interpolation = cv2.INTER_AREA if width > tile_width else cv2.INTER_LINEAR
    return cv2.resize(frame, (tile_width, tile_height), interpolation=interpolation)


def write_image(path: Path, image: Any) -> None:
    """用 imencode+tofile 兼容 Windows Unicode 路径。"""
    import cv2

    ok, encoded = cv2.imencode(path.suffix or ".jpg", image)
    if not ok:
        raise RuntimeError(f"无法编码图片: {path}")
    encoded.tofile(str(path))


def build_contact_sheet(
    rows: list[tuple[int, dict[str, Any]]],
    labels: list[str],
    tile_width: int,
) -> Any | None:
    import cv2
    import numpy as np

    strips = []
    for _, by_model in sorted(rows, key=lambda item: item[0]):
        if any(label not in by_model for label in labels):
            continue
        tiles = [resize_tile(by_model[label], tile_width) for label in labels]
        strips.append(np.hstack(tiles))
    if not strips:
        return None
    separator = np.full((6, strips[0].shape[1], 3), 35, dtype=np.uint8)
    parts = []
    for index, strip in enumerate(strips):
        if index:
            parts.append(separator)
        parts.append(strip)
    return cv2.vconcat(parts)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "同视频、同设置比较旧3类/新4类 YOLO 模型的预测稳定性；"
            "无 GT，不计算准确率。"
        )
    )
    parser.add_argument("--video", required=True, help="输入视频路径")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="重复传入至少两个模型；LABEL 用于输出文件名",
    )
    parser.add_argument("--out", default="runs/video_compare", help="输出目录")
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="每隔多少个源视频帧推理一次（默认 1）",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="最多处理多少个抽样帧；0 表示不限制",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument(
        "--device",
        default=None,
        help="Ultralytics device，例如 cpu、0；默认由框架选择",
    )
    parser.add_argument(
        "--codec",
        default="mp4v",
        help="四字符 OpenCV VideoWriter codec（默认 mp4v）",
    )
    parser.add_argument(
        "--tile-width",
        type=int,
        default=480,
        help="拼接视频/接触表中每个模型列的宽度",
    )
    parser.add_argument(
        "--contact-frames",
        type=int,
        default=8,
        help="接触表等距保留的抽样帧数；0 表示不生成",
    )
    parser.add_argument(
        "--no-tiled-video",
        action="store_true",
        help="不生成横向拼接 comparison_tiled.mp4",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.stride < 1:
        raise ValueError("--stride 必须 >= 1")
    if args.max_frames < 0:
        raise ValueError("--max-frames 必须 >= 0")
    if not 0 <= args.conf <= 1:
        raise ValueError("--conf 必须在 [0,1]")
    if not 0 <= args.iou <= 1:
        raise ValueError("--iou 必须在 [0,1]")
    if args.imgsz < 32:
        raise ValueError("--imgsz 必须 >= 32")
    if args.tile_width < 64:
        raise ValueError("--tile-width 必须 >= 64")
    if args.contact_frames < 0:
        raise ValueError("--contact-frames 必须 >= 0")
    if len(args.codec) != 4:
        raise ValueError("--codec 必须正好四个字符，例如 mp4v")


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        _validate_args(args)
        specs = parse_model_specs(args.model)
    except ValueError as exc:
        parser.error(str(exc))

    video_path = Path(args.video)
    if not video_path.is_file():
        print(f"[STOP] 输入视频不存在: {video_path}", file=sys.stderr)
        return 2
    missing_models = [str(path) for _, path in specs if not path.is_file()]
    if missing_models:
        print(f"[STOP] 模型文件不存在: {missing_models}", file=sys.stderr)
        return 2
    source_video_sha256 = sha256_file(video_path)

    import cv2
    import numpy as np
    from ultralytics import YOLO

    models: dict[str, dict[str, Any]] = {}
    try:
        for label, path in specs:
            model = YOLO(str(path))
            names, schema = identify_schema(model.names)
            models[label] = {
                "model": model,
                "path": path,
                "names": names,
                "schema": schema,
                "sha256": sha256_file(path),
            }
            print(f"[MODEL] {label}: {schema}, {len(names)} 类, {path}")
    except (ValueError, RuntimeError) as exc:
        print(f"[STOP] 模型类别契约校验失败: {exc}", file=sys.stderr)
        return 2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        print(f"[STOP] 无法打开视频: {video_path}", file=sys.stderr)
        return 2
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if source_fps <= 0 or source_width <= 0 or source_height <= 0:
        capture.release()
        print("[STOP] 视频 FPS 或分辨率元数据无效", file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_fps = source_fps / args.stride
    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    writers: dict[str, Any] = {}
    model_video_paths: dict[str, Path] = {}
    tiled_writer = None
    tiled_path = out_dir / "comparison_tiled.mp4"

    try:
        for label in models:
            output_path = out_dir / f"{label}_annotated.mp4"
            writer = cv2.VideoWriter(
                str(output_path),
                fourcc,
                output_fps,
                (source_width, source_height),
            )
            if not writer.isOpened():
                raise RuntimeError(
                    f"无法创建视频 {output_path}；可尝试 --codec avc1 或 mp4v"
                )
            writers[label] = writer
            model_video_paths[label] = output_path

        tiled_height = max(
            1, round(source_height * args.tile_width / source_width)
        )
        if not args.no_tiled_video:
            tiled_writer = cv2.VideoWriter(
                str(tiled_path),
                fourcc,
                output_fps,
                (args.tile_width * len(models), tiled_height),
            )
            if not tiled_writer.isOpened():
                raise RuntimeError(f"无法创建拼接视频 {tiled_path}")
    except RuntimeError as exc:
        capture.release()
        for writer in writers.values():
            writer.release()
        if tiled_writer is not None:
            tiled_writer.release()
        print(f"[STOP] {exc}", file=sys.stderr)
        return 2

    contact_indices = planned_contact_indices(
        source_frame_count,
        args.stride,
        args.max_frames,
        args.contact_frames,
    )
    contact_rows: list[tuple[int, dict[str, Any]]] = []
    metric_frames: dict[str, list[dict[str, Any]]] = {
        label: [] for label in models
    }
    frames_path = out_dir / "frames.json"
    frames_tmp = out_dir / "frames.json.tmp"
    frame_handle = frames_tmp.open("w", encoding="utf-8")
    frame_handle.write("[\n")
    first_json_record = True
    source_index = 0
    processed_index = 0
    warmup_done = False
    failed: Exception | None = None

    predict_kwargs: dict[str, Any] = {
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "verbose": False,
    }
    if args.device is not None:
        predict_kwargs["device"] = args.device

    print(
        f"[VIDEO] {source_width}x{source_height}, {source_fps:.3f} FPS, "
        f"{source_frame_count} 帧; stride={args.stride}"
    )
    print(f"[NOTICE] {NO_GT_DISCLAIMER}")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if source_index % args.stride != 0:
                source_index += 1
                continue
            if args.max_frames and processed_index >= args.max_frames:
                break

            # 每个模型先在完全相同的首个抽样帧上预热一次，冷启动不进入耗时统计。
            if not warmup_done:
                for info in models.values():
                    info["model"].predict(frame, **predict_kwargs)
                warmup_done = True

            frame_record: dict[str, Any] = {
                "processed_index": processed_index,
                "source_frame_index": source_index,
                "timestamp_seconds": source_index / source_fps,
                "models": {},
            }
            annotated_by_model: dict[str, Any] = {}

            for label, info in models.items():
                started = time.perf_counter()
                result = info["model"].predict(frame, **predict_kwargs)[0]
                wall_ms = (time.perf_counter() - started) * 1000
                detections = extract_detections(result, info["names"])
                speed = dict(getattr(result, "speed", {}) or {})
                inference_ms = float(speed.get("inference", wall_ms))
                timing = {
                    "preprocess_ms": (
                        float(speed["preprocess"])
                        if speed.get("preprocess") is not None
                        else None
                    ),
                    "inference_ms": inference_ms,
                    "postprocess_ms": (
                        float(speed["postprocess"])
                        if speed.get("postprocess") is not None
                        else None
                    ),
                    "wall_predict_call_ms": wall_ms,
                }
                frame_record["models"][label] = {
                    "schema": info["schema"],
                    "timing": timing,
                    "detections": detections,
                }
                metric_frames[label].append(
                    {
                        "inference_ms": inference_ms,
                        "wall_ms": wall_ms,
                        "detections": [
                            {
                                "class_id": detection["class_id"],
                                "confidence": detection["confidence"],
                            }
                            for detection in detections
                        ],
                    }
                )
                annotated = annotate_frame(
                    frame,
                    detections,
                    label,
                    info["schema"],
                    source_index,
                )
                writers[label].write(annotated)
                annotated_by_model[label] = annotated

            if tiled_writer is not None:
                tiled = np.hstack(
                    [
                        resize_tile(annotated_by_model[label], args.tile_width)
                        for label in models
                    ]
                )
                tiled_writer.write(tiled)

            if processed_index in contact_indices:
                # 接触表只需缩略图。这里立即降采样，避免 4K 视频按
                # contact_frames × model_count 长时间占用数百 MB 内存。
                contact_rows.append(
                    (
                        source_index,
                        {
                            label: resize_tile(
                                annotated_by_model[label], args.tile_width
                            )
                            for label in models
                        },
                    )
                )

            if not first_json_record:
                frame_handle.write(",\n")
            json.dump(frame_record, frame_handle, ensure_ascii=False)
            first_json_record = False

            processed_index += 1
            source_index += 1
            if processed_index % 100 == 0:
                print(f"  ... 已处理 {processed_index} 个抽样帧")
    except Exception as exc:  # 释放视频资源后再向上报告
        failed = exc
    finally:
        capture.release()
        for writer in writers.values():
            writer.release()
        if tiled_writer is not None:
            tiled_writer.release()
        if failed is None:
            frame_handle.write("\n]\n")
        frame_handle.close()

    if failed is not None:
        print(f"[FAIL] 比较过程中断: {failed}", file=sys.stderr)
        print(f"[INFO] 未完成的临时逐帧文件保留在 {frames_tmp}", file=sys.stderr)
        return 1
    if processed_index == 0:
        print("[STOP] 没有读到可处理的视频帧", file=sys.stderr)
        return 2
    os.replace(frames_tmp, frames_path)
    prediction_frames_sha256 = sha256_file(frames_path)

    contact_path: Path | None = None
    if args.contact_frames > 0:
        sheet = build_contact_sheet(
            contact_rows, list(models), args.tile_width
        )
        if sheet is not None:
            contact_path = out_dir / "comparison_contact.jpg"
            write_image(contact_path, sheet)

    sample_period_seconds = args.stride / source_fps
    summary = {
        "schema_version": 1,
        "evaluation_kind": "prediction_stability_without_ground_truth",
        "ground_truth_used": False,
        "accuracy_claim_allowed": False,
        "disclaimer": NO_GT_DISCLAIMER,
        "settings": {
            "conf": args.conf,
            "iou": args.iou,
            "imgsz": args.imgsz,
            "device": args.device,
            "stride": args.stride,
            "max_processed_frames": args.max_frames,
            "codec": args.codec,
            "warmup_frame_excluded_from_timing": True,
            "same_decoded_frame_passed_to_every_model": True,
        },
        "model_order": list(models),
        "video": {
            "path": str(video_path),
            "sha256": source_video_sha256,
            "width": source_width,
            "height": source_height,
            "source_fps": source_fps,
            "source_frame_count_reported": source_frame_count,
            "sampled_frames_processed": processed_index,
            "output_fps": output_fps,
        },
        "metric_definitions": {
            "coverage": "fraction of sampled frames with >=1 prediction of that class",
            "gap_run_seconds": (
                "sampled-frame run length multiplied by stride/source_fps; approximate"
            ),
            "prediction_set_transition": (
                "adjacent sampled frames have different sets of predicted class ids"
            ),
            "dominant_class_transition": (
                "highest-confidence class (or none) changes between adjacent samples"
            ),
            "flicker": (
                "one-sample presence spike (off-on-off) or one-sample absence "
                "dropout (on-off-on); descriptive, not an error rate"
            ),
        },
        "models": {},
        "artifacts": {
            "per_frame_json": str(frames_path),
            "per_frame_json_sha256": prediction_frames_sha256,
            "tiled_video": (
                str(tiled_path) if not args.no_tiled_video else None
            ),
            "contact_sheet": str(contact_path) if contact_path else None,
        },
    }
    for label, info in models.items():
        summary["models"][label] = {
            "path": str(info["path"]),
            "sha256": info["sha256"],
            "schema": info["schema"],
            "class_count": len(info["names"]),
            "class_names": info["names"],
            "annotated_video": str(model_video_paths[label]),
            "prediction_stability": summarize_prediction_frames(
                metric_frames[label],
                info["names"],
                sample_period_seconds,
            ),
        }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[OUT] 逐帧预测: {frames_path}")
    print(f"[OUT] 汇总: {summary_path}")
    for label, path in model_video_paths.items():
        print(f"[OUT] {label} 画框视频: {path}")
    if not args.no_tiled_video:
        print(f"[OUT] 拼接视频: {tiled_path}")
    if contact_path:
        print(f"[OUT] 接触表: {contact_path}")
    print(f"[DONE] 共处理 {processed_index} 个抽样帧。无 GT，不作准确率结论。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
