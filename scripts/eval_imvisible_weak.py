#!/usr/bin/env python3
"""ImVisible 上海外部域的图级弱标签评估。

ImVisible 官方 CSV 只提供整图主行人灯状态和斑马线中线端点，完全没有
信号灯检测框。因此本脚本只报告图级诊断，绝不计算或声称 bbox AP、IoU、
precision/recall：

* mode=None：统计出现任一人行类 0/1/2 的图像比例；class 3
  ``vehicle_signal`` 被明确忽略，因为 None 图中仍可能有车行灯。
* Red / Green / Countdown Green：分别统计期望颜色是否至少命中一次，以及
  是否至少命中任一人行类。
* Countdown Blank：契约无法安全映射，保留数量但不评分。

所有 CSV 行先按原图 SHA256 去重；同一内容若有冲突 mode 会 fail closed。

示例：
  python scripts/eval_imvisible_weak.py \
      --model runs/y11n_clean4/weights/best.pt \
      --images-dir data/raw/imvisible/images \
      --testing-csv data/raw/imvisible/Annotations/testing_file.csv \
      --conf 0.25 --imgsz 960 \
      --out runs/y11n_clean4/imvisible_weak.md
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PEDESTRIAN_CLASSES = frozenset({0, 1, 2})
EXPECTED_NAMES = [
    "pedestrian_signal_red",
    "pedestrian_signal_green",
    "pedestrian_signal_unknown",
    "vehicle_signal",
]
MODE_NAMES = {
    "0": "red",
    "1": "green",
    "2": "countdown_green",
    "3": "countdown_blank",
    "4": "none",
}
POSITIVE_MODES = {
    "0": ("red", 0),
    "1": ("green", 1),
    "2": ("countdown_green", 1),
}


def normalized_names(raw: Any) -> list[str]:
    if isinstance(raw, dict):
        return [raw[key] for key in sorted(raw, key=lambda value: int(value))]
    return list(raw or [])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deduplicate_by_sha(
    records: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """按 SHA 保留首行；相同内容的 mode 冲突时拒绝继续。

    该函数无 I/O，便于单测去重和 fail-closed 语义。
    """
    unique: list[dict[str, Any]] = []
    seen: dict[str, dict[str, Any]] = {}
    duplicate_groups: dict[str, dict[str, Any]] = {}

    for source in records:
        record = dict(source)
        sha = str(record.get("sha256", "")).strip().lower()
        mode = str(record.get("mode", "")).strip()
        filename = str(record.get("file", "")).strip()
        if not sha:
            raise ValueError(f"记录缺少 sha256: {filename or record!r}")
        if mode not in MODE_NAMES:
            raise ValueError(f"非法 ImVisible mode={mode!r}: {filename}")

        kept = seen.get(sha)
        if kept is None:
            seen[sha] = record
            unique.append(record)
            continue
        if str(kept["mode"]) != mode:
            raise ValueError(
                "相同图像 SHA 的官方 mode 冲突: "
                f"{kept.get('file')}={kept.get('mode')} / {filename}={mode}"
            )
        group = duplicate_groups.setdefault(
            sha,
            {
                "sha256": sha,
                "kept": str(kept.get("file", "")),
                "removed": [],
            },
        )
        group["removed"].append(filename)

    return unique, list(duplicate_groups.values())


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def summarize_predictions(
    records: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """汇总已过阈值的图级类别集合，不使用任何框坐标。"""
    positive = {
        name: {
            "mode": mode,
            "expected_class": expected_class,
            "total_images": 0,
            "same_color_hit_images": 0,
            "same_color_hit_rate": None,
            "any_pedestrian_hit_images": 0,
            "any_pedestrian_hit_rate": None,
            "same_color_miss_files": [],
            "no_pedestrian_hit_files": [],
        }
        for mode, (name, expected_class) in POSITIVE_MODES.items()
    }
    none = {
        "mode": "4",
        "total_images": 0,
        "pedestrian_false_positive_images": 0,
        "pedestrian_false_positive_rate": None,
        "vehicle_signal_image_hits_ignored": 0,
        "no_pedestrian_hit_images": 0,
        "false_positive_files": [],
    }
    countdown_blank = {
        "mode": "3",
        "total_images": 0,
        "scored": False,
        "reason": "官方 Countdown Blank 无法安全映射到当前逐灯头四类契约",
    }
    class_image_hits: Counter[int] = Counter()
    total_images = 0

    for record in records:
        total_images += 1
        filename = str(record.get("file", ""))
        mode = str(record.get("mode", "")).strip()
        if mode not in MODE_NAMES:
            raise ValueError(f"非法 ImVisible mode={mode!r}: {filename}")

        predicted: set[int] = set()
        for raw_class in record.get("predicted_classes", []):
            class_id = int(raw_class)
            if class_id not in range(4):
                raise ValueError(
                    f"模型输出非四类 class id={class_id}: {filename}"
                )
            predicted.add(class_id)
        class_image_hits.update(predicted)
        has_pedestrian = bool(predicted & PEDESTRIAN_CLASSES)

        if mode in POSITIVE_MODES:
            name, expected_class = POSITIVE_MODES[mode]
            group = positive[name]
            group["total_images"] += 1
            if expected_class in predicted:
                group["same_color_hit_images"] += 1
            else:
                group["same_color_miss_files"].append(filename)
            if has_pedestrian:
                group["any_pedestrian_hit_images"] += 1
            else:
                group["no_pedestrian_hit_files"].append(filename)
        elif mode == "4":
            none["total_images"] += 1
            if has_pedestrian:
                none["pedestrian_false_positive_images"] += 1
                none["false_positive_files"].append(filename)
            else:
                none["no_pedestrian_hit_images"] += 1
            if 3 in predicted:
                none["vehicle_signal_image_hits_ignored"] += 1
        else:
            countdown_blank["total_images"] += 1

    for group in positive.values():
        total = group["total_images"]
        group["same_color_hit_rate"] = _rate(
            group["same_color_hit_images"], total
        )
        group["any_pedestrian_hit_rate"] = _rate(
            group["any_pedestrian_hit_images"], total
        )
    none["pedestrian_false_positive_rate"] = _rate(
        none["pedestrian_false_positive_images"], none["total_images"]
    )

    return {
        "total_unique_images": total_images,
        "positive_modes": positive,
        "none": none,
        "countdown_blank_unscored": countdown_blank,
        "class_image_hits": {
            str(class_id): class_image_hits.get(class_id, 0)
            for class_id in range(4)
        },
    }


def _image_index(images_dir: Path) -> dict[str, Path]:
    if not images_dir.is_dir():
        raise ValueError(f"图片目录不存在: {images_dir}")
    index: dict[str, Path] = {}
    for path in sorted(images_dir.rglob("*"), key=lambda item: str(item).casefold()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        key = path.stem.casefold()
        previous = index.get(key)
        if previous is not None:
            raise ValueError(
                f"图片 stem 大小写冲突，无法唯一匹配 CSV: {previous} / {path}"
            )
        index[key] = path
    if not index:
        raise ValueError(f"图片目录没有支持的图像: {images_dir}")
    return index


def load_official_test_records(
    images_dir: Path,
    testing_csv: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """读取官方测试 CSV、解析原图并按内容 SHA 去重。"""
    if not testing_csv.is_file():
        raise ValueError(f"测试 CSV 不存在: {testing_csv}")
    image_index = _image_index(images_dir)

    records: list[dict[str, Any]] = []
    with testing_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"file", "mode"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"CSV 缺少列 {sorted(required)}: {reader.fieldnames}"
            )
        for line_number, row in enumerate(reader, start=2):
            filename = str(row.get("file", "")).strip()
            mode = str(row.get("mode", "")).strip()
            if not filename:
                raise ValueError(f"CSV 第 {line_number} 行 file 为空")
            if mode not in MODE_NAMES:
                raise ValueError(
                    f"CSV 第 {line_number} 行 mode={mode!r} 非法"
                )
            path = image_index.get(Path(filename).stem.casefold())
            if path is None:
                raise ValueError(
                    f"CSV 第 {line_number} 行找不到原图: {filename}"
                )
            records.append(
                {
                    "file": filename,
                    "mode": mode,
                    "block": str(row.get("block", "")).strip(),
                    "path": path,
                    "sha256": sha256_file(path),
                }
            )

    unique, duplicate_groups = deduplicate_by_sha(records)
    integrity = {
        "csv_rows": len(records),
        "unique_sha256_images": len(unique),
        "duplicates_removed": len(records) - len(unique),
        "duplicate_groups": duplicate_groups,
        "mode_counts_before_dedup": dict(
            sorted(Counter(record["mode"] for record in records).items())
        ),
        "mode_counts_after_dedup": dict(
            sorted(Counter(record["mode"] for record in unique).items())
        ),
    }
    return unique, integrity


def predict_image_classes(
    model: Any,
    records: Iterable[dict[str, Any]],
    conf: float,
    imgsz: int,
) -> list[dict[str, Any]]:
    """运行模型并只保留每图的类别存在性；框坐标有意丢弃。"""
    import numpy as np
    from PIL import Image, ImageFile

    materialized = list(records)
    predictions: list[dict[str, Any]] = []
    for index, record in enumerate(materialized, start=1):
        path = Path(record["path"])
        decode_warning = None
        try:
            with Image.open(path) as image:
                rgb = np.asarray(image.convert("RGB"))
        except OSError as exc:
            if "truncated" not in str(exc).lower():
                raise
            # 官方测试集有一张 JPEG 只缺尾部 25 bytes。Pillow 可在显式
            # 开启容错后稳定解码；记录到报告，不能静默跳过或缩小分母。
            previous = ImageFile.LOAD_TRUNCATED_IMAGES
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            try:
                with Image.open(path) as image:
                    rgb = np.asarray(image.convert("RGB"))
            finally:
                ImageFile.LOAD_TRUNCATED_IMAGES = previous
            decode_warning = f"truncated JPEG decoded: {exc}"
        # Ultralytics 对 numpy 输入按 OpenCV BGR 解释；Pillow 解码为 RGB。
        source = np.ascontiguousarray(rgb[:, :, ::-1])
        result = model.predict(
            source,
            conf=conf,
            imgsz=imgsz,
            verbose=False,
        )[0]
        boxes = result.boxes
        class_values = (
            boxes.cls.detach().cpu().tolist()
            if boxes is not None and boxes.cls is not None
            else []
        )
        confidence_values = (
            boxes.conf.detach().cpu().tolist()
            if boxes is not None and boxes.conf is not None
            else []
        )
        class_counts: Counter[int] = Counter()
        max_confidence: dict[int, float] = {}
        for raw_class, raw_confidence in zip(
            class_values, confidence_values, strict=True
        ):
            class_id = int(raw_class)
            if class_id not in range(4):
                raise ValueError(
                    f"模型输出非四类 class id={class_id}: {record['file']}"
                )
            confidence = float(raw_confidence)
            class_counts[class_id] += 1
            max_confidence[class_id] = max(
                max_confidence.get(class_id, 0.0), confidence
            )
        predictions.append(
            {
                "file": record["file"],
                "mode": record["mode"],
                "predicted_classes": sorted(class_counts),
                "prediction_box_counts": {
                    str(class_id): class_counts[class_id]
                    for class_id in sorted(class_counts)
                },
                "max_confidence_by_class": {
                    str(class_id): max_confidence[class_id]
                    for class_id in sorted(max_confidence)
                },
                "decode_warning": decode_warning,
            }
        )
        if index == 1 or index % 50 == 0 or index == len(materialized):
            print(f"[PRED] {index}/{len(materialized)}")
    return predictions


def _percent(rate: float | None) -> str:
    return "—" if rate is None else f"{100 * rate:.2f}%"


def render_markdown(payload: dict[str, Any]) -> str:
    dataset = payload["dataset"]
    metrics = payload["metrics"]
    settings = payload["settings"]
    positive = metrics["positive_modes"]
    none = metrics["none"]

    lines = [
        "# ImVisible 上海外部域弱标签评估",
        "",
        "> **这不是检测框评估：不读取信号灯框，不计算 bbox AP、IoU、"
        "检测 precision 或检测 recall。** 官方 CSV 只有整图主灯状态和"
        "斑马线中线端点。",
        "",
        "## 运行信息",
        "",
        f"- 模型：`{settings['model']}`",
        f"- 阈值：conf={settings['conf']}，imgsz={settings['imgsz']}",
        f"- 官方 CSV 行数：{dataset['csv_rows']}",
        (
            "- SHA256 去重后图像："
            f"{dataset['unique_sha256_images']}（移除完全重复 "
            f"{dataset['duplicates_removed']}）"
        ),
        (
            "- 容错解码图像："
            f"{len(dataset.get('decode_warnings', []))}"
        ),
        "- 第 4 类 `vehicle_signal` 在 None 误报统计中明确忽略。",
        "",
        "## None：人行类图级误报",
        "",
        f"- 唯一图像：{none['total_images']}",
        (
            "- 出现任一人行类 0/1/2："
            f"**{none['pedestrian_false_positive_images']} / "
            f"{none['total_images']} "
            f"({_percent(none['pedestrian_false_positive_rate'])})**"
        ),
        (
            "- 出现 class 3（不计误报）："
            f"{none['vehicle_signal_image_hits_ignored']}"
        ),
        "",
        "## 有灯模式：图级弱命中",
        "",
        "| 官方 mode | 期望类别 | 唯一图像 | 同色至少一次 | 同色命中率 | "
        "任一人行类至少一次 | 任一人行命中率 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in (
        ("red", "Red"),
        ("green", "Green"),
        ("countdown_green", "Countdown Green"),
    ):
        group = positive[key]
        lines.append(
            f"| {label} | {group['expected_class']} | "
            f"{group['total_images']} | {group['same_color_hit_images']} | "
            f"{_percent(group['same_color_hit_rate'])} | "
            f"{group['any_pedestrian_hit_images']} | "
            f"{_percent(group['any_pedestrian_hit_rate'])} |"
        )

    unscored = metrics["countdown_blank_unscored"]
    lines += [
        "",
        "## 不评分层",
        "",
        (
            f"- Countdown Blank：{unscored['total_images']} 张。"
            f"{unscored['reason']}，只保留覆盖数量。"
        ),
        "",
        "## 解释限制",
        "",
        "- “同色命中”只表示整张图至少出现一次该类别；不能证明定位正确，"
        "也不能证明命中了官方选定的主灯。",
        "- 图中可能同时存在多个方向、不同颜色的人行灯与车行灯；因此这些"
        "数字只能用于跨域回归和失败样本筛查。",
        "- SHA256 只移除字节完全相同的图；连拍近邻仍不是独立样本。",
        "- 若要报告 bbox AP/IoU，必须另建经过四类逐框人工复核的上海 GT。",
        "",
        "## None 人行类误报文件",
        "",
    ]
    false_positive_files = none["false_positive_files"]
    if false_positive_files:
        lines.extend(f"- `{filename}`" for filename in false_positive_files)
    else:
        lines.append("- 无")
    lines.append("")
    return "\n".join(lines)


def _output_paths(raw_out: str) -> tuple[Path, Path]:
    out = Path(raw_out)
    if out.suffix.lower() == ".md":
        markdown = out
    elif out.suffix:
        raise ValueError("--out 必须是 .md 文件或无扩展名目录")
    else:
        markdown = out / "imvisible_weak_report.md"
    return markdown, markdown.with_suffix(".json")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ImVisible 上海外部域图级弱标签评估（不计算 bbox AP）"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--images-dir",
        default="data/raw/imvisible/images",
        help="ImVisible 原图根目录，可递归包含 Part1/Part2",
    )
    parser.add_argument(
        "--testing-csv",
        "--testing-file",
        dest="testing_csv",
        default="data/raw/imvisible/Annotations/testing_file.csv",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument(
        "--out",
        required=True,
        help="Markdown 文件路径；也可给目录，JSON 写到同名 .json",
    )
    args = parser.parse_args()

    if not 0 <= args.conf <= 1:
        print("[STOP] --conf 必须在 [0, 1]。")
        return 2
    if args.imgsz <= 0:
        print("[STOP] --imgsz 必须为正整数。")
        return 2

    try:
        records, integrity = load_official_test_records(
            Path(args.images_dir), Path(args.testing_csv)
        )
        markdown_path, json_path = _output_paths(args.out)
    except (OSError, ValueError, csv.Error) as exc:
        print(f"[STOP] {exc}")
        return 2

    print(
        "[WEAK] 仅做图级弱标签诊断；不读取灯框，不计算 bbox AP/IoU。"
    )
    print(
        f"[DATA] CSV={integrity['csv_rows']}，SHA唯一="
        f"{integrity['unique_sha256_images']}，"
        f"去重={integrity['duplicates_removed']}"
    )

    from ultralytics import YOLO

    model = YOLO(args.model)
    model_names = normalized_names(model.names)
    if model_names != EXPECTED_NAMES:
        print("[STOP] 模型不是当前四类契约。")
        print("  实际:", model_names)
        print("  期望:", EXPECTED_NAMES)
        return 2

    try:
        predictions = predict_image_classes(
            model, records, args.conf, args.imgsz
        )
        integrity["decode_warnings"] = [
            {
                "file": prediction["file"],
                "warning": prediction["decode_warning"],
            }
            for prediction in predictions
            if prediction.get("decode_warning")
        ]
        metrics = summarize_predictions(predictions)
    except (RuntimeError, TypeError, ValueError) as exc:
        print(f"[STOP] 预测或汇总失败: {exc}")
        return 2

    payload = {
        "schema_version": 1,
        "evaluation_kind": "image_level_weak_label",
        "bbox_metrics_computed": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "settings": {
            "model": str(args.model),
            "images_dir": str(args.images_dir),
            "testing_csv": str(args.testing_csv),
            "conf": args.conf,
            "imgsz": args.imgsz,
            "class_names": EXPECTED_NAMES,
        },
        "dataset": integrity,
        "metrics": metrics,
    }
    report = render_markdown(payload)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(report + "\n", encoding="utf-8")
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(report)
    print(f"[OUT] Markdown: {markdown_path}")
    print(f"[OUT] JSON: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
