"""通用 I/O 工具函数。

提供被多个脚本重复使用的文件操作、YOLO 标签解析、几何计算等功能。
所有函数都是纯工具，不依赖 ultralytics 等重型库。
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import Any

from ptl.constants import EXPECTED_NAMES

# ---------------------------------------------------------------------------
# 文件哈希
# ---------------------------------------------------------------------------


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """计算文件的 SHA256 摘要（分块读取，适合大文件）。

    Returns:
        小写十六进制摘要字符串。
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# YOLO 标签 I/O
# ---------------------------------------------------------------------------


def read_yolo_labels(
    label_path: Path,
    width: int,
    height: int,
) -> list[tuple[int, float, float, float, float]]:
    """读取 YOLO 格式标签文件，返回 [(cls_id, x1, y1, x2, y2), ...]。

    坐标从归一化 (cx, cy, w, h) 转换为像素绝对坐标 (x1, y1, x2, y2)。
    如果标签文件不存在，返回空列表。
    """
    out: list[tuple[int, float, float, float, float]] = []
    if not label_path.exists():
        return out
    for line in label_path.read_text(encoding="utf-8").strip().splitlines():
        if not line.strip():
            continue
        parts = line.split()[:5]
        cls_id = int(parts[0])
        cx, cy, w, h = map(float, parts[1:5])
        out.append((
            cls_id,
            (cx - w / 2) * width,
            (cy - h / 2) * height,
            (cx + w / 2) * width,
            (cy + h / 2) * height,
        ))
    return out


def validate_yolo_box(
    item_id: str,
    index: int,
    box: dict[str, Any],
    num_classes: int = len(EXPECTED_NAMES),
) -> tuple[int, float, float, float, float]:
    """校验一个标注框的合法性。

    检查:
    - cls 是整数且在 [0, num_classes) 范围内
    - cx/cy/w/h 是有限浮点数
    - w > 0, h > 0
    - 框不超出 [0, 1] 范围（允许 epsilon 容差）

    Returns:
        (cls_id, cx, cy, w, h) 归一化坐标元组。

    Raises:
        ValueError: 任何校验失败时。
    """
    try:
        raw_cls = box["cls"]
        if isinstance(raw_cls, bool) or not isinstance(raw_cls, int):
            raise TypeError("class id must be an integer, not a coercible value")
        cls = raw_cls
        cx, cy, w, h = (
            float(box["cx"]),
            float(box["cy"]),
            float(box["w"]),
            float(box["h"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{item_id}: invalid box #{index}: {box!r}") from exc

    if cls not in range(num_classes):
        raise ValueError(f"{item_id}: class {cls} is outside 0..{num_classes - 1}")
    values = (cx, cy, w, h)
    if not all(v == v and abs(v) != float("inf") for v in values):
        raise ValueError(f"{item_id}: non-finite box #{index}: {box!r}")
    epsilon = 1e-6
    if (
        w <= 0
        or h <= 0
        or cx - w / 2 < -epsilon
        or cy - h / 2 < -epsilon
        or cx + w / 2 > 1 + epsilon
        or cy + h / 2 > 1 + epsilon
    ):
        raise ValueError(f"{item_id}: out-of-bounds box #{index}: {box!r}")
    return cls, cx, cy, w, h


# ---------------------------------------------------------------------------
# 几何计算
# ---------------------------------------------------------------------------


def box_iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """计算两个 (x1, y1, x2, y2) 框的 IoU（Intersection over Union）。"""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def size_bucket(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
) -> str:
    """按面积占比将检测框分为 small/medium/large 三档。

    阈值:
    - small:  < 0.2% (远处小目标)
    - medium: < 1%
    - large:  >= 1%
    """
    frac = ((box[2] - box[0]) * (box[3] - box[1])) / (width * height)
    if frac < 0.002:
        return "small"
    if frac < 0.01:
        return "medium"
    return "large"


# ---------------------------------------------------------------------------
# 数据集辅助
# ---------------------------------------------------------------------------


def copy_or_link(source: Path, destination: Path, hardlink: bool = False) -> None:
    """复制文件或使用硬链接，自动创建目标目录。"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if hardlink:
        os.link(source, destination)
    else:
        shutil.copy2(source, destination)


def ordered_names_from_yaml(data_yaml: Path) -> list[str]:
    """从 YOLO data.yaml 读取有序类名列表。

    处理 names 为 list 或 dict（如 {0: 'red', 1: 'green', ...}）两种格式。
    """
    import yaml

    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = config.get("names", [])
    if isinstance(names, dict):
        names = [names[key] for key in sorted(names, key=lambda v: int(v))]
    return list(names)
