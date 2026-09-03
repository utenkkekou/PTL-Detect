"""安全评估核心逻辑。

头号指标: P(pred=green | gt=red) == 0
误报一次绿灯的后果和漏检完全不对等。

本模块提供与 CLI 无关的安全评估计算，可被 eval_safety.py 脚本和
测试代码复用。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ptl.constants import (
    CLS_ID,
)
from ptl.io_utils import box_iou, size_bucket

# ---------------------------------------------------------------------------
# 类型别名
# ---------------------------------------------------------------------------
# (cls_id, x1, y1, x2, y2)
Box = tuple[int, float, float, float, float]
# (cls_id, conf, x1, y1, x2, y2)
ScoredBox = tuple[int, float, float, float, float, float]


def match_predictions(
    gt_boxes: list[Box],
    pred_boxes: list[ScoredBox],
    match_iou: float,
    image_width: int,
    image_height: int,
) -> tuple[dict[int | str, dict[int | str, int]], dict[str, int], dict[str, int]]:
    """将预测框与 GT 框按 IoU 贪心匹配（降序 conf，一对一）。

    Args:
        gt_boxes: 真值框 [(cls_id, x1, y1, x2, y2), ...]
        pred_boxes: 预测框 [(cls_id, conf, x1, y1, x2, y2), ...]
        match_iou: 最低 IoU 阈值
        image_width: 图片宽度（像素）
        image_height: 图片高度（像素）

    Returns:
        (confusion, size_total, size_hit) 三元组:
        - confusion: gt_cls -> {pred_cls -> count}，"bg"=多检，"miss"=漏检
        - size_total: 按尺寸分桶的 GT 总数
        - size_hit: 按尺寸分桶的匹配数
    """
    confusion: dict[int | str, dict[int | str, int]] = defaultdict(lambda: defaultdict(int))
    size_total: dict[str, int] = defaultdict(int)
    size_hit: dict[str, int] = defaultdict(int)

    # 按 conf 降序排列预测
    sorted_preds = sorted(pred_boxes, key=lambda p: -p[1])
    gt_used = [False] * len(gt_boxes)

    for pc, _conf, px1, py1, px2, py2 in sorted_preds:
        best_iou, best_idx = match_iou, -1
        for gi, (_gc, gx1, gy1, gx2, gy2) in enumerate(gt_boxes):
            if gt_used[gi]:
                continue
            v = box_iou((px1, py1, px2, py2), (gx1, gy1, gx2, gy2))
            if v >= best_iou:
                best_iou, best_idx = v, gi
        if best_idx >= 0:
            gt_used[best_idx] = True
            confusion[gt_boxes[best_idx][0]][pc] += 1
        else:
            confusion["bg"][pc] += 1  # 多检

    for gi, used in enumerate(gt_used):
        gc = gt_boxes[gi][0]
        bucket = size_bucket(gt_boxes[gi][1:], image_width, image_height)
        size_total[bucket] += 1
        if used:
            size_hit[bucket] += 1
        else:
            confusion[gc]["miss"] += 1

    return confusion, size_total, size_hit


def compute_danger_metrics(
    confusion: dict[int | str, dict[int | str, int]],
) -> dict[str, Any]:
    """从混淆矩阵中提取安全关键指标。

    Returns:
        包含以下字段的字典:
        - danger_count: 真实红灯被判成绿灯的框数
        - red_total: 真实红灯总框数
        - danger_rate: 危险误报率 P(pred=green | gt=red)
        - vehicle_as_green: 非行人信号灯被判成绿灯的框数
        - vehicle_total: 非行人信号灯总框数
        - passed: 安全硬门是否通过（danger_count == 0 且 vehicle_as_green == 0）
    """
    gt_red = confusion.get(CLS_ID["pedestrian_signal_red"], {})
    red_total = sum(gt_red.values())
    danger_count = gt_red.get(CLS_ID["pedestrian_signal_green"], 0)
    danger_rate = danger_count / red_total if red_total else 0.0

    gt_vehicle = confusion.get(CLS_ID["vehicle_signal"], {})
    vehicle_total = sum(gt_vehicle.values())
    vehicle_as_green = gt_vehicle.get(CLS_ID["pedestrian_signal_green"], 0)

    return {
        "danger_count": danger_count,
        "red_total": red_total,
        "danger_rate": danger_rate,
        "vehicle_as_green": vehicle_as_green,
        "vehicle_total": vehicle_total,
        "passed": danger_count == 0 and vehicle_as_green == 0,
    }
