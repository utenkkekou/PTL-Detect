"""测试 ptl.safety 中的安全评估逻辑。"""

from __future__ import annotations

from ptl.safety import compute_danger_metrics, match_predictions

# ---------------------------------------------------------------------------
# match_predictions
# ---------------------------------------------------------------------------


def test_match_predictions_perfect():
    """完美预测 — 所有 GT 框正确匹配，无误报。"""
    gt = [(0, 10, 10, 20, 20), (1, 50, 50, 70, 70)]
    preds = [
        (0, 0.95, 10, 10, 20, 20),  # 匹配第一个 GT
        (1, 0.90, 50, 50, 70, 70),  # 匹配第二个 GT
    ]
    confusion, size_total, size_hit = match_predictions(gt, preds, 0.5, 1000, 1000)
    assert confusion[0][0] == 1  # red -> red
    assert confusion[1][1] == 1  # green -> green
    assert confusion["bg"].get(0, 0) == 0  # 无多检


def test_match_predictions_missed():
    """漏检 — GT 框无匹配预测。"""
    gt = [(0, 10, 10, 20, 20)]
    preds: list = []
    confusion, _, _ = match_predictions(gt, preds, 0.5, 1000, 1000)
    assert confusion[0]["miss"] == 1  # red 漏检


def test_match_predictions_false_positive():
    """多检 — 预测框无匹配 GT。"""
    gt: list = []
    preds = [(1, 0.90, 10, 10, 20, 20)]
    confusion, _, _ = match_predictions(gt, preds, 0.5, 1000, 1000)
    assert confusion["bg"][1] == 1  # 背景被检成 green


def test_match_predictions_dangerous():
    """危险误报 — 红灯被预测为绿灯。"""
    gt = [(0, 100, 100, 200, 200)]
    preds = [(1, 0.90, 100, 100, 200, 200)]  # 预测为 green
    confusion, _, _ = match_predictions(gt, preds, 0.5, 1000, 1000)
    assert confusion[0][1] == 1  # red -> green（危险！）


def test_match_predictions_iou_too_low():
    """IoU 低于阈值 — 不算匹配。"""
    gt = [(0, 10, 10, 20, 20)]
    preds = [(0, 0.95, 50, 50, 60, 60)]  # 位置完全不同
    confusion, _, _ = match_predictions(gt, preds, 0.5, 1000, 1000)
    assert confusion[0]["miss"] == 1  # GT 漏检
    assert confusion["bg"][0] == 1  # 预测多检


# ---------------------------------------------------------------------------
# compute_danger_metrics
# ---------------------------------------------------------------------------


def test_compute_danger_metrics_safe():
    """安全场景 — 无危险误报。"""
    confusion = {
        0: {0: 10, 2: 2, "miss": 1},  # red -> red/unknown/miss, 无 green
        1: {1: 20},
        3: {3: 5},
    }
    metrics = compute_danger_metrics(confusion)
    assert metrics["danger_count"] == 0
    assert metrics["danger_rate"] == 0.0
    assert metrics["vehicle_as_green"] == 0
    assert metrics["passed"] is True


def test_compute_danger_metrics_dangerous():
    """危险场景 — 红灯被判成绿灯。"""
    confusion = {
        0: {0: 8, 1: 2},  # 2 个红灯 -> 绿灯！
        1: {1: 20},
    }
    metrics = compute_danger_metrics(confusion)
    assert metrics["danger_count"] == 2
    assert metrics["red_total"] == 10
    assert metrics["danger_rate"] == 0.2
    assert metrics["passed"] is False


def test_compute_danger_metrics_vehicle_leak():
    """车行灯泄漏 — 非行人灯被判成绿灯。"""
    confusion = {
        0: {0: 10},
        1: {1: 20},
        3: {1: 3, 3: 7},  # 3 个车行灯 -> 绿灯
    }
    metrics = compute_danger_metrics(confusion)
    assert metrics["danger_count"] == 0
    assert metrics["vehicle_as_green"] == 3
    assert metrics["passed"] is False


def test_compute_danger_metrics_empty():
    """空混淆矩阵 — 无数据。"""
    confusion = {}
    metrics = compute_danger_metrics(confusion)
    assert metrics["danger_count"] == 0
    assert metrics["red_total"] == 0
    assert metrics["danger_rate"] == 0.0
    assert metrics["passed"] is True
