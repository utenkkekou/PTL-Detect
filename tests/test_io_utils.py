"""测试 ptl.io_utils 中的工具函数。"""

from __future__ import annotations

from pathlib import Path

import pytest

from ptl.constants import EXPECTED_NAMES
from ptl.io_utils import (
    box_iou,
    ordered_names_from_yaml,
    read_yolo_labels,
    sha256_file,
    size_bucket,
    validate_yolo_box,
)

# ---------------------------------------------------------------------------
# sha256_file
# ---------------------------------------------------------------------------


def test_sha256_file(tmp_path: Path):
    """空文件摘要应匹配标准值。"""
    f = tmp_path / "empty.bin"
    f.write_bytes(b"")
    # SHA256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
    assert sha256_file(f).startswith("e3b0c44298")


def test_sha256_file_content(tmp_path: Path):
    """相同内容的文件应有相同摘要。"""
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_bytes(b"hello world")
    f2.write_bytes(b"hello world")
    assert sha256_file(f1) == sha256_file(f2)


def test_sha256_file_different(tmp_path: Path):
    """不同内容的文件摘要不同。"""
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_bytes(b"hello")
    f2.write_bytes(b"world")
    assert sha256_file(f1) != sha256_file(f2)


# ---------------------------------------------------------------------------
# read_yolo_labels
# ---------------------------------------------------------------------------


def test_read_yolo_labels_basic(sample_yolo_labels: Path):
    """正确解析 YOLO 标签文件。"""
    labels = read_yolo_labels(sample_yolo_labels, width=1000, height=1000)
    assert len(labels) == 3

    # 第一个框: cls=0 (red), cx=0.5, cy=0.3, w=0.05, h=0.08
    cls, x1, y1, x2, y2 = labels[0]
    assert cls == 0
    assert x1 == pytest.approx((0.5 - 0.025) * 1000)
    assert y1 == pytest.approx((0.3 - 0.04) * 1000)
    assert x2 == pytest.approx((0.5 + 0.025) * 1000)
    assert y2 == pytest.approx((0.3 + 0.04) * 1000)


def test_read_yolo_labels_missing(tmp_path: Path):
    """不存在的文件返回空列表。"""
    labels = read_yolo_labels(tmp_path / "nonexistent.txt", 100, 100)
    assert labels == []


def test_read_yolo_labels_empty(tmp_path: Path):
    """空文件返回空列表。"""
    f = tmp_path / "empty.txt"
    f.write_text("", encoding="utf-8")
    labels = read_yolo_labels(f, 100, 100)
    assert labels == []


def test_read_yolo_labels_blank_lines(tmp_path: Path):
    """忽略空行。"""
    f = tmp_path / "blanks.txt"
    f.write_text("\n0 0.5 0.5 0.1 0.1\n\n", encoding="utf-8")
    labels = read_yolo_labels(f, 100, 100)
    assert len(labels) == 1


# ---------------------------------------------------------------------------
# box_iou
# ---------------------------------------------------------------------------


def test_box_iou_identical():
    """相同框的 IoU = 1。"""
    box = (0.0, 0.0, 10.0, 10.0)
    assert box_iou(box, box) == pytest.approx(1.0)


def test_box_iou_no_overlap():
    """无重叠框的 IoU = 0。"""
    assert box_iou((0, 0, 5, 5), (10, 10, 20, 20)) == 0.0


def test_box_iou_partial():
    """部分重叠的 IoU 在 (0, 1) 之间。"""
    v = box_iou((0, 0, 10, 10), (5, 5, 15, 15))
    assert 0.0 < v < 1.0
    # 交集 = 5*5=25, 并集 = 100+100-25=175
    assert v == pytest.approx(25 / 175)


def test_box_iou_contained():
    """包含关系的 IoU。"""
    # 小框完全在大框内: 交集=25, 并集=400
    v = box_iou((0, 0, 20, 20), (5, 5, 10, 10))
    assert v == pytest.approx(25 / 400)


# ---------------------------------------------------------------------------
# size_bucket
# ---------------------------------------------------------------------------


def test_size_bucket_small():
    """面积 < 0.2% 为 small。"""
    # 框面积 = 10*10 = 100, 图片面积 = 1000*1000 = 1M, 占比 0.01%
    assert size_bucket((0, 0, 10, 10), 1000, 1000) == "small"


def test_size_bucket_medium():
    """面积在 [0.2%, 1%) 为 medium。"""
    # 框面积 = 50*50 = 2500, 占比 0.25%
    assert size_bucket((0, 0, 50, 50), 1000, 1000) == "medium"


def test_size_bucket_large():
    """面积 >= 1% 为 large。"""
    # 框面积 = 200*200 = 40000, 占比 4%
    assert size_bucket((0, 0, 200, 200), 1000, 1000) == "large"


# ---------------------------------------------------------------------------
# validate_yolo_box
# ---------------------------------------------------------------------------


def test_validate_yolo_box_valid():
    """合法框返回正确元组。"""
    box = {"cls": 1, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1}
    result = validate_yolo_box("test", 0, box)
    assert result == (1, 0.5, 0.5, 0.1, 0.1)


def test_validate_yolo_box_invalid_class():
    """类别越界报错。"""
    box = {"cls": 99, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1}
    with pytest.raises(ValueError, match="outside"):
        validate_yolo_box("test", 0, box)


def test_validate_yolo_box_bool_class():
    """布尔类型的 cls 被拒绝。"""
    box = {"cls": True, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1}
    with pytest.raises(ValueError, match="invalid box"):
        validate_yolo_box("test", 0, box)


def test_validate_yolo_box_negative_size():
    """负尺寸报错。"""
    box = {"cls": 0, "cx": 0.5, "cy": 0.5, "w": -0.1, "h": 0.1}
    with pytest.raises(ValueError, match="out-of-bounds"):
        validate_yolo_box("test", 0, box)


def test_validate_yolo_box_missing_key():
    """缺失字段报错。"""
    box = {"cls": 0, "cx": 0.5}
    with pytest.raises(ValueError, match="invalid box"):
        validate_yolo_box("test", 0, box)


# ---------------------------------------------------------------------------
# ordered_names_from_yaml
# ---------------------------------------------------------------------------


def test_ordered_names_from_yaml_list(sample_data_yaml: Path):
    """从 list 格式的 data.yaml 读取类名。"""
    names = ordered_names_from_yaml(sample_data_yaml)
    assert names == EXPECTED_NAMES


def test_ordered_names_from_yaml_dict(tmp_path: Path):
    """从 dict 格式的 data.yaml 读取类名（按 key 排序）。"""
    import yaml

    config = {
        "path": str(tmp_path),
        "names": {0: "pedestrian_signal_red", 1: "pedestrian_signal_green",
                  2: "pedestrian_signal_unknown", 3: "vehicle_signal"},
    }
    yaml_file = tmp_path / "data.yaml"
    yaml_file.write_text(yaml.dump(config), encoding="utf-8")
    names = ordered_names_from_yaml(yaml_file)
    assert names == EXPECTED_NAMES
