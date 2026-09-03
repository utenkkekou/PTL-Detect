"""pytest 公共 fixtures。"""

from __future__ import annotations

from pathlib import Path

import pytest

from ptl.constants import EXPECTED_NAMES


@pytest.fixture
def project_root() -> Path:
    """项目根目录。"""
    return Path(__file__).parent.parent


@pytest.fixture
def num_classes() -> int:
    """当前检测类别数。"""
    return len(EXPECTED_NAMES)


@pytest.fixture
def sample_yolo_labels(tmp_path: Path) -> Path:
    """创建临时 YOLO 标签文件，返回标签路径。

    内容:
    - 1 个红灯框 (cls=0)
    - 1 个绿灯框 (cls=1)
    - 1 个车行灯框 (cls=3)
    """
    label_file = tmp_path / "test_frame.txt"
    label_file.write_text(
        "0 0.500 0.300 0.050 0.080\n"  # red: 中心偏上，小目标
        "1 0.700 0.500 0.060 0.100\n"  # green: 右侧中部
        "3 0.200 0.600 0.080 0.120\n",  # vehicle: 左侧偏下
        encoding="utf-8",
    )
    return label_file


@pytest.fixture
def sample_data_yaml(tmp_path: Path) -> Path:
    """创建临时 YOLO data.yaml 配置文件。"""
    import yaml

    config = {
        "path": str(tmp_path),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(EXPECTED_NAMES),
        "names": EXPECTED_NAMES,
    }
    yaml_file = tmp_path / "data.yaml"
    yaml_file.write_text(
        yaml.dump(config, default_flow_style=False),
        encoding="utf-8",
    )
    return yaml_file
