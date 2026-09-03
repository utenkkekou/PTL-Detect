"""验证 ptl.constants 中常量的一致性。"""

from __future__ import annotations

from ptl.constants import (
    ATTESTATION_KEY,
    CLS_ID,
    CLS_NAME,
    DEFAULT_CONF,
    DEFAULT_IMGSZ,
    DEFAULT_MATCH_IOU,
    EXPECTED_NAMES,
    GREEN_CONF_THRESHOLD,
    LEGACY3_NAMES,
    RED_CONF_THRESHOLD,
)


def test_expected_names_count():
    """必须有 4 个类。"""
    assert len(EXPECTED_NAMES) == 4


def test_expected_names_order():
    """类名顺序固定（red=0, green=1, unknown=2, vehicle=3）。"""
    assert EXPECTED_NAMES[0] == "pedestrian_signal_red"
    assert EXPECTED_NAMES[1] == "pedestrian_signal_green"
    assert EXPECTED_NAMES[2] == "pedestrian_signal_unknown"
    assert EXPECTED_NAMES[3] == "vehicle_signal"


def test_cls_id_mapping():
    """CLS_ID 与 EXPECTED_NAMES 一致。"""
    for i, name in enumerate(EXPECTED_NAMES):
        assert CLS_ID[name] == i


def test_cls_name_mapping():
    """CLS_NAME 与 EXPECTED_NAMES 一致。"""
    for i, name in enumerate(EXPECTED_NAMES):
        assert CLS_NAME[i] == name


def test_cls_id_and_cls_name_inverse():
    """CLS_ID 和 CLS_NAME 互为反函数。"""
    for name, idx in CLS_ID.items():
        assert CLS_NAME[idx] == name
    for idx, name in CLS_NAME.items():
        assert CLS_ID[name] == idx


def test_legacy3_is_prefix():
    """LEGACY3_NAMES 是 EXPECTED_NAMES 的前 3 个。"""
    assert EXPECTED_NAMES[:3] == LEGACY3_NAMES
    assert len(LEGACY3_NAMES) == 3


def test_safety_thresholds():
    """安全阈值在合理范围内。"""
    assert 0.0 < RED_CONF_THRESHOLD < GREEN_CONF_THRESHOLD <= 1.0
    assert GREEN_CONF_THRESHOLD >= 0.5, "绿灯阈值不能太低"
    assert RED_CONF_THRESHOLD <= 0.5, "红灯阈值不能太高（保证召回）"


def test_default_parameters():
    """默认参数合理。"""
    assert 0.0 < DEFAULT_CONF <= 1.0
    assert 0.0 < DEFAULT_MATCH_IOU <= 1.0
    assert DEFAULT_IMGSZ > 0
    assert DEFAULT_IMGSZ % 32 == 0, "imgsz 必须是 32 的倍数"


def test_attestation_key():
    """attestation 标记不为空。"""
    assert len(ATTESTATION_KEY) > 0
    assert isinstance(ATTESTATION_KEY, str)
