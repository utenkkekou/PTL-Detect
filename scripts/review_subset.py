#!/usr/bin/env python3
"""从已复核任务抽出少量问题帧做二次纠错，并安全合并回主进度。

准备纠错清单:
  python scripts/review_subset.py prepare \
    --manifest data/interim/mine_t10/review_manifest_4cls.json \
    --progress data/interim/mine_t10/review_progress_4cls.json \
    --out-manifest data/interim/mine_t10/review_manifest_corrections.json \
    --out-progress data/interim/mine_t10/review_progress_corrections.json \
    --ios-threshold 0.8 --id 美院路口3__00013

纠错全部完成后合并:
  python scripts/review_subset.py merge \
    --subset-manifest data/interim/mine_t10/review_manifest_corrections.json \
    --subset-progress data/interim/mine_t10/review_progress_corrections.json \
    --target-progress data/interim/mine_t10/review_progress_4cls.json
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

ATTESTATION = "independent-visual-qa-complete"

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def overlap_over_smaller(a: dict, b: dict) -> float:
    ax1, ay1 = a["cx"] - a["w"] / 2, a["cy"] - a["h"] / 2
    ax2, ay2 = a["cx"] + a["w"] / 2, a["cy"] + a["h"] / 2
    bx1, by1 = b["cx"] - b["w"] / 2, b["cy"] - b["h"] / 2
    bx2, by2 = b["cx"] + b["w"] / 2, b["cy"] + b["h"] / 2
    intersection = (
        max(0.0, min(ax2, bx2) - max(ax1, bx1))
        * max(0.0, min(ay2, by2) - max(ay1, by1))
    )
    return intersection / max(1e-12, min(a["w"] * a["h"], b["w"] * b["h"]))


def overlap_reasons(boxes: list[dict], threshold: float) -> list[str]:
    reasons = []
    for i, a in enumerate(boxes):
        for j, b in enumerate(boxes[i + 1:], start=i + 1):
            score = overlap_over_smaller(a, b)
            if score >= threshold:
                relation = "同类" if a["cls"] == b["cls"] else "跨类"
                reasons.append(
                    f"{relation}强重叠框 #{i + 1}/#{j + 1} (IoS={score:.3f})"
                )
    return reasons


def build_subset(manifest: list[dict], progress: dict, explicit_ids: set[str],
                 ios_threshold: float | None = None,
                 class_ids: set[int] | None = None,
                 id_prefixes: tuple[str, ...] = (),
                 seed_unreviewed_from_manifest: bool = False) -> list[dict]:
    manifest_ids = [item["id"] for item in manifest]
    if len(manifest_ids) != len(set(manifest_ids)):
        raise ValueError("主 manifest 含重复 ID")
    missing = sorted(explicit_ids - set(manifest_ids))
    if missing:
        raise ValueError(f"显式问题 ID 不在主 manifest: {missing[:10]}")

    subset = []
    for item in manifest:
        iid = item["id"]
        saved = progress.get(iid)
        used_manifest_seed = False
        if not saved or saved.get("status") != "reviewed" or "boxes" not in saved:
            if not seed_unreviewed_from_manifest:
                raise ValueError(f"主进度尚未完整复核或缺 boxes: {iid}")
            boxes = item["boxes"]
            used_manifest_seed = True
        else:
            boxes = saved["boxes"]
        reasons = []
        if iid in explicit_ids:
            reasons.append("目视抽检命中")
        matching_classes = sorted({
            box["cls"] for box in boxes
            if box["cls"] in (class_ids or set())
        })
        if matching_classes:
            reasons.append(f"命中待核类别 {matching_classes}")
        if id_prefixes and iid.startswith(id_prefixes):
            reasons.append("命中待核视频段")
        if ios_threshold is not None:
            reasons.extend(overlap_reasons(boxes, ios_threshold))
        if not reasons:
            continue
        if used_manifest_seed:
            reasons.append("原任务尚未复核，使用 manifest 框作为新任务种子")
        selected = dict(item)
        selected["boxes"] = boxes
        selected["n_boxes"] = len(boxes)
        selected["multi"] = len(boxes) > 1
        selected["target_class"] = boxes[0]["cls"] if boxes else -1
        selected["sort_key"] = -1.0
        selected["review_note"] = "；".join(reasons)
        subset.append(selected)
    if not subset:
        raise ValueError("没有选出任何纠错帧")
    # ID 中帧号已补零；按 ID 排序可让同视频相邻时序连续出现，便于判断暗灯头类别。
    subset.sort(key=lambda item: item["id"])
    return subset


def _append_review_note(item: dict, note: str | None) -> None:
    if not note:
        return
    current = item.get("review_note", "").strip()
    item["review_note"] = f"{current}；{note}" if current else note


def _check_expected_box(box: dict, spec: dict) -> None:
    if "from" in spec and box.get("cls") != spec["from"]:
        raise ValueError(
            f"{spec['id']} box #{spec['index'] + 1} 类别保护失败: "
            f"期望 {spec['from']}，实际 {box.get('cls')}"
        )
    for key in ("cx", "cy", "w", "h"):
        if key in spec and abs(float(box.get(key, -99)) - float(spec[key])) > 1e-6:
            raise ValueError(
                f"{spec['id']} box #{spec['index'] + 1} 几何保护失败: {key}"
            )


def _validated_index(spec: dict, boxes: list[dict]) -> int:
    required = {"id", "index", "from", "cx", "cy", "w", "h"}
    missing = sorted(required - set(spec))
    if missing:
        raise ValueError(f"精确建议缺少保护字段 {missing}: {spec.get('id')}")
    index = spec["index"]
    if (
        not isinstance(index, int)
        or isinstance(index, bool)
        or index < 0
        or index >= len(boxes)
    ):
        raise ValueError(f"{spec['id']} box index 越界或类型错误: {index}")
    source_class = spec["from"]
    if (
        not isinstance(source_class, int)
        or isinstance(source_class, bool)
        or source_class not in range(4)
    ):
        raise ValueError(f"{spec['id']} 精确建议来源类别非法: {source_class}")
    for key in ("cx", "cy", "w", "h"):
        value = spec[key]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"{spec['id']} 精确建议几何保护非法: {key}={value}")
    return index


def _validate_box(box: dict, context: str) -> None:
    if not isinstance(box, dict):
        raise ValueError(f"{context} box 必须是对象")
    required = {"cls", "cx", "cy", "w", "h"}
    missing = sorted(required - set(box))
    if missing:
        raise ValueError(f"{context} box 缺少字段: {missing}")
    cls = box["cls"]
    if not isinstance(cls, int) or isinstance(cls, bool) or cls not in range(4):
        raise ValueError(f"{context} box 类别非法: {cls}")
    values = {}
    for key in ("cx", "cy", "w", "h"):
        value = box[key]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"{context} box 几何非法: {key}={value}")
        values[key] = float(value)
    cx, cy, width, height = (
        values["cx"], values["cy"], values["w"], values["h"]
    )
    if (
        width <= 0
        or height <= 0
        or cx < 0
        or cx > 1
        or cy < 0
        or cy > 1
        or cx - width / 2 < 0
        or cx + width / 2 > 1
        or cy - height / 2 < 0
        or cy + height / 2 > 1
    ):
        raise ValueError(f"{context} box 超出归一化图像边界")
    if "conf" in box:
        conf = box["conf"]
        if (
            not isinstance(conf, (int, float))
            or isinstance(conf, bool)
            or not math.isfinite(float(conf))
            or float(conf) < 0
            or float(conf) > 1
        ):
            raise ValueError(f"{context} box conf 非法: {conf}")


def _validate_box_add(spec: dict, item: dict) -> None:
    required = {"id", "cls", "cx", "cy", "w", "h", "expected_n_boxes"}
    optional = {"conf", "note"}
    if not isinstance(spec, dict):
        raise ValueError("box_add 建议必须是对象")
    missing = sorted(required - set(spec))
    if missing:
        raise ValueError(f"box_add 缺少保护字段 {missing}: {spec.get('id')}")
    unknown = sorted(set(spec) - required - optional)
    if unknown:
        raise ValueError(f"box_add 含未知字段 {unknown}: {spec.get('id')}")
    iid = spec["id"]
    if not isinstance(iid, str) or not iid:
        raise ValueError(f"box_add ID 非法: {iid}")
    expected_n_boxes = spec["expected_n_boxes"]
    if (
        not isinstance(expected_n_boxes, int)
        or isinstance(expected_n_boxes, bool)
        or expected_n_boxes < 0
    ):
        raise ValueError(f"{iid} box_add expected_n_boxes 非法: {expected_n_boxes}")
    if expected_n_boxes != len(item["boxes"]):
        raise ValueError(
            f"{iid} box_add 原始框数保护失败: "
            f"期望 {expected_n_boxes}，实际 {len(item['boxes'])}"
        )
    candidate = {
        key: spec[key] for key in ("cls", "cx", "cy", "w", "h")
    }
    if "conf" in spec:
        candidate["conf"] = spec["conf"]
    _validate_box(candidate, f"{iid} box_add")
    if "note" in spec and not isinstance(spec["note"], str):
        raise ValueError(f"{iid} box_add note 必须是字符串")


def _box_geometry_key(box: dict) -> tuple[float, float, float, float]:
    return tuple(float(box[key]) for key in ("cx", "cy", "w", "h"))


def _box_semantic_key(box: dict) -> tuple[int, float, float, float, float]:
    return (int(box["cls"]), *_box_geometry_key(box))


def _boxes_semantically_equal(left: list[dict], right: list[dict]) -> bool:
    """Compare labels/geometry while ignoring confidence and box ordering."""
    return sorted(_box_semantic_key(box) for box in left) == sorted(
        _box_semantic_key(box) for box in right
    )


def apply_suggestions(subset: list[dict], suggestions: dict) -> list[dict]:
    """Apply guarded, auditable seed edits before a human correction pass."""
    result = copy.deepcopy(subset)
    by_id = {item["id"]: item for item in result}
    if len(by_id) != len(result):
        raise ValueError("纠错子集含重复 ID")

    add_specs = suggestions.get("box_add", [])
    if not isinstance(add_specs, list):
        raise ValueError("box_add 必须是列表")
    add_keys = []
    for spec in add_specs:
        if not isinstance(spec, dict):
            raise ValueError("box_add 建议必须是对象")
        item = by_id.get(spec.get("id"))
        if item is None:
            raise ValueError(f"box_add ID 不在纠错子集: {spec.get('id')}")
        _validate_box_add(spec, item)
        add_keys.append(
            (spec["id"], spec["cls"], *_box_geometry_key(spec))
        )
    if len(add_keys) != len(set(add_keys)):
        raise ValueError("同一新增 box 不能重复出现")

    exact_ops: list[tuple[str, dict]] = []
    for operation in ("box_reclass", "box_delete"):
        for spec in suggestions.get(operation, []):
            exact_ops.append((operation, spec))
    op_keys = [(spec.get("id"), spec.get("index")) for _, spec in exact_ops]
    if len(op_keys) != len(set(op_keys)):
        raise ValueError("同一 box 不能同时出现多个精确建议")
    # 精确保护先对原始种子验证；之后前缀改类可以安全地被更精确的操作覆盖。
    for operation, spec in exact_ops:
        item = by_id.get(spec.get("id"))
        if item is None:
            raise ValueError(f"精确建议 ID 不在纠错子集: {spec.get('id')}")
        index = _validated_index(spec, item["boxes"])
        _check_expected_box(item["boxes"][index], spec)
        if operation == "box_reclass":
            target = spec.get("to")
            if (
                not isinstance(target, int)
                or isinstance(target, bool)
                or target not in range(4)
            ):
                raise ValueError(f"精确改类目标非法: {target}")

    prefix_rules = suggestions.get("prefix_reclass", [])
    for rule in prefix_rules:
        prefix = rule.get("prefix")
        source_values = rule.get("from")
        target_class = rule.get("to")
        expected_changes = rule.get("expected_changes")
        if (
            not isinstance(prefix, str)
            or not prefix
            or not isinstance(source_values, list)
            or not source_values
            or any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value not in range(4)
                for value in source_values
            )
        ):
            raise ValueError(f"前缀改类来源类别非法: {rule}")
        if (
            not isinstance(target_class, int)
            or isinstance(target_class, bool)
            or target_class not in range(4)
            or target_class in source_values
        ):
            raise ValueError(f"前缀改类目标非法: {target_class}")
        if (
            not isinstance(expected_changes, int)
            or isinstance(expected_changes, bool)
            or expected_changes < 0
        ):
            raise ValueError(f"前缀改类必须声明非负 expected_changes: {prefix}")
        if "vehicle_only" in rule and not isinstance(rule["vehicle_only"], bool):
            raise ValueError(f"vehicle_only 必须是布尔值: {prefix}")
    matched_rule_count = {
        item["id"]: sum(
            item["id"].startswith(rule.get("prefix", ""))
            for rule in prefix_rules
        )
        for item in result
    }
    overlapping = sorted(
        iid for iid, count in matched_rule_count.items() if count > 1
    )
    if overlapping:
        raise ValueError(f"同一帧命中多个前缀改类规则: {overlapping[:10]}")
    for rule in prefix_rules:
        prefix = rule["prefix"]
        source_values = rule.get("from")
        expected_changes = rule.get("expected_changes")
        source_classes = set(source_values)
        target_class = rule["to"]
        matched = [item for item in result if item["id"].startswith(prefix)]
        if not matched:
            raise ValueError(f"前缀建议未命中任何帧: {prefix}")
        total_changed = 0
        for item in matched:
            changed = 0
            for box in item["boxes"]:
                if box["cls"] in source_classes:
                    box["cls"] = target_class
                    changed += 1
            total_changed += changed
            if rule.get("note"):
                _append_review_note(item, rule["note"])
            elif changed:
                _append_review_note(
                    item,
                    f"已预改 {changed} 框为 class {target_class}，请逐框确认",
                )
            if "vehicle_only" in rule:
                if rule["vehicle_only"]:
                    item["vehicle_only"] = True
                else:
                    item.pop("vehicle_only", None)
        if total_changed != expected_changes:
            raise ValueError(
                f"前缀 {prefix} 预改数量保护失败: "
                f"期望 {expected_changes}，实际 {total_changed}"
            )

    for operation, spec in (
        pair for pair in exact_ops if pair[0] == "box_reclass"
    ):
        item = by_id.get(spec["id"])
        index = spec["index"]
        box = item["boxes"][index]
        box["cls"] = spec["to"]
        _append_review_note(item, spec.get("note"))

    deletes_by_id: dict[str, list[dict]] = {}
    for operation, spec in exact_ops:
        if operation == "box_delete":
            deletes_by_id.setdefault(spec["id"], []).append(spec)
    for iid, specs in deletes_by_id.items():
        item = by_id.get(iid)
        for spec in sorted(specs, key=lambda value: value["index"], reverse=True):
            index = spec["index"]
            del item["boxes"][index]
            _append_review_note(item, spec.get("note"))

    for spec in add_specs:
        item = by_id[spec["id"]]
        geometry = _box_geometry_key(spec)
        if any(_box_geometry_key(box) == geometry for box in item["boxes"]):
            raise ValueError(f"{spec['id']} box_add 与已有 box 重复")
        added = {
            key: spec[key] for key in ("cls", "cx", "cy", "w", "h")
        }
        if "conf" in spec:
            added["conf"] = spec["conf"]
        item["boxes"].append(added)
        _append_review_note(item, spec.get("note"))

    for iid, note in suggestions.get("notes", {}).items():
        item = by_id.get(iid)
        if item is None:
            raise ValueError(f"复核提示 ID 不在纠错子集: {iid}")
        _append_review_note(item, note)

    required_ids = suggestions.get("requires_change", [])
    if (
        not isinstance(required_ids, list)
        or any(not isinstance(iid, str) or not iid for iid in required_ids)
        or len(required_ids) != len(set(required_ids))
    ):
        raise ValueError("requires_change 必须是无重复的非空 ID 列表")
    missing_required = sorted(set(required_ids) - set(by_id))
    if missing_required:
        raise ValueError(f"必须修改 ID 不在纠错子集: {missing_required[:10]}")
    for iid in required_ids:
        by_id[iid]["requires_change"] = True

    for item in result:
        boxes = item["boxes"]
        item["n_boxes"] = len(boxes)
        item["multi"] = len(boxes) > 1
        item["target_class"] = boxes[0]["cls"] if boxes else -1
    return result


def merge_subset(target: dict, subset_manifest: list[dict],
                 subset_progress: dict) -> dict:
    ids = [item["id"] for item in subset_manifest]
    if len(ids) != len(set(ids)):
        raise ValueError("纠错 manifest 含重复 ID")
    missing_target = sorted(set(ids) - set(target))
    if missing_target:
        raise ValueError(f"纠错 ID 不在主进度: {missing_target[:10]}")
    incomplete = [
        iid for iid in ids
        if not subset_progress.get(iid)
        or subset_progress[iid].get("status") != "reviewed"
        or "boxes" not in subset_progress[iid]
    ]
    if incomplete:
        raise ValueError(f"纠错尚未全部完成: {len(incomplete)} 项，例如 {incomplete[:10]}")

    merged = dict(target)
    for iid in ids:
        corrected = dict(subset_progress[iid])
        corrected["changed"] = not _boxes_semantically_equal(
            corrected["boxes"],
            target[iid].get("boxes", []),
        )
        merged[iid] = corrected
    return merged


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_new_json(path: Path, value) -> None:
    if path.exists():
        raise FileExistsError(f"拒绝覆盖已有文件: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _check_sha256(path: Path, expected: str, label: str) -> None:
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or any(char not in "0123456789abcdefABCDEF" for char in expected)
    ):
        raise ValueError(f"{label} expected sha256 格式非法")
    actual = sha256(path)
    if actual != expected.upper():
        raise ValueError(
            f"{label} sha256 不匹配: 期望 {expected.upper()}，实际 {actual}"
        )


def attest(args) -> int:
    if args.attestation != ATTESTATION:
        raise ValueError(f"attestation 必须精确为 {ATTESTATION}")
    if not isinstance(args.reviewer, str) or not args.reviewer.strip():
        raise ValueError("reviewer 必须是非空字符串")

    subset_manifest_path = Path(args.subset_manifest)
    source_progress_path = Path(args.source_progress)
    out_progress_path = Path(args.out_progress)
    _check_sha256(
        subset_manifest_path,
        args.expected_manifest_sha256,
        "subset manifest",
    )
    _check_sha256(
        source_progress_path,
        args.expected_source_sha256,
        "source progress",
    )
    subset_manifest = read_json(subset_manifest_path)
    source_progress = read_json(source_progress_path)
    if not isinstance(subset_manifest, list) or not subset_manifest:
        raise ValueError("subset manifest 必须是非空列表")
    if not isinstance(source_progress, dict):
        raise ValueError("source progress 必须是对象")

    ids = []
    attested = {}
    for item in subset_manifest:
        if not isinstance(item, dict):
            raise ValueError("subset manifest 每项必须是对象")
        iid = item.get("id")
        if not isinstance(iid, str) or not iid:
            raise ValueError(f"subset manifest ID 非法: {iid}")
        ids.append(iid)
        if "boxes" not in item or not isinstance(item["boxes"], list):
            raise ValueError(f"subset manifest 缺少合法 boxes: {iid}")
        saved = source_progress.get(iid)
        if (
            not isinstance(saved, dict)
            or saved.get("status") != "reviewed"
            or not isinstance(saved.get("boxes"), list)
        ):
            raise ValueError(f"subset ID 不在已复核 source progress: {iid}")
        for index, box in enumerate(item["boxes"]):
            _validate_box(box, f"subset {iid} #{index + 1}")
        for index, box in enumerate(saved["boxes"]):
            _validate_box(box, f"source {iid} #{index + 1}")
        boxes = copy.deepcopy(item["boxes"])
        attested[iid] = {
            "status": "reviewed",
            "boxes": boxes,
            "changed": not _boxes_semantically_equal(boxes, saved["boxes"]),
            "by": args.reviewer,
        }
    if len(ids) != len(set(ids)):
        raise ValueError("subset manifest 含重复 ID")

    write_new_json(out_progress_path, attested)
    print(f"[ATTEST] 已物化独立目视 QA 结果: {len(attested)} 帧")
    print(f"         progress={out_progress_path}")
    print(f"         sha256={sha256(out_progress_path)}")
    return 0


def prepare(args) -> int:
    manifest_path = Path(args.manifest)
    progress_path = Path(args.progress)
    out_manifest = Path(args.out_manifest)
    out_progress = Path(args.out_progress)
    if out_manifest.resolve() == out_progress.resolve():
        raise ValueError("纠错 manifest 与 progress 不能写到同一路径")
    explicit_ids = set(args.id)
    if args.ids_file:
        explicit_ids.update(
            line.strip()
            for line in Path(args.ids_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    suggestions = read_json(Path(args.suggestions)) if args.suggestions else {}
    suggestion_prefixes = tuple(
        rule["prefix"] for rule in suggestions.get("prefix_reclass", [])
    )
    for operation in ("box_reclass", "box_delete", "box_add"):
        specs = suggestions.get(operation, [])
        if not isinstance(specs, list):
            raise ValueError(f"{operation} 必须是列表")
        for spec in specs:
            if (
                not isinstance(spec, dict)
                or not isinstance(spec.get("id"), str)
                or not spec["id"]
            ):
                raise ValueError(f"{operation} 建议缺少合法 ID")
            explicit_ids.add(spec["id"])
    explicit_ids.update(suggestions.get("notes", {}).keys())
    explicit_ids.update(suggestions.get("requires_change", []))
    subset = build_subset(
        read_json(manifest_path),
        read_json(progress_path),
        explicit_ids,
        args.ios_threshold,
        set(args.class_id),
        tuple(args.id_prefix) + suggestion_prefixes,
        args.seed_unreviewed_from_manifest,
    )
    if suggestions:
        subset = apply_suggestions(subset, suggestions)
    # 两个输出都预检，避免只写成功一半。
    for path in (out_manifest, out_progress):
        if path.exists():
            raise FileExistsError(f"拒绝覆盖已有文件: {path}")
    write_new_json(out_manifest, subset)
    write_new_json(out_progress, {})
    print(f"[REVIEW] 已生成定向纠错清单: {len(subset)} 帧")
    print(f"         manifest={out_manifest}")
    print(f"         progress={out_progress}")
    print("         规则：一物理灯头一框；不框倒计时数字；机动车/自行车灯用 class 4。")
    return 0


def merge(args) -> int:
    subset_manifest_path = Path(args.subset_manifest)
    subset_progress_path = Path(args.subset_progress)
    target_path = Path(args.target_progress)
    target = read_json(target_path)
    merged = merge_subset(
        target,
        read_json(subset_manifest_path),
        read_json(subset_progress_path),
    )
    backup = (
        Path(args.backup)
        if args.backup
        else target_path.with_name(
            f"{target_path.stem}.before_{subset_progress_path.stem}.json"
        )
    )
    if backup.exists():
        raise FileExistsError(f"备份文件已存在，拒绝覆盖: {backup}")
    shutil.copy2(target_path, backup)
    tmp = target_path.with_suffix(".merge.tmp")
    tmp.write_text(
        json.dumps(merged, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    tmp.replace(target_path)
    print(f"[MERGE] 合并 {len(read_json(subset_manifest_path))} 帧 -> {target_path}")
    print(f"        原进度备份: {backup} (sha256={sha256(backup)})")
    print(f"        新进度 sha256={sha256(target_path)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--manifest", required=True)
    prepare_parser.add_argument("--progress", required=True)
    prepare_parser.add_argument("--out-manifest", required=True)
    prepare_parser.add_argument("--out-progress", required=True)
    prepare_parser.add_argument("--id", action="append", default=[])
    prepare_parser.add_argument("--id-prefix", action="append", default=[],
                                help="选出 ID 以此前缀开头的全部帧（可重复）")
    prepare_parser.add_argument("--class-id", action="append", type=int, default=[],
                                help="选出当前框含该类别的全部帧（可重复）")
    prepare_parser.add_argument("--ids-file", default=None)
    prepare_parser.add_argument(
        "--suggestions",
        default=None,
        help="带类别/几何保护的预修正建议 JSON；仍需逐帧人工确认",
    )
    prepare_parser.add_argument(
        "--ios-threshold", type=float, default=None,
        help="自动选出 intersection/min(area) 不小于阈值的强重叠框帧",
    )
    prepare_parser.add_argument(
        "--seed-unreviewed-from-manifest",
        action="store_true",
        help="显式允许新任务对未复核 ID 使用 manifest 框；默认仍拒绝不完整主进度",
    )
    prepare_parser.set_defaults(func=prepare)

    merge_parser = sub.add_parser("merge")
    merge_parser.add_argument("--subset-manifest", required=True)
    merge_parser.add_argument("--subset-progress", required=True)
    merge_parser.add_argument("--target-progress", required=True)
    merge_parser.add_argument("--backup", default=None)
    merge_parser.set_defaults(func=merge)

    attest_parser = sub.add_parser("attest")
    attest_parser.add_argument("--subset-manifest", required=True)
    attest_parser.add_argument("--source-progress", required=True)
    attest_parser.add_argument("--out-progress", required=True)
    attest_parser.add_argument("--reviewer", required=True)
    attest_parser.add_argument("--expected-manifest-sha256", required=True)
    attest_parser.add_argument("--expected-source-sha256", required=True)
    attest_parser.add_argument("--attestation", required=True)
    attest_parser.set_defaults(func=attest)

    args = parser.parse_args()
    try:
        return args.func(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[ERR] {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
