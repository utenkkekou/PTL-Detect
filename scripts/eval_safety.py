#!/usr/bin/env python3
"""T7 — 安全导向评估（mAP 之外的关键指标）。

头号指标: 危险误报率 P(pred=green | gt=red) —— 真实红灯被判成绿灯的比例, 目标 0。
误报一次绿灯的后果和漏检完全不对等。

还输出:
- red/green/unknown/vehicle_signal 混淆矩阵(含背景行列: 漏检/多检)
- 非行人信号灯被错认成行人灯的比例（第 4 类拒识能力）
- 按目标尺寸分桶(小/中/大)的召回率

匹配: 预测框与 GT 框按像素 IoU 贪心匹配(降序conf, IoU>=--match-iou 一对一)。

用法:
  python scripts/eval_safety.py --model runs/y11n_clean4/weights/best.pt \
      --data data/yolo_clean4/data.yaml --split val --conf 0.25 --match-iou 0.5
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from ptl.constants import DEFAULT_CONF, DEFAULT_IMGSZ, DEFAULT_MATCH_IOU, EXPECTED_NAMES
from ptl.io_utils import box_iou, read_yolo_labels, size_bucket

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 显示用短名（报告输出用，与 CLS_NAME 的全名不同）
CLS_NAME = {0: "red", 1: "green", 2: "unknown", 3: "vehicle_signal"}
CLASS_IDS = tuple(CLS_NAME)


def normalized_names(raw):
    if isinstance(raw, dict):
        return [raw[k] for k in sorted(raw, key=lambda x: int(x))]
    return list(raw or [])


read_gt = read_yolo_labels  # 向后兼容别名
iou = box_iou  # 向后兼容别名
size_bucket_fn = size_bucket  # 向后兼容别名


def evaluate(model, images, labels_dir, conf, match_iou):
    # 混淆: gt_cls -> pred_cls 计数; "bg" 表示漏检(gt无匹配)或多检(pred无匹配)
    confusion = defaultdict(lambda: defaultdict(int))
    size_total = defaultdict(int); size_hit = defaultdict(int)
    from ultralytics import YOLO  # noqa
    for img_p in images:
        res = model.predict(str(img_p), conf=conf, imgsz=DEFAULT_IMGSZ, verbose=False)[0]
        H, W = res.orig_shape
        gts = read_gt(labels_dir / f"{img_p.stem}.txt", W, H)
        preds = [(int(b.cls[0]), float(b.conf[0]), *b.xyxy[0].tolist()) for b in res.boxes]
        preds.sort(key=lambda p: -p[1])  # 高conf优先匹配
        gt_used = [False]*len(gts)
        for pc, pconf, x1, y1, x2, y2 in preds:
            best, bi = match_iou, -1
            for gi, (gc, gx1, gy1, gx2, gy2) in enumerate(gts):
                if gt_used[gi]:
                    continue
                v = iou((x1, y1, x2, y2), (gx1, gy1, gx2, gy2))
                if v >= best:
                    best, bi = v, gi
            if bi >= 0:
                gt_used[bi] = True
                confusion[gts[bi][0]][pc] += 1
            else:
                confusion["bg"][pc] += 1  # 多检(背景被检成某类)
        for gi, used in enumerate(gt_used):
            gc = gts[gi][0]
            size_total[size_bucket(gts[gi][1:], W, H)] += 1
            if used:
                size_hit[size_bucket(gts[gi][1:], W, H)] += 1
            else:
                confusion[gc]["miss"] += 1  # 漏检
    return confusion, size_total, size_hit


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/yolo_clean4/data.yaml")
    ap.add_argument("--split", default="val", choices=["val", "test", "train"])
    ap.add_argument("--conf", type=float, default=DEFAULT_CONF)
    ap.add_argument("--match-iou", type=float, default=DEFAULT_MATCH_IOU)
    ap.add_argument("--out", default=None, help="报告输出 md(默认 runs 同级)")
    ap.add_argument("--allow-noncanonical-classes", action="store_true",
                    help="仅用于复现旧3类实验；默认要求D19四类契约")
    ap.add_argument("--report-only", action="store_true",
                    help="危险混淆非零时仍返回0；默认作为硬验收门返回非零")
    args = ap.parse_args()

    import yaml
    from ultralytics import YOLO
    cfg = yaml.safe_load(Path(args.data).read_text(encoding="utf-8"))
    data_names = normalized_names(cfg.get("names"))
    if data_names != EXPECTED_NAMES and not args.allow_noncanonical_classes:
        print("[STOP] 评估数据不是当前 D19 四类契约。")
        print("  实际:", data_names)
        print("  期望:", EXPECTED_NAMES)
        print("复现旧报告时才加 --allow-noncanonical-classes。")
        return 2
    base = Path(cfg["path"])
    img_dir = base / cfg[args.split]
    lbl_dir = Path(str(img_dir).replace("images", "labels"))
    images = [p for p in img_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    print(f"[EVAL] {args.split}: {len(images)} 图, model={args.model}, conf={args.conf}")

    model = YOLO(args.model)
    model_names = normalized_names(model.names)
    if model_names != EXPECTED_NAMES and not args.allow_noncanonical_classes:
        print("[STOP] 模型不是当前 D19 四类契约。")
        print("  实际:", model_names)
        print("  期望:", EXPECTED_NAMES)
        return 2
    confusion, size_total, size_hit = evaluate(model, images, lbl_dir, args.conf, args.match_iou)

    # 头号指标: P(pred=green | gt=red)
    gt_red = confusion[0]
    red_total = sum(gt_red.values())
    dangerous = gt_red.get(1, 0)  # 真红 -> 判绿
    danger_rate = dangerous / red_total if red_total else 0

    lines = ["# T7 安全评估报告", "",
             f"- 模型: `{args.model}`  数据: {args.split} ({len(images)} 图)  conf={args.conf} match-iou={args.match_iou}",
             "",
             "## 🚨 头号指标: 真实红灯 → 绿灯 (危险误报)",
             f"- 真实红灯框数: {red_total}",
             f"- 被判成绿灯: **{dangerous}**  → 危险误报率 **{danger_rate*100:.2f}%** (目标 0)",
             ""]
    # 红灯去向明细
    lines.append("### 真实红灯的判定去向")
    for k in [0, 1, 2, 3, "miss"]:
        name = {
            0: "→red(正确)",
            1: "→green(🚨危险)",
            2: "→unknown(安全)",
            3: "→vehicle_signal(安全拒识，但分类错误)",
            "miss": "漏检(未检出)",
        }[k]
        v = gt_red.get(k, 0)
        lines.append(f"- {name}: {v} ({100*v/red_total:.1f}%)" if red_total else f"- {name}: {v}")
    lines.append("")

    # 第4类的目的就是显式拒绝车行/自行车信号灯；它被报成行人 green 仍会穿过 App 的忽略门。
    gt_vehicle = confusion[3]
    vehicle_total = sum(gt_vehicle.values())
    vehicle_as_ped = sum(gt_vehicle.get(k, 0) for k in (0, 1, 2))
    vehicle_as_green = gt_vehicle.get(1, 0)
    lines += [
        "## 非行人信号灯拒识",
        f"- 非行人信号灯 GT 框数: {vehicle_total}",
        (f"- 被错认成任一行人灯类: **{vehicle_as_ped}** "
         f"({100*vehicle_as_ped/vehicle_total:.2f}%)"
         if vehicle_total else "- 被错认成任一行人灯类: 0（无 GT）"),
        (f"- 被错认成 green: **{vehicle_as_green}** "
         f"({100*vehicle_as_green/vehicle_total:.2f}%)"
         if vehicle_total else "- 被错认成 green: 0（无 GT）"),
        "",
    ]

    # 混淆矩阵
    lines.append("## 混淆矩阵 (行=真值, 列=预测)")
    lines.append("| gt\\pred | red | green | unknown | vehicle_signal | 漏检 |")
    lines.append("|---|---|---|---|---|---|")
    for gc in CLASS_IDS:
        row = confusion[gc]
        lines.append(
            f"| **{CLS_NAME[gc]}** | {row.get(0,0)} | {row.get(1,0)} | "
            f"{row.get(2,0)} | {row.get(3,0)} | {row.get('miss',0)} |"
        )
    bg = confusion["bg"]
    lines.append(
        f"| **背景(多检)** | {bg.get(0,0)} | {bg.get(1,0)} | "
        f"{bg.get(2,0)} | {bg.get(3,0)} | — |"
    )
    lines.append("")

    # 尺寸分桶召回
    lines.append("## 按目标尺寸的召回率")
    lines.append("| 尺寸 | GT数 | 检出 | 召回 |")
    lines.append("|---|---|---|---|")
    for b in ["small", "medium", "large"]:
        t = size_total.get(b, 0); h = size_hit.get(b, 0)
        lines.append(f"| {b} | {t} | {h} | {100*h/t:.1f}% |" if t else f"| {b} | 0 | 0 | — |")
    lines.append("")
    lines.append("> small=面积<0.2% (远处小目标, 最难也最关键); medium<1%; large≥1%")

    out = Path(args.out) if args.out else Path(args.model).parent.parent / "safety_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[OUT] 报告: {out}")
    if danger_rate > 0:
        print(f"\n⚠️ 危险误报率 {danger_rate*100:.2f}% > 0 —— 见附录A: 绿灯用高阈值/N帧一致 等应用层缓解。")
    if vehicle_as_green > 0:
        print(f"\n⚠️ 非行人信号灯→green 共 {vehicle_as_green} 个 —— 第4类拒识未达标，不能靠降低 green 阈值补召回。")
    failed = dangerous > 0 or vehicle_as_green > 0
    if failed and not args.report_only:
        print("\n[FAIL] 安全硬门未通过（加 --report-only 仅生成报告，不代表验收通过）。")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
