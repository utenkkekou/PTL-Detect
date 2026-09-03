#!/usr/bin/env python3
"""T6 — 训练 YOLO11n 四类信号灯检测器（锁定红绿灯安全超参）。

当前有效数据契约见 docs/decisions.md D19。关键增强项见 D7/D18:
  hsv_h=0 (色相是判别信号本身, 扰动=教模型把红认成绿)
  fliplr=0 (小人图标有朝向)
  imgsz>=640, 默认960(新契约推荐)

用法:
  python scripts/train.py --data data/yolo_clean4/data.yaml \
      --name y11n_clean4 --imgsz 960 --epochs 150
"""
import argparse
import sys
from pathlib import Path

from ptl.constants import DEFAULT_IMGSZ, EXPECTED_NAMES


def read_names(data_yaml):
    import yaml
    cfg = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    names = cfg.get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=lambda x: int(x))]
    return list(names)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--data", required=True)
    ap.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--batch", default="16", help="int 或 -1(自动)")
    ap.add_argument("--name", default="y11n")
    ap.add_argument("--project", default="runs")
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=6, help="DataLoader进程数(Windows共享内存紧张时调低)")
    ap.add_argument("--scale", type=float, default=0.5,
                    help="尺度增强幅度(1±scale); 提高可让模型见到更大目标, 治近大灯尺度盲区")
    ap.add_argument("--mosaic", type=float, default=1.0, help="mosaic概率(会缩小目标, 治大灯时可调低)")
    ap.add_argument("--optimizer", default="auto",
                    help="Ultralytics optimizer；增量微调可显式设 AdamW")
    ap.add_argument("--lr0", type=float, default=0.01,
                    help="初始学习率；从现有权重微调时应显著低于默认值")
    ap.add_argument("--lrf", type=float, default=0.01,
                    help="末端学习率系数")
    ap.add_argument("--allow-noncanonical-classes", action="store_true",
                    help="仅用于复现旧实验；默认拒绝非 D19 四类数据，防止误训污染集")
    args = ap.parse_args()

    names = read_names(args.data)
    if names != EXPECTED_NAMES and not args.allow_noncanonical_classes:
        sys.exit(
            "[STOP] data.yaml 不是当前 D19 四类契约，拒绝训练以防误用旧 ImVisible 数据。\n"
            f"  实际: {names}\n  期望: {EXPECTED_NAMES}\n"
            "复现历史实验时才加 --allow-noncanonical-classes。")

    from ultralytics import YOLO
    model = YOLO(args.model)
    batch = int(args.batch)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        patience=args.patience,
        batch=batch,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=args.name,
        seed=0,
        plots=True,
        # --- 红绿灯安全增强锁定 (D7) ---
        hsv_h=0.0,     # 必须0: 色相=判别信号
        hsv_s=0.5,
        hsv_v=0.4,
        fliplr=0.0,    # 小人图标有朝向
        flipud=0.0,
        degrees=0.0,
        translate=0.1,
        scale=args.scale,
        mosaic=args.mosaic,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        close_mosaic=15,
        mixup=0.0,
        copy_paste=0.0,
    )
    print(f"\n[DONE] 权重: {args.project}/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
