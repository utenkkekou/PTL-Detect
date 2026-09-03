#!/usr/bin/env python3
"""T3(B 计划) — 用行人灯专用检测器(v0)重新预标注剩余图片。

⚠️ 历史脚本：D19 已弃用被车灯污染的 ImVisible 数据。当前 clean4 流程不要运行本脚本；
仅在明确复现旧实验时加 ``--allow-deprecated-imvisible``。

与 ptl_pre_annotate.py(COCO)不同:
  - 用训好的 3 类模型, 每个框直接带【预测类别】(red/green/unknown), 无需图级赋色
  - 只出行人灯候选(不会框车行灯), 复核负担应远低于 COCO
  - 复用已生成的预览图(data/interim/ptl/preview), 不重复生成
  - 跳过已复核的种子集(--exclude 指向种子进度)

产出: --out/reann_manifest.json (复核工具可直接用)

用法:
  python scripts/ptl_reannotate.py --model runs/detect/runs/y11n_v0_seed/weights/best.pt \
      --images-dir data/raw/imvisible/images \
      --exclude data/interim/ptl/seed_progress.json \
      --out data/interim/ptl --tag reann
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REPO = Path(__file__).resolve().parents[1]
CSV_DIR = REPO / "data/raw/imvisible/Annotations"


def load_modes() -> dict[str, dict]:
    out = {}
    for split in ("training", "validation", "testing"):
        p = CSV_DIR / f"{split}_file.csv"
        if p.exists():
            with p.open(encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    out[row["file"].strip()] = {"mode": int(row["mode"]), "split": split}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="训好的3类检测器 best.pt")
    ap.add_argument("--images-dir", default=str(REPO / "data/raw/imvisible/images"))
    ap.add_argument("--exclude", default=None,
                    help="进度json(其中 reviewed 的 id 跳过, 如种子集)")
    ap.add_argument("--out", default=str(REPO / "data/interim/ptl"))
    ap.add_argument("--tag", default="reann", help="输出 manifest 前缀")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--preview-max", type=int, default=1280)
    ap.add_argument("--allow-deprecated-imvisible", action="store_true",
                    help="明确允许复现已弃用的 ImVisible 流程")
    args = ap.parse_args()
    if not args.allow_deprecated_imvisible:
        print("[STOP] D19 已弃用 ImVisible；本脚本只供历史复现。")
        print("       当前自采数据请用 video_to_dataset.py + prepare_four_class_review.py。")
        return 2

    modes = load_modes()
    img_idx = {p.name: p for p in Path(args.images_dir).rglob("*")
               if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}

    exclude = set()
    if args.exclude and Path(args.exclude).exists():
        prog = json.loads(Path(args.exclude).read_text(encoding="utf-8"))
        exclude = {k for k, v in prog.items() if v.get("status") == "reviewed"}
    print(f"[INFO] 图片 {len(img_idx)}, 排除已复核 {len(exclude)}")

    items = [(fn, m) for fn, m in modes.items()
             if fn in img_idx and Path(img_idx[fn]).stem not in exclude]
    if args.limit:
        items = items[: args.limit]
    print(f"[INFO] 待重标注 {len(items)}")

    from ultralytics import YOLO
    from PIL import Image
    model = YOLO(args.model)
    print(f"[INFO] 模型类名: {model.names}")

    out_dir = Path(args.out)
    prev_dir = out_dir / "preview"
    prev_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    cls_cnt = Counter()
    n_box_hist = Counter()
    n_fail = 0
    for i, (fn, meta) in enumerate(items):
        img_p = img_idx[fn]
        stem = img_p.stem
        try:
            res = model.predict(str(img_p), conf=args.conf, imgsz=args.imgsz,
                                iou=args.iou, verbose=False)
            if not res:
                raise RuntimeError("empty")
            res = res[0]
        except Exception as e:
            n_fail += 1
            print(f"  [SKIP] {fn}: {e}")
            continue
        W, H = res.orig_shape[1], res.orig_shape[0]
        boxes = []
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxyn[0].tolist()
            cls = int(b.cls[0]); conf = float(b.conf[0])
            boxes.append({"cls": cls, "cx": round((x1 + x2) / 2, 6),
                          "cy": round((y1 + y2) / 2, 6),
                          "w": round(x2 - x1, 6), "h": round(y2 - y1, 6),
                          "conf": round(conf, 4)})
            cls_cnt[cls] += 1
        n_box_hist[len(boxes)] += 1

        # 复用预览, 缺则生成
        prev = prev_dir / f"{stem}.jpg"
        if not prev.exists():
            try:
                im = Image.open(img_p).convert("RGB")
                im.thumbnail((args.preview_max, args.preview_max))
                im.save(prev, quality=85)
            except Exception:
                pass
        # 排序键: 低conf优先(可疑); 无框但ImVisible说有灯 -> 最前
        expect_light = meta["mode"] != 4
        if expect_light and not boxes:
            sort_key = -1.0
        elif not expect_light and not boxes:
            sort_key = 2.0
        else:
            sort_key = min((b["conf"] for b in boxes), default=1.5)
        manifest.append({
            "id": stem, "file": fn, "preview": f"preview/{stem}.jpg",
            "width": W, "height": H, "mode": meta["mode"],
            "target_class": (boxes[0]["cls"] if boxes else -1),  # 仅供显示
            "boxes": boxes, "n_boxes": len(boxes),
            "sort_key": round(sort_key, 4), "multi": len(boxes) > 1,
            "split": meta["split"],
        })
        if (i + 1) % 200 == 0:
            print(f"  ... {i+1}/{len(items)}")

    manifest.sort(key=lambda m: m["sort_key"])
    out_path = out_dir / f"{args.tag}_manifest.json"
    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[OUT] {out_path} ({len(manifest)} 项), 跳过失败 {n_fail}")
    total = sum(cls_cnt.values())
    print(f"[STAT] 总框 {total}, 均 {total/max(1,len(manifest)):.2f} 框/图")
    print(f"[STAT] 类别: " + ", ".join(f"{k}={v}" for k, v in sorted(cls_cnt.items()))
          + "  (0=red,1=green,2=unknown)")
    print(f"[STAT] 框数分布: " + ", ".join(f"{k}:{v}" for k, v in sorted(n_box_hist.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
