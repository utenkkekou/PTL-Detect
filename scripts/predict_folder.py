#!/usr/bin/env python3
"""在一个文件夹的图片上跑检测, 画框+打印结果表。用于快速目视测试模型。

用法:
  python scripts/predict_folder.py --model models/PedestrianSignalIcon_v1_yolo11n.pt \
      --source data/raw/mine/test6 --out data/interim/pred_test6 --conf 0.25
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

NAME = {0: "red", 1: "green", 2: "unknown", 3: "vehicle_signal"}
COLOR = {
    0: (255, 60, 60),
    1: (60, 220, 60),
    2: (255, 180, 32),
    3: (70, 155, 255),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/PedestrianSignalIcon_v1_yolo11n.pt")
    ap.add_argument("--source", default="data/raw/mine/test6")
    ap.add_argument("--out", default="data/interim/pred_test6")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=960)
    args = ap.parse_args()

    from ultralytics import YOLO
    from PIL import Image, ImageDraw
    model = YOLO(args.model)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    imgs = sorted(p for p in Path(args.source).rglob("*")
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"})
    if not imgs:
        print(f"[ERR] {args.source} 里没有图片"); return 2
    print(f"[PRED] {len(imgs)} 张, model={args.model}, conf={args.conf}, imgsz={args.imgsz}\n")

    for p in imgs:
        res = model.predict(str(p), conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        im = Image.open(p).convert("RGB"); W, H = im.size; d = ImageDraw.Draw(im)
        dets = []
        for b in res.boxes:
            c = int(b.cls[0]); cf = float(b.conf[0])
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            dets.append((c, cf, x1, y1, x2, y2))
        dets.sort(key=lambda t: -t[1])
        print(f"■ {p.name}  ({W}x{H})  检出 {len(dets)} 个:")
        if not dets:
            print("   (无检测 -> 空列表)")
        for c, cf, x1, y1, x2, y2 in dets:
            name = NAME.get(c, str(model.names.get(c, c)))
            print(f"   {name:16} conf={cf:.2f}  box=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")
            col = COLOR.get(c, (220, 220, 220))
            d.rectangle([x1, y1, x2, y2], outline=col, width=max(3, W // 300))
            d.text((x1 + 2, max(0, y1 - 16)), f"{name} {cf:.2f}", fill=col)
        # 缩放保存便于查看
        if max(im.size) > 1400:
            im.thumbnail((1400, 1400))
        im.save(out / f"pred_{p.stem}.jpg", quality=90)
        print()
    print(f"[OUT] 画框图 -> {out}")


if __name__ == "__main__":
    raise SystemExit(main())
