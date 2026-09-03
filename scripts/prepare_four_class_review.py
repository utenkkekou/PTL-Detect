#!/usr/bin/env python3
"""把已复核的 3 类 T10 标注升级成待复核的 4 类清单。

D19 后的第 4 类是 ``vehicle_signal``：机动车、非机动车/自行车信号灯。
旧 3 类模型不会给出这一类，因此用 COCO ``traffic light``（class 9）只做高召回
候选，再由标注员按 4 类规则纠正。COCO 候选不是最终标签。

重要：输出必须配一个全新的 progress 文件。旧 progress 会覆盖 manifest 中新增的
class 3 候选，造成“看起来已完成、实际一个新框都没复核”的静默错误。

用法:
  python scripts/prepare_four_class_review.py \
      --manifest data/interim/mine_t10/review_manifest.json \
      --progress data/interim/mine_t10/review_progress.json \
      --root data/interim/mine_t10 \
      --vehicle-model yolo11x.pt \
      --out data/interim/mine_t10/review_manifest_4cls.json

  python tools/review/serve_review.py \
      --manifest data/interim/mine_t10/review_manifest_4cls.json \
      --root data/interim/mine_t10 \
      --progress data/interim/mine_t10/review_progress_4cls.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def reviewed_seed(manifest, progress, allow_unreviewed=False):
    """返回 [(manifest_item, 人工框)]，默认要求每一项均已复核。"""
    seeded = []
    missing = []
    for item in manifest:
        saved = progress.get(item["id"])
        if saved and saved.get("status") == "reviewed":
            boxes = saved.get("boxes", [])
        elif allow_unreviewed:
            boxes = item.get("boxes", [])
        else:
            missing.append(item["id"])
            continue
        bad = sorted({int(b["cls"]) for b in boxes if int(b["cls"]) not in (0, 1, 2)})
        if bad:
            raise ValueError(
                f"{item['id']}: 种子进度应为旧 3 类，出现类别 {bad}；"
                "若已经做过 4 类复核，不要重新生成 manifest。")
        seeded.append((item, [dict(b) for b in boxes]))
    if missing:
        sample = ", ".join(missing[:5])
        raise ValueError(
            f"{len(missing)} 项尚未复核（例: {sample}）。"
            "先完成旧标注，或明确加 --allow-unreviewed 接受自动框。")
    return seeded


def append_vehicle_candidates(seeded, root, model_path, coco_class, conf, imgsz, iou):
    from ultralytics import YOLO

    paths = []
    for item, _ in seeded:
        path = (root / item["preview"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{item['id']}: 图片不存在: {path}")
        paths.append(str(path))

    model = YOLO(str(model_path))
    results = model.predict(
        paths,
        classes=[coco_class],
        conf=conf,
        imgsz=imgsz,
        iou=iou,
        stream=True,
        verbose=False,
    )

    output = []
    vehicle_count = 0
    vehicle_images = 0
    for (item, base_boxes), result in zip(seeded, results, strict=True):
        candidates = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxyn[0].tolist()
            candidates.append({
                "cls": 3,
                "cx": round((x1 + x2) / 2, 6),
                "cy": round((y1 + y2) / 2, 6),
                "w": round(x2 - x1, 6),
                "h": round(y2 - y1, 6),
                "conf": round(float(box.conf[0]), 4),
            })
        boxes = base_boxes + candidates
        upgraded = dict(item)
        upgraded["boxes"] = boxes
        upgraded["n_boxes"] = len(boxes)
        upgraded["multi"] = len(boxes) > 1
        # 先看最不确定的新增候选；没有新增候选的放到最后，但仍需最终确认。
        upgraded["sort_key"] = (
            min(b["conf"] for b in candidates) if candidates else 2.0
        )
        output.append(upgraded)
        vehicle_count += len(candidates)
        vehicle_images += bool(candidates)
    output.sort(key=lambda item: item["sort_key"])
    return output, vehicle_count, vehicle_images


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="旧 3 类 review_manifest.json")
    ap.add_argument("--progress", required=True, help="旧 3 类、已完成人工复核的 progress")
    ap.add_argument("--root", required=True, help="manifest.preview 的根目录")
    ap.add_argument("--vehicle-model", default="yolo11x.pt",
                    help="COCO 模型，仅用于召回 traffic-light 候选")
    ap.add_argument("--coco-class", type=int, default=9)
    ap.add_argument("--conf", type=float, default=0.20)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--iou", type=float, default=0.50)
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-unreviewed", action="store_true")
    ap.add_argument("--force", action="store_true", help="允许覆盖已有输出 manifest")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists() and not args.force:
        print(f"[STOP] 输出已存在，拒绝覆盖: {out}")
        print("       这是人工复核入口；确认要重建时才加 --force。")
        return 2

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    progress = json.loads(Path(args.progress).read_text(encoding="utf-8"))
    ids = [item["id"] for item in manifest]
    if len(ids) != len(set(ids)):
        raise ValueError("manifest 含重复 id，进度保存会互相覆盖")

    seeded = reviewed_seed(manifest, progress, args.allow_unreviewed)
    upgraded, n_vehicle, n_vehicle_images = append_vehicle_candidates(
        seeded,
        Path(args.root),
        args.vehicle_model,
        args.coco_class,
        args.conf,
        args.imgsz,
        args.iou,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(upgraded, ensure_ascii=False, indent=1), encoding="utf-8")

    counts = Counter(int(b["cls"]) for item in upgraded for b in item["boxes"])
    print(f"[OUT] {out}: {len(upgraded)} 图 / {sum(counts.values())} 框")
    print("      " + ", ".join(f"class {k}={counts[k]}" for k in range(4)))
    print(f"      新增 vehicle_signal 候选 {n_vehicle} 框，覆盖 {n_vehicle_images} 图")
    suggested_progress = out.with_name(out.stem.replace("manifest", "progress") + ".json")
    print("\n下一步必须使用全新进度文件（不要复用旧 3 类 progress）：")
    print("  python tools/review/serve_review.py "
          f"--manifest {out} --root {args.root} --progress {suggested_progress}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
