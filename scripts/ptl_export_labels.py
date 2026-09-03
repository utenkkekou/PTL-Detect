#!/usr/bin/env python3
"""T3/T10 收尾 — 复核后的 manifest/进度 -> 4 类 YOLO 标签。

读 review_manifest.json + review_progress.json：
- 已复核项(status=reviewed) 用复核后的框；未复核项默认跳过(可 --include-unreviewed 用自动框)
- 负样本(target<0 或 0 框) 写空 .txt
输出：<out>/labels/<stem>.txt（不复制原图）
     <out>/ptl_index.csv（stem, 原图路径, split, n_boxes, target）供 T4/T5 用

同时给出「自动打标准确率」：复核时被改动的比例（衡量 Step1-3 流水线质量）。

用法:
  python scripts/ptl_export_labels.py \
      --images-dir data/interim/mine_t10/frames \
      --manifest data/interim/mine_t10/review_manifest_4cls.json \
      --progress data/interim/mine_t10/review_progress_4cls.json \
      --out data/interim/mine_t10_clean4
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", default="data/interim/mine_t10/frames")
    ap.add_argument("--manifest", default="data/interim/mine_t10/review_manifest_4cls.json")
    ap.add_argument("--progress", default="data/interim/mine_t10/review_progress_4cls.json")
    ap.add_argument("--out", default="data/interim/mine_t10_clean4")
    ap.add_argument("--names",
                    default=("pedestrian_signal_red,pedestrian_signal_green,"
                             "pedestrian_signal_unknown,vehicle_signal"),
                    help="逗号分隔的类别名；类别编号必须落在此范围")
    ap.add_argument("--include-unreviewed", action="store_true",
                    help="未复核项也用自动框导出(默认只导已复核, 保证质量)")
    ap.add_argument("--allow-partial", action="store_true",
                    help="允许只导出已复核子集；默认要求manifest全部复核完成")
    ap.add_argument("--allow-existing", action="store_true",
                    help="允许写入已有 labels 目录；默认拒绝，防止混入旧三类标签")
    args = ap.parse_args()
    names = [s.strip() for s in args.names.split(",") if s.strip()]

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    prog = {}
    if Path(args.progress).exists():
        prog = json.loads(Path(args.progress).read_text(encoding="utf-8"))
    unreviewed = [
        m["id"] for m in manifest
        if not (prog.get(m["id"]) and prog[m["id"]].get("status") == "reviewed")
    ]
    if unreviewed and not args.include_unreviewed and not args.allow_partial:
        print(f"[STOP] 还有 {len(unreviewed)}/{len(manifest)} 项未复核，拒绝生成不完整训练标签。")
        print("       完成复核后重跑；只做调试子集时才加 --allow-partial。")
        return 2

    # 建立原图索引(basename -> 相对路径)
    idx = {}
    for p in Path(args.images_dir).rglob("*"):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            idx[p.name] = p

    out = Path(args.out)
    lbl_dir = out / "labels"
    existing = list(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []
    if existing and not args.allow_existing:
        print(f"[STOP] {lbl_dir} 已有 {len(existing)} 个标签，拒绝与旧数据混写。")
        print("       请换全新 --out；明确续写时才加 --allow-existing。")
        return 2
    lbl_dir.mkdir(parents=True, exist_ok=True)

    cls_cnt = Counter()
    n_pos = n_neg = n_exported = n_skipped = 0
    n_reviewed = n_changed = 0
    rows = []

    for m in manifest:
        stem, fn = m["id"], m["file"]
        p = prog.get(stem)
        reviewed = bool(p and p.get("status") == "reviewed")
        if reviewed:
            n_reviewed += 1
            if p.get("changed"):
                n_changed += 1
            boxes = p.get("boxes", [])
        else:
            if not args.include_unreviewed:
                n_skipped += 1
                continue
            boxes = m["boxes"]

        bad = sorted({int(b["cls"]) for b in boxes
                      if int(b["cls"]) < 0 or int(b["cls"]) >= len(names)})
        if bad:
            raise ValueError(
                f"{stem}: 出现契约外类别 {bad}; 有效范围 0..{len(names)-1} ({names})")

        # 写标签
        lines = [f"{int(b['cls'])} {b['cx']:.6f} {b['cy']:.6f} {b['w']:.6f} {b['h']:.6f}"
                 for b in boxes]
        (lbl_dir / f"{stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        for b in boxes:
            cls_cnt[int(b["cls"])] += 1
        if boxes:
            n_pos += 1
        else:
            n_neg += 1
        n_exported += 1
        src = idx.get(fn)
        rows.append([stem, str(src) if src else "MISSING", m.get("split", ""),
                     len(boxes), m.get("target_class", "")])

    with (out / "ptl_index.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stem", "image_path", "orig_split", "n_boxes", "target"])
        w.writerows(rows)

    print("==== T3 导出结果 ====")
    print(f"导出标签      : {n_exported}  (正样本 {n_pos}, 负样本 {n_neg})")
    print(f"跳过(未复核)  : {n_skipped}"
          + ("" if args.include_unreviewed else "  (加 --include-unreviewed 可含自动框)"))
    print("类别分布(框)  : " + ", ".join(
        f"{k}={names[k]}:{cls_cnt.get(k, 0)}" for k in range(len(names))))
    if n_reviewed:
        acc = 100 * (n_reviewed - n_changed) / n_reviewed
        print(f"自动打标准确率: {acc:.1f}%  (已复核 {n_reviewed} 张, 其中改动 {n_changed} 张)")
    print(f"索引: {out/'ptl_index.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
