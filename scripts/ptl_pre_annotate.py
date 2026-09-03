#!/usr/bin/env python3
"""T3 Step 1-3 — ImVisible/PTL 半自动预标注。

PTL 数据集本身【没有】信号灯检测框（标注是斑马线中线端点）。
本脚本用 COCO 预训练大模型跑出 traffic-light 候选框，按规则过滤成人行灯候选，
再用该图的 image-level class 赋标签，产出供人工复核界面(tools/review)使用的 manifest。

产出:
  data/interim/ptl/review_manifest.json   # 复核清单(含候选框, 归一化坐标)
  data/interim/ptl/preview/<stem>.jpg      # 供网页复核的降采样预览(max 1280)
  data/interim/ptl/prior_report.md         # --verify-prior 模式的假设验证报告

用法:
  # 先验证假设(附录C)：COCO召回率 + "人行灯在中上部"位置先验
  python scripts/ptl_pre_annotate.py --images-dir data/raw/imvisible/images \
      --verify-prior --limit 50

  # 正式生成 manifest(全量)
  python scripts/ptl_pre_annotate.py --images-dir data/raw/imvisible/images
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
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

REPO = Path(__file__).resolve().parents[1]
CSV_DIR = REPO / "data/raw/imvisible/Annotations"
DEFAULT_CFG = REPO / "configs/ptl_filter.yaml"
OUT_DIR = REPO / "data/interim/ptl"


def load_cfg(path: Path) -> dict:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_imvisible_labels() -> dict[str, dict]:
    """读三份 CSV -> {filename: {mode:int, split:str}}。表头有 BOM。"""
    out: dict[str, dict] = {}
    for split in ("training", "validation", "testing"):
        p = CSV_DIR / f"{split}_file.csv"
        if not p.exists():
            print(f"[WARN] 缺 {p}", file=sys.stderr)
            continue
        with p.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                fn = row["file"].strip()
                out[fn] = {"mode": int(row["mode"]), "split": split}
    return out


def index_images(images_dir: Path) -> dict[str, Path]:
    idx = {}
    if images_dir.exists():
        for p in images_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                idx[p.name] = p
    return idx


def passes_filter(cx, cy, w, h, fcfg) -> tuple[bool, str]:
    area = w * h
    if area < fcfg["min_area_frac"]:
        return False, "too_small"
    if area > fcfg["max_area_frac"]:
        return False, "too_large"
    aspect = w / h if h > 0 else 99
    if not (fcfg["min_aspect"] <= aspect <= fcfg["max_aspect"]):
        return False, "aspect"
    pos = fcfg.get("position", {})
    if pos.get("enabled") and cy > pos.get("max_cy", 1.0):
        return False, "position"
    return True, "ok"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", default=str(REPO / "data/raw/imvisible/images"))
    ap.add_argument("--config", default=str(DEFAULT_CFG))
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 张(调试/验证)")
    ap.add_argument("--verify-prior", action="store_true",
                    help="只做假设验证(COCO召回/位置先验), 不写 manifest")
    ap.add_argument("--preview-max", type=int, default=1280)
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--no-resume", action="store_true", help="不续跑, 从头重来(删除已有 partial)")
    args = ap.parse_args()

    cfg = load_cfg(Path(args.config))
    dcfg, fcfg = cfg["detect"], cfg["filter"]
    cmap = {int(k): int(v) for k, v in cfg["class_map"].items()}

    labels = load_imvisible_labels()
    print(f"[CSV] ImVisible 标注 {len(labels)} 条")
    img_idx = index_images(Path(args.images_dir))
    print(f"[DISK] 图片 {len(img_idx)} 张 @ {args.images_dir}")
    if not img_idx:
        print("[ERR] 未找到图片。请先下载 ImVisible 图片到 --images-dir。", file=sys.stderr)
        return 2

    # 交集：既有标注又有图片文件的
    items = [(fn, meta) for fn, meta in labels.items() if fn in img_idx]
    missing = len(labels) - len(items)
    print(f"[MATCH] 标注∩图片 = {len(items)} 张 (标注有但缺图 {missing} 张)")
    if args.limit:
        items = items[: args.limit]
        print(f"[LIMIT] 只处理前 {len(items)} 张")

    from ultralytics import YOLO
    from PIL import Image
    model = YOLO(dcfg["model"])

    out_dir = Path(args.out)
    prev_dir = out_dir / "preview"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_preview:
        prev_dir.mkdir(exist_ok=True)

    # 假设验证统计
    stat = {
        "cy_all": [], "aspect_all": [], "area_all": [],
        "expect_light_total": 0, "expect_light_recalled": 0,
        "none_total": 0, "none_with_box": 0,
        "n_cand_hist": Counter(),
    }
    tgt_dist = Counter()

    # 增量落盘 + 续跑（崩溃不丢进度）。verify-prior 模式不落盘。
    partial_path = out_dir / "_manifest_partial.jsonl"
    done_ids = set()
    if not args.verify_prior:
        if args.no_resume and partial_path.exists():
            partial_path.unlink()
        if partial_path.exists():
            for line in partial_path.read_text(encoding="utf-8").splitlines():
                try:
                    done_ids.add(json.loads(line)["id"])
                except Exception:
                    pass
            print(f"[RESUME] 已完成 {len(done_ids)} 项, 本次跳过它们")
    partial_f = partial_path.open("a", encoding="utf-8") if not args.verify_prior else None

    n_fail = 0
    for i, (fn, meta) in enumerate(items):
        img_p = img_idx[fn]
        stem = img_p.stem
        mode = meta["mode"]
        target = cmap.get(mode, -1)

        if not args.verify_prior and stem in done_ids:
            continue

        try:
            res_list = model.predict(str(img_p), classes=[dcfg["coco_class"]],
                                     conf=dcfg["conf"], imgsz=dcfg["imgsz"],
                                     iou=dcfg["iou"], verbose=False)
            if not res_list:
                raise RuntimeError("empty predict result")
            res = res_list[0]
            W, H = res.orig_shape[1], res.orig_shape[0]
        except Exception as e:
            n_fail += 1
            print(f"  [SKIP] {fn}: 检测失败 ({e})")
            continue

        boxes = []
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxyn[0].tolist()  # normalized
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            w, h = x2 - x1, y2 - y1
            conf = float(b.conf[0])
            stat["cy_all"].append(cy)
            stat["aspect_all"].append(w / h if h > 0 else 0)
            stat["area_all"].append(w * h)
            ok, _ = passes_filter(cx, cy, w, h, fcfg)
            if ok:
                boxes.append({"cls": target if target >= 0 else 0,
                              "cx": round(cx, 6), "cy": round(cy, 6),
                              "w": round(w, 6), "h": round(h, 6),
                              "conf": round(conf, 4)})

        # 统计
        stat["n_cand_hist"][len(boxes)] += 1
        if mode == 4:
            stat["none_total"] += 1
            if boxes:
                stat["none_with_box"] += 1
        else:
            stat["expect_light_total"] += 1
            if boxes:
                stat["expect_light_recalled"] += 1

        if not args.verify_prior:
            # 排序键：期望有灯却0框 -> 最可疑(最前); 否则按最低框conf; 正确负样本最后
            if target >= 0 and not boxes:
                sort_key = -1.0
            elif target < 0 and not boxes:
                sort_key = 2.0
            else:
                sort_key = min((b["conf"] for b in boxes), default=1.5)

            if not args.no_preview:
                try:
                    im = Image.open(img_p).convert("RGB")
                    im.thumbnail((args.preview_max, args.preview_max))
                    im.save(prev_dir / f"{stem}.jpg", quality=85)
                except Exception as e:
                    n_fail += 1
                    print(f"  [SKIP] {fn}: 预览生成失败 ({e})")
                    continue

            item = {
                "id": stem, "file": fn,
                "preview": f"preview/{stem}.jpg",
                "width": W, "height": H,
                "mode": mode, "target_class": target,
                "boxes": boxes, "n_boxes": len(boxes),
                "sort_key": round(sort_key, 4),
                "multi": len(boxes) > 1,
                "split": meta["split"],
            }
            partial_f.write(json.dumps(item, ensure_ascii=False) + "\n")
            partial_f.flush()
            tgt_dist[target] += 1

        if (i + 1) % 50 == 0:
            print(f"  ... {i+1}/{len(items)}")

    if partial_f:
        partial_f.close()
    if n_fail:
        print(f"[WARN] 跳过 {n_fail} 张坏图/失败图")

    # 输出
    if args.verify_prior:
        _write_prior_report(out_dir, stat, len(items))
    else:
        # 从 partial JSONL 汇总(含本次+历史续跑)
        manifest = []
        for line in partial_path.read_text(encoding="utf-8").splitlines():
            try:
                manifest.append(json.loads(line))
            except Exception:
                pass
        manifest.sort(key=lambda m: m["sort_key"])
        (out_dir / "review_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n[OUT] manifest: {out_dir/'review_manifest.json'} ({len(manifest)} 项)")
        print(f"[OUT] target 分布: " +
              ", ".join(f"{k}={v}" for k, v in sorted(tgt_dist.items())) +
              "  (0=red,1=green,2=unknown,-1=不检测)")
        _write_prior_report(out_dir, stat, len(items))  # 顺便出一份统计
    return 0


def _write_prior_report(out_dir: Path, stat: dict, n: int) -> None:
    import statistics as st
    def pct(v, t): return f"{100*v/t:.1f}%" if t else "n/a"
    cy = stat["cy_all"]; asp = stat["aspect_all"]; ar = stat["area_all"]
    lines = [
        "# T3 假设验证报告 (COCO 召回 + 位置先验)", "",
        f"- 处理图片: {n}",
        f"- COCO 候选框总数: {len(cy)}", "",
        "## 附录C假设: COCO 能召回足够比例人行灯?",
        f"- 期望有灯的图(mode 0-3): {stat['expect_light_total']}，"
        f"其中至少检出1个通过过滤的框: {stat['expect_light_recalled']} "
        f"({pct(stat['expect_light_recalled'], stat['expect_light_total'])})",
        "  → 这个比例就是【图级召回率】。若明显偏低(<70%)，"
        "T3 需改走'先标200张训粗检测器再自举'。", "",
        "## 附录C假设: 'None'图不该有灯",
        f"- mode=4(None)图: {stat['none_total']}，"
        f"其中被检出框: {stat['none_with_box']} "
        f"({pct(stat['none_with_box'], stat['none_total'])}) → 这些是误检候选", "",
        "## 位置先验: '人行灯在画面中上部'成立吗?",
    ]
    if cy:
        lines += [
            f"- 候选框中心 cy 分布(0=顶,1=底): "
            f"min={min(cy):.2f} 中位={st.median(cy):.2f} max={max(cy):.2f}",
            f"- cy<0.5(上半屏)占比: {pct(sum(1 for v in cy if v<0.5), len(cy))}",
            f"- 长宽比 w/h 中位: {st.median(asp):.2f}",
            f"- 面积占比 中位: {st.median(ar):.5f}",
            "",
            "> 判断: 若 cy 中位数明显<0.5 且集中，位置先验成立，可在 "
            "configs/ptl_filter.yaml 里启用 position 过滤；否则保持 enabled:false。",
        ]
    else:
        lines.append("- 无候选框，无法统计。")
    (out_dir / "prior_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[OUT] 假设验证报告: {out_dir/'prior_report.md'}")
    for l in lines:
        if l.startswith("- ") or l.startswith("  →"):
            print("   " + l)


if __name__ == "__main__":
    raise SystemExit(main())
