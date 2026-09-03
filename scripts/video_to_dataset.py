#!/usr/bin/env python3
"""T10 — 自采视频 -> 复核清单（抽帧 + pHash去重 + 模型预标注）。

两阶段(可分开跑):
  阶段1 --frames-only : ffmpeg 抽帧 + 同视频内 pHash 去重(去掉近似连拍帧)。纯CPU。
  阶段2 (给 --model)  : 对留下的帧跑检测预标注, 生成 review_manifest(复核工具可直接用)。

场景名默认采用完整视频名；只有明确给 ``--merge-numbered-suffix`` 才合并末尾数字。
增量批次可用 ``--exclude-index`` 排除已经抽过的视频。

用法:
  # 阶段1: 抽帧去重
  python scripts/video_to_dataset.py --videos data/raw/mine/videos --out data/interim/mine_t10 --fps 2 --frames-only
  # 阶段2: 预标注(用最好的模型)
  python scripts/video_to_dataset.py --videos data/raw/mine/videos --out data/interim/mine_t10 --model models/best.pt
"""
from __future__ import annotations
import argparse, json, math, re, subprocess, sys
from pathlib import Path
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np
from PIL import Image

VID_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
NAMES_3 = [
    "pedestrian_signal_red",
    "pedestrian_signal_green",
    "pedestrian_signal_unknown",
]
NAMES_4 = NAMES_3 + ["vehicle_signal"]


def model_names(model):
    names = model.names
    if isinstance(names, dict):
        return [names[k] for k in sorted(names, key=lambda x: int(x))]
    return list(names)


def phash(im, hs=8, hf=4):
    try:
        im.draft("L", (64, 64))
        g = im.convert("L").resize((hs*hf, hs*hf), Image.LANCZOS)
    except Exception:
        return None
    a = np.asarray(g, dtype=np.float32); N = a.shape[0]; k = np.arange(N)
    d = np.cos(np.pi*(2*k[:, None]+1)*k[None, :]/(2*N)); d[0, :] *= 1/np.sqrt(2)
    dct = d @ a @ d.T; low = dct[:hs, :hs]
    med = np.median(low[1:].flatten())
    h = 0
    for b in (low > med).flatten():
        h = (h << 1) | int(b)
    return h


def hamming(a, b): return bin(a ^ b).count("1")


def location_of(stem, merge_numbered_suffix=False):
    """视频名作场景。仅显式请求时才把“公司楼下2”并回“公司楼下”。

    默认保留数字，因为“未知路口2/3”更可能是不同路口；误合并虽不泄漏，
    但会让验证集失去大量独立场景。
    """
    return re.sub(r"\d+$", "", stem) if merge_numbered_suffix else stem


def parse_scene_aliases(values):
    aliases = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"--scene-alias 必须是 VIDEO=SCENE: {value!r}")
        video, scene = (part.strip() for part in value.split("=", 1))
        if not video or not scene:
            raise ValueError(f"--scene-alias 两侧都不能为空: {value!r}")
        if video in aliases and aliases[video] != scene:
            raise ValueError(
                f"同一视频给了两个场景: {video!r} -> {aliases[video]!r}/{scene!r}"
            )
        aliases[video] = scene
    return aliases


def load_excluded_videos(index_paths, explicit_videos=()):
    excluded = set(explicit_videos)
    for index_path in index_paths or []:
        records = json.loads(Path(index_path).read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError(f"排除索引不是 JSON 数组: {index_path}")
        for rec in records:
            video = str(rec.get("video", "")).strip()
            if not video:
                raise ValueError(f"排除索引记录缺 video: {index_path}")
            excluded.add(video)
    return excluded


def _validate_unique_video_stems(videos):
    counts = {}
    for video in videos:
        key = video.stem.casefold()
        counts[key] = counts.get(key, 0) + 1
    duplicates = sorted(stem for stem, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(
            f"同一批存在同 stem 的不同视频文件，会覆盖输出: {duplicates}"
        )


def _force_temporal_sample(index, last_kept_index, total, fps,
                           max_gap_seconds):
    if last_kept_index is None:
        return True
    if max_gap_seconds > 0:
        max_gap_frames = max(1, math.ceil(fps * max_gap_seconds))
        if index - last_kept_index >= max_gap_frames:
            return True
    # 保证视频尾段有真实样本；若最后两秒内已有保留帧，则不制造近重复。
    tail_gap_frames = max(1, math.ceil(fps * 2))
    return index == total and index - last_kept_index >= tail_gap_frames


def extract_and_dedup(videos_dir, out, fps, dedup_thr, exclude_videos=None,
                      merge_numbered_suffix=False, scene_aliases=None,
                      max_gap_seconds=10.0):
    frames_root = out / "frames"; frames_root.mkdir(parents=True, exist_ok=True)
    exclude_videos = set(exclude_videos or ())
    scene_aliases = dict(scene_aliases or {})
    vids = sorted(
        p for p in Path(videos_dir).iterdir()
        if p.suffix.lower() in VID_EXT and p.stem not in exclude_videos
    )
    if not vids:
        raise ValueError("没有待处理的新视频（可能都已被 --exclude-index 排除）")
    _validate_unique_video_stems(vids)
    unused_aliases = sorted(set(scene_aliases) - {p.stem for p in vids})
    if unused_aliases:
        raise ValueError(f"--scene-alias 未命中本批视频: {unused_aliases}")
    conflicts = [
        frames_root / video.stem
        for video in vids
        if (frames_root / video.stem).exists()
        and any(
            path.is_file()
            for path in (frames_root / video.stem).rglob("*")
        )
    ]
    if conflicts:
        raise FileExistsError(
            "输出已含待处理视频的帧，拒绝半途覆盖: "
            + ", ".join(str(path) for path in conflicts[:10])
        )
    kept_total = 0
    manifest_stub = []
    for v in vids:
        vdir = frames_root / v.stem
        vdir.mkdir(exist_ok=True)
        # ffmpeg 抽帧
        raw = vdir / "_raw"
        raw.mkdir(exist_ok=True)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(v),
                        "-vf", f"fps={fps}", "-q:v", "3", str(raw / "%05d.jpg")],
                       check=True)
        raws = sorted(raw.glob("*.jpg"))
        if not raws:
            raise RuntimeError(f"ffmpeg 未从 {v.name} 抽出任何帧")
        # 先找出可读帧，使“尾帧保证”落在最后一张实际可读图片上。
        readable = []
        for index, p in enumerate(raws, start=1):
            try:
                with Image.open(p) as image:
                    h = phash(image)
            except Exception:
                continue
            if h is None:
                continue
            readable.append((index, p, h))
        if not readable:
            raise RuntimeError(f"{v.name} 的抽帧图片全部不可读")
        # pHash 去重(与上一张保留帧差异>阈值才留)
        last = None; last_kept_index = None; kept = 0
        last_readable_index = readable[-1][0]
        for index, p, h in readable:
            if (
                last is None
                or (h is not None and hamming(h, last) > dedup_thr)
                or _force_temporal_sample(
                    index, last_kept_index, last_readable_index,
                    fps, max_gap_seconds
                )
            ):
                dst = vdir / f"{v.stem}__{p.stem}.jpg"
                p.replace(dst)
                last = h; last_kept_index = index; kept += 1
                manifest_stub.append({"path": dst, "video": v.stem,
                                      "location": scene_aliases.get(
                                          v.stem,
                                          location_of(
                                              v.stem, merge_numbered_suffix))})
        # 清理 _raw
        for p in raw.glob("*.jpg"): p.unlink()
        raw.rmdir()
        print(f"  {v.name}: {len(raws)} 抽帧 -> 去重后留 {kept}")
        kept_total += kept
    print(f"[T10] 共留 {kept_total} 帧 -> {frames_root}")
    (out / "_frames_index.json").write_text(
        json.dumps([{"path": str(m["path"]), "video": m["video"], "location": m["location"]}
                    for m in manifest_stub], ensure_ascii=False), encoding="utf-8")
    return manifest_stub


def _overlap_over_smaller(a, b):
    """两个归一化 cxcywh 框的 intersection/min(area)，用于去掉同目标松紧重复框。"""
    ax1, ay1, ax2, ay2 = a["cx"]-a["w"]/2, a["cy"]-a["h"]/2, a["cx"]+a["w"]/2, a["cy"]+a["h"]/2
    bx1, by1, bx2, by2 = b["cx"]-b["w"]/2, b["cy"]-b["h"]/2, b["cx"]+b["w"]/2, b["cy"]+b["h"]/2
    iw, ih = max(0.0, min(ax2, bx2)-max(ax1, bx1)), max(0.0, min(ay2, by2)-max(ay1, by1))
    inter = iw * ih
    return inter / max(1e-12, min(a["w"]*a["h"], b["w"]*b["h"]))


def _normalized_box(cls_id, x1, y1, x2, y2, confidence):
    """Round a normalized box without letting rounding move it outside [0, 1]."""
    values = tuple(float(value) for value in (x1, y1, x2, y2, confidence))
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"模型返回非有限数值: {values}")
    x1, y1, x2, y2, confidence = values
    x1, y1 = max(0.0, float(x1)), max(0.0, float(y1))
    x2, y2 = min(1.0, float(x2)), min(1.0, float(y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"模型返回无效归一化框: {(x1, y1, x2, y2)}"
        )
    width = round(x2 - x1, 6)
    height = round(y2 - y1, 6)
    if width <= 0 or height <= 0:
        raise ValueError(
            f"模型框在六位小数精度下退化: {(x1, y1, x2, y2)}"
        )
    cx = round((x1 + x2) / 2, 6)
    cy = round((y1 + y2) / 2, 6)
    # Odd sixth-decimal widths can put a rounded center half a unit outside.
    cx = max(width / 2, min(1 - width / 2, cx))
    cy = max(height / 2, min(1 - height / 2, cy))
    return {
        "cls": int(cls_id),
        "cx": cx,
        "cy": cy,
        "w": width,
        "h": height,
        "conf": round(float(confidence), 4),
    }


def annotate(out, model_path, conf, imgsz, vehicle_model_path=None,
             vehicle_conf=0.20, vehicle_imgsz=960):
    idx = json.loads((out / "_frames_index.json").read_text(encoding="utf-8"))
    from ultralytics import YOLO
    model = YOLO(model_path)
    names = model_names(model)
    if names not in (NAMES_3, NAMES_4):
        raise ValueError(
            "预标注模型类别不受支持；请用旧3类或D19四类专用模型，不能直接用COCO模型。\n"
            f"实际 model.names={names}")
    if names == NAMES_3:
        if vehicle_model_path:
            print("[T10] 主模型为旧3类；将由 COCO 辅助模型补 vehicle_signal 候选。")
        else:
            print("[WARN] 当前是旧3类模型；生成清单后还必须用 "
                  "prepare_four_class_review.py 补 vehicle_signal 候选。")
    vehicle_model = None
    if vehicle_model_path:
        vehicle_model = YOLO(vehicle_model_path)
        if str(vehicle_model.names.get(9, "")).lower() != "traffic light":
            raise ValueError("辅助模型的 class 9 不是 COCO traffic light")
        print(f"[T10] 辅助高召回候选: {vehicle_model_path} class9, conf={vehicle_conf}")
    is_four_class = names == NAMES_4 or vehicle_model is not None
    manifest_name = "review_manifest_4cls.json" if is_four_class else "review_manifest.json"
    progress_name = "review_progress_4cls.json" if is_four_class else "review_progress.json"
    manifest_path = out / manifest_name
    progress_path = out / progress_name
    for path in (manifest_path, progress_path):
        if path.exists():
            raise FileExistsError(f"复核任务文件已存在，拒绝覆盖/重算: {path}")
    print(f"[T10] 预标注 {len(idx)} 帧, model={model_path}")
    manifest = []
    aux_added = 0
    for i, rec in enumerate(idx):
        p = Path(rec["path"])
        if not p.exists():
            raise FileNotFoundError(f"抽帧索引中的图片不存在: {p}")
        res = model.predict(str(p), conf=conf, imgsz=imgsz, verbose=False)[0]
        H, W = res.orig_shape
        boxes = []
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxyn[0].tolist()
            boxes.append(_normalized_box(
                b.cls[0], x1, y1, x2, y2, b.conf[0]))
        if vehicle_model is not None:
            aux = vehicle_model.predict(
                str(p), classes=[9], conf=vehicle_conf, imgsz=vehicle_imgsz,
                verbose=False)[0]
            for b in aux.boxes:
                x1, y1, x2, y2 = b.xyxyn[0].tolist()
                candidate = _normalized_box(
                    3, x1, y1, x2, y2, b.conf[0])
                if any(_overlap_over_smaller(candidate, existing) >= 0.70
                       for existing in boxes):
                    continue
                boxes.append(candidate)
                aux_added += 1
        # 排序: 无框帧(可能漏检近大灯)优先看; 否则低conf优先
        sort_key = -1.0 if not boxes else min(b["conf"] for b in boxes)
        manifest.append({"id": p.stem, "file": p.name,
                         "preview": f"frames/{rec['video']}/{p.name}",
                         "width": W, "height": H, "mode": -1,
                         "target_class": (boxes[0]["cls"] if boxes else -1),
                         "boxes": boxes, "n_boxes": len(boxes),
                         "sort_key": round(sort_key, 4), "multi": len(boxes) > 1,
                         "split": rec["location"]})
        if (i+1) % 50 == 0:
            print(f"  ... {i+1}/{len(idx)}")
    manifest.sort(key=lambda m: m["sort_key"])
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    progress_path.write_text("{}\n", encoding="utf-8")
    from collections import Counter
    cls = Counter(b["cls"] for m in manifest for b in m["boxes"])
    empty = sum(1 for m in manifest if not m["boxes"])
    print(f"[T10] manifest: {len(manifest)} 帧, 无框 {empty}, 框类别 {dict(sorted(cls.items()))}")
    if vehicle_model is not None:
        print(f"[T10] COCO辅助新增且未与主模型重叠: {aux_added} 个 class3 候选")
    print(f"[OUT] {manifest_path}")
    print("[OUT] 复核命令: python tools/review/serve_review.py "
          f"--manifest {manifest_path} --root {out} --progress {progress_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="data/raw/mine/videos")
    ap.add_argument("--out", default="data/interim/mine_t10")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--dedup-thr", type=int, default=8, help="同视频内pHash汉明>此才保留(去近似连拍)")
    ap.add_argument(
        "--max-gap-seconds",
        type=float,
        default=10.0,
        help="近似画面也强制保留的最大时间间隔；并保证视频尾段有样本",
    )
    ap.add_argument("--frames-only", action="store_true")
    ap.add_argument("--model", default=None,
                    help="旧3类或D19四类专用模型；COCO vehicle候选走prepare_four_class_review.py")
    ap.add_argument("--vehicle-model", default=None,
                    help="可选COCO模型(如yolo11x.pt)，追加class9高召回候选并映射为vehicle_signal")
    ap.add_argument("--vehicle-conf", type=float, default=0.20)
    ap.add_argument("--vehicle-imgsz", type=int, default=960)
    ap.add_argument("--exclude-index", action="append", default=[],
                    help="已有 _frames_index.json（可重复）；其中视频名将被排除")
    ap.add_argument("--exclude-video", action="append", default=[],
                    help="额外排除的视频 stem（可重复），用于剔除已确认的同源转码")
    ap.add_argument("--scene-alias", action="append", default=[],
                    help="显式场景映射 VIDEO=SCENE（可重复）；左右/反向同路口应映到同一场景")
    ap.add_argument("--merge-numbered-suffix", action="store_true",
                    help="把视频名末尾数字视为同路口的分段；默认保留为不同场景")
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--imgsz", type=int, default=960)
    args = ap.parse_args()
    if not math.isfinite(args.fps) or args.fps <= 0:
        raise ValueError("--fps 必须大于 0")
    if not math.isfinite(args.max_gap_seconds) or args.max_gap_seconds < 0:
        raise ValueError("--max-gap-seconds 不能小于 0；用 0 关闭周期保留")
    out = Path(args.out)
    exclude_videos = load_excluded_videos(
        args.exclude_index, args.exclude_video)
    scene_aliases = parse_scene_aliases(args.scene_alias)
    index_exists = (out / "_frames_index.json").exists()
    if index_exists and (args.scene_alias or args.merge_numbered_suffix):
        raise ValueError(
            "抽帧索引已存在，场景映射参数不会被静默重写；"
            "请在首次抽帧时设置 --scene-alias/--merge-numbered-suffix"
        )
    if args.frames_only and index_exists:
        raise FileExistsError(
            f"抽帧索引已存在，拒绝用 --frames-only 重建/覆盖: {out/'_frames_index.json'}"
        )
    if not index_exists:
        extract_and_dedup(
            Path(args.videos), out, args.fps, args.dedup_thr,
            exclude_videos=exclude_videos,
            merge_numbered_suffix=args.merge_numbered_suffix,
            scene_aliases=scene_aliases,
            max_gap_seconds=args.max_gap_seconds)
    if not args.frames_only:
        if not args.model:
            print("[T10] 未给 --model, 只抽帧。预标注请加 --model")
            return 0
        annotate(out, args.model, args.conf, args.imgsz,
                 args.vehicle_model, args.vehicle_conf, args.vehicle_imgsz)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
