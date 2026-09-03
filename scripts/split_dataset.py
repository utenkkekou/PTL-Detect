#!/usr/bin/env python3
"""T5 — 按场景/路口分组切分（不能按图片随机切！）。

PTL 是在路口连拍的，同路口大量近似图。按图片随机切会让同场景照片同时进
train/val，验证指标虚高、一上真机就崩。必须【按场景分组】切分。

分组信号优先级：
  1. T10 文件名中的显式路口提示（<视频名>__<帧号>）
  2. EXIF GPS 坐标聚类 —— 同路口 GPS 接近
  3. 无 GPS 的：贡献者前缀 + 拍摄时间聚类（同人同场次时间接近）
  4. 文件名 IMG 序号连续性兜底

切完按 §5 做泄漏自检（调用 check_leakage.py）。

用法:
  # 先干跑看分组是否合理(不复制文件)
  python scripts/split_dataset.py --images-dir data/interim/mine_t10/frames --dry-run
  # 正式切分(读 labels, 复制到全新 clean4 目录)
  python scripts/split_dataset.py --images-dir data/interim/mine_t10/frames \
      --labels-dir data/interim/mine_t10_clean4/labels --out data/yolo_clean4
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

IMG_EXT = {".jpg", ".jpeg", ".png"}
EXPECTED_NAMES = [
    "pedestrian_signal_red",
    "pedestrian_signal_green",
    "pedestrian_signal_unknown",
    "vehicle_signal",
]


def dms_to_deg(dms, ref) -> float | None:
    try:
        d, m, s = [float(x) for x in dms]
        val = d + m / 60 + s / 3600
        if ref in ("S", "W"):
            val = -val
        return val
    except Exception:
        return None


import numpy as np


def _phash(im, hash_size=8, highfreq=4):
    """DCT 感知哈希 -> 64-bit 整数(接收已打开的 PIL Image)。失败返回 None。"""
    try:
        try:
            im.draft("L", (64, 64))  # JPEG 以约1/8分辨率快速解码, pHash 够用
        except Exception:
            pass
        g = im.convert("L").resize((hash_size*highfreq, hash_size*highfreq), Image.LANCZOS)
    except Exception:
        return None
    a = np.asarray(g, dtype=np.float32); N = a.shape[0]
    k = np.arange(N)
    d = np.cos(np.pi * (2*k[:, None] + 1) * k[None, :] / (2*N)); d[0, :] *= 1/np.sqrt(2)
    dct = d @ a @ d.T
    low = dct[:hash_size, :hash_size]
    med = np.median(low[1:].flatten() if low.size > 1 else low.flatten())
    h = 0
    for b in (low > med).flatten():
        h = (h << 1) | int(b)
    return h


def load_scene_hints(index_paths: list[str] | None) -> dict[str, str]:
    """读取抽帧时确认的“帧 stem -> 实际场景”映射。"""
    hints: dict[str, str] = {}
    for index_path in index_paths or []:
        records = json.loads(Path(index_path).read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError(f"抽帧索引不是 JSON 数组: {index_path}")
        for rec in records:
            raw_path = str(rec.get("path", "")).replace("\\", "/")
            stem = Path(raw_path).stem
            location = str(rec.get("location", "")).strip()
            if not stem or not location:
                raise ValueError(f"抽帧索引缺 path/location: {index_path}")
            previous = hints.get(stem)
            if previous is not None and previous != location:
                raise ValueError(
                    f"同一帧对应两个场景: {stem} -> {previous!r}/{location!r}"
                )
            hints[stem] = location
    return hints


def extract_features(p: Path, scene_hints: dict[str, str] | None = None) -> dict:
    """EXIF: lat/lon(十进制)、拍摄时间；文件名: 贡献者、IMG序号；pHash 近重复用。"""
    feat = {"path": p, "lat": None, "lon": None, "dt": None,
            "contrib": "unknown", "seq": None, "phash": None,
            "scene_hint": None}
    # 文件名: [contrib_]IMG_<seq>.ext
    stem = p.stem
    import re
    m = re.match(r"(?:([a-zA-Z]+)_)?IMG[_-]?(\d+)", stem)
    if m:
        feat["contrib"] = m.group(1) or "noprefix"
        feat["seq"] = int(m.group(2))
    # 优先采用抽帧索引中的 location。没有索引时保留完整视频名，不擅自把
    # “未知路口2..9”一类确实不同的场景合并。
    if "__" in stem:
        feat["scene_hint"] = (
            (scene_hints or {}).get(stem)
            or stem.split("__", 1)[0]
        )
    try:
        im = Image.open(p)
        try:
            exif = im._getexif() or {}     # 先读 EXIF(不受 draft 影响)
        except Exception:
            exif = {}
        feat["phash"] = _phash(im)         # 再算 pHash(会 draft 降采样解码)
    except Exception:
        exif = {}
    tags = {TAGS.get(k, k): v for k, v in exif.items()}
    dt = tags.get("DateTimeOriginal") or tags.get("DateTime")
    if dt:
        try:
            feat["dt"] = datetime.strptime(dt, "%Y:%m:%d %H:%M:%S")
        except Exception:
            pass
    gps = tags.get("GPSInfo")
    if gps:
        g = {GPSTAGS.get(k, k): v for k, v in gps.items()}
        feat["lat"] = dms_to_deg(g.get("GPSLatitude"), g.get("GPSLatitudeRef"))
        feat["lon"] = dms_to_deg(g.get("GPSLongitude"), g.get("GPSLongitudeRef"))
    return feat


class UF:
    def __init__(self, n): self.p = list(range(n))
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b): self.p[self.find(a)] = self.find(b)


def group_scenes(feats: list[dict], gps_m: float, time_min: float,
                 phash_thr: int = 8) -> list[int]:
    """返回每张图的场景组号。GPS 近 / 无GPS按贡献者+时间 / pHash近重复 都并入同一场景。"""
    n = len(feats)
    uf = UF(n)

    # 0. 自采视频显式路口提示（最可靠）：同一路口的所有视频/帧先并组。
    by_hint = defaultdict(list)
    for i, f in enumerate(feats):
        if f.get("scene_hint"):
            by_hint[f["scene_hint"]].append(i)
    for members in by_hint.values():
        for i in members[1:]:
            uf.union(members[0], i)
    if by_hint:
        print(f"[T5] 文件名路口提示: {len(by_hint)} 组（<视频名>__<帧号>）")

    # 1. GPS 聚类：网格分桶(≈gps_m 米)后桶内+邻桶两两比距离
    #    上海纬度~31.2°, 1°lat≈111000m, 1°lon≈95000m
    LAT_M, LON_M = 111000.0, 95000.0
    cell = gps_m
    grid = defaultdict(list)
    gps_idx = [i for i, f in enumerate(feats) if f["lat"] is not None]
    for i in gps_idx:
        gx = int(feats[i]["lon"] * LON_M / cell)
        gy = int(feats[i]["lat"] * LAT_M / cell)
        grid[(gx, gy)].append(i)
    for (gx, gy), members in grid.items():
        neigh = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                neigh += grid.get((gx + dx, gy + dy), [])
        for a in members:
            for b in neigh:
                if a < b:
                    dlat = (feats[a]["lat"] - feats[b]["lat"]) * LAT_M
                    dlon = (feats[a]["lon"] - feats[b]["lon"]) * LON_M
                    if math.hypot(dlat, dlon) <= gps_m:
                        uf.union(a, b)

    # 2. 无 GPS：同贡献者 + 时间接近 合并；也把无GPS的挂到时间最近的同贡献者GPS图上
    by_contrib = defaultdict(list)
    for i, f in enumerate(feats):
        by_contrib[f["contrib"]].append(i)
    for contrib, idxs in by_contrib.items():
        timed = sorted([i for i in idxs if feats[i]["dt"]], key=lambda i: feats[i]["dt"])
        for a, b in zip(timed, timed[1:]):
            gap = (feats[b]["dt"] - feats[a]["dt"]).total_seconds() / 60
            # 同贡献者、时间间隔 < time_min，且至少一方无GPS(有GPS的已按位置聚过)
            if gap <= time_min and (feats[a]["lat"] is None or feats[b]["lat"] is None):
                uf.union(a, b)
        # 3. 序号连续兜底：都无时间的，按 IMG 序号相邻合并
        seqd = sorted([i for i in idxs if feats[i]["seq"] is not None and feats[i]["dt"] is None],
                      key=lambda i: feats[i]["seq"])
        for a, b in zip(seqd, seqd[1:]):
            if feats[b]["seq"] - feats[a]["seq"] <= 3:
                uf.union(a, b)

    # 4. pHash 近重复合并(兜底, §5.1)：跨贡献者/无GPS 的近重复靠这个抓
    #    汉明距离 <= phash_thr 的并入同场景, 防止近重复图跨 train/val 泄漏。
    if phash_thr > 0:
        valid = [i for i in range(n) if feats[i]["phash"] is not None]
        if valid:
            arr = np.array([feats[i]["phash"] for i in valid], dtype=np.uint64)
            merged = 0
            for r, i in enumerate(valid):
                pc = np.unpackbits((arr ^ arr[r]).view(np.uint8)).reshape(-1, 64).sum(axis=1)
                for c in np.where(pc <= phash_thr)[0]:
                    j = valid[int(c)]
                    if j > i and uf.find(i) != uf.find(j):
                        uf.union(i, j); merged += 1
            print(f"[T5] pHash 近重复合并: {merged} 对 (阈值汉明<= {phash_thr})")

    roots = [uf.find(i) for i in range(n)]
    # 归一化组号
    remap = {r: k for k, r in enumerate(sorted(set(roots)))}
    return [remap[r] for r in roots]


def assign_splits(scene_of: list[int], ratios=(0.70, 0.15, 0.15), seed=42,
                  class_counts: list[Counter] | None = None):
    """整场景分配到 train/val/test，兼顾容量和可行的类别覆盖。

    ``class_counts`` 与图片一一对应。某类若至少出现在三个独立场景中，就优先
    保证三个 split 都覆盖该类；场景原子性始终高于凑图片比例。
    """
    import itertools
    import random

    if class_counts is not None and len(class_counts) != len(scene_of):
        raise ValueError("class_counts 必须与 scene_of 等长")

    scene_sizes = Counter(scene_of)
    scenes = list(scene_sizes)
    total = len(scene_of)
    split_names = ("train", "val", "test")
    targets = tuple(r * total for r in ratios)

    scene_class_counts = {scene: Counter() for scene in scenes}
    if class_counts is not None:
        for scene, counts in zip(scene_of, class_counts):
            scene_class_counts[scene].update(counts)
    total_class_counts = Counter()
    for counts in scene_class_counts.values():
        total_class_counts.update(counts)
    required_classes = {
        cls for cls in total_class_counts
        if sum(scene_class_counts[scene][cls] > 0 for scene in scenes) >= 3
    }

    def size_objective(counts):
        # 归一化平方误差：val/test 的 10 张偏差不能被 train 的大目标淹没。
        return sum(((counts[i] - targets[i]) / max(1.0, targets[i])) ** 2
                   for i in range(3))

    def class_objective(split_class_counts):
        if not total_class_counts:
            return 0.0
        errors = []
        for cls, total_for_class in total_class_counts.items():
            for split_idx in range(3):
                target = ratios[split_idx] * total_for_class
                errors.append(
                    ((split_class_counts[split_idx][cls] - target)
                     / max(1.0, target)) ** 2
                )
        return sum(errors) / len(errors)

    def missing_coverage(split_class_counts):
        return sum(
            split_class_counts[split_idx][cls] == 0
            for cls in required_classes
            for split_idx in range(3)
        )

    def assignment_counts(assignment):
        image_counts = [0, 0, 0]
        split_class_counts = [Counter(), Counter(), Counter()]
        for scene, split_idx in zip(scenes, assignment):
            image_counts[split_idx] += scene_sizes[scene]
            split_class_counts[split_idx].update(scene_class_counts[scene])
        return image_counts, split_class_counts

    # 少量路口时完整枚举（最大 3^12=531441），先满足类别覆盖，再求容量最优。
    if len(scenes) <= 12:
        best_covered = None
        best_any = None
        for assignment in itertools.product(range(3), repeat=len(scenes)):
            if len(scenes) >= 3 and len(set(assignment)) < 3:
                continue
            counts, split_class_counts = assignment_counts(assignment)
            candidate = (
                size_objective(counts),
                class_objective(split_class_counts),
                tuple(abs(counts[i] - targets[i]) for i in range(3)),
                assignment,
                counts,
            )
            if best_any is None or candidate[:3] < best_any[:3]:
                best_any = candidate
            if missing_coverage(split_class_counts) == 0:
                if best_covered is None or candidate[:3] < best_covered[:3]:
                    best_covered = candidate
        best = best_covered or best_any
        assignment, counts = best[3], best[4]
        scene_split = {
            scene: split_names[assignment[i]]
            for i, scene in enumerate(scenes)
        }
        return scene_split, dict(zip(split_names, counts))

    # 场景较多时使用确定性的多起点贪心，并把最终缺类数作为第一排序键。
    def greedy(order):
        counts = [0, 0, 0]
        split_class_counts = [Counter(), Counter(), Counter()]
        mapping = {}
        for scene in order:
            size = scene_sizes[scene]
            choices = []
            for split_idx in range(3):
                nxt = counts.copy()
                nxt[split_idx] += size
                nxt_classes = [c.copy() for c in split_class_counts]
                nxt_classes[split_idx].update(scene_class_counts[scene])
                coverage_gain = sum(
                    scene_class_counts[scene][cls] > 0
                    and split_class_counts[split_idx][cls] == 0
                    for cls in required_classes
                )
                choices.append((
                    -coverage_gain,
                    size_objective(nxt),
                    class_objective(nxt_classes),
                    split_idx,
                    nxt,
                    nxt_classes,
                ))
            choice = min(choices, key=lambda item: item[:4])
            split_idx = choice[3]
            counts = choice[4]
            split_class_counts = choice[5]
            mapping[scene] = split_names[split_idx]
        rank = (
            missing_coverage(split_class_counts),
            size_objective(counts),
            class_objective(split_class_counts),
            tuple(abs(counts[i] - targets[i]) for i in range(3)),
        )
        return rank, mapping, counts

    base_order = sorted(scenes, key=lambda s: (-scene_sizes[s], s))
    best = greedy(base_order)
    rng = random.Random(seed)
    for _ in range(1024):
        order = scenes.copy()
        rng.shuffle(order)
        candidate = greedy(order)
        if candidate[0] < best[0]:
            best = candidate
    _, scene_split, counts = best
    return scene_split, dict(zip(split_names, counts))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", action="append", required=True,
                    help="图片根目录；多批数据可重复给")
    ap.add_argument("--labels-dir", action="append", default=[],
                    help="YOLO标签目录；可重复给，按图片 stem 唯一匹配")
    ap.add_argument("--scene-index", action="append", default=[],
                    help=("抽帧 _frames_index.json，可重复给多批；其中 location 是场景真值。"
                          "未提供时保留完整视频名，不自动去掉末尾数字"))
    ap.add_argument("--out", default="data/yolo")
    ap.add_argument("--gps-m", type=float, default=40, help="同路口GPS距离阈值(米)")
    ap.add_argument("--time-min", type=float, default=20, help="无GPS时同场次时间阈值(分钟)")
    ap.add_argument("--phash-thr", type=int, default=0,
                    help="pHash并入同场景阈值; 默认0关闭(宽阈值会链式塌缩成巨型场景)")
    ap.add_argument("--leak-thr", type=int, default=0,
                    help=("可选的切分后pHash消解阈值；命中时整场景移入train以保持原子性。"
                          "默认0关闭（跨路口pHash会误碰撞，优先人工审计）"))
    ap.add_argument("--dry-run", action="store_true", help="只分组+报告, 不复制文件")
    ap.add_argument("--link", action="store_true", help="用硬链接代替复制(省空间; 与--resize互斥)")
    ap.add_argument("--resize", type=int, default=1280,
                    help="训练图最长边(全分辨率I/O太重); 0=保持原尺寸。标签归一化不受影响")
    ap.add_argument("--names",
                    default=("pedestrian_signal_red,pedestrian_signal_green,"
                             "pedestrian_signal_unknown,vehicle_signal"),
                    help="data.yaml 类名(逗号分隔; 顺序即类别索引)")
    ap.add_argument("--allow-existing", action="store_true",
                    help="允许写入已有 images/labels 输出（危险：改切分后可能残留旧文件）")
    ap.add_argument("--allow-missing-labels", action="store_true",
                    help="缺标签文件时写空标签；默认报错，防止把漏导标签误当负样本")
    ap.add_argument("--allow-noncanonical-classes", action="store_true",
                    help="仅用于复现旧实验；默认要求 D19 四类顺序")
    args = ap.parse_args()
    names = [s.strip() for s in args.names.split(",") if s.strip()]
    if names != EXPECTED_NAMES and not args.allow_noncanonical_classes:
        print("[ERR] --names 不是当前 D19 四类契约，拒绝生成训练集。")
        print("      实际:", names)
        print("      期望:", EXPECTED_NAMES)
        return 2

    image_roots = [Path(path) for path in args.images_dir]
    imgs = sorted(
        (
            p for root in image_roots for p in root.rglob("*")
            if p.suffix.lower() in IMG_EXT
        ),
        key=lambda p: (p.name, p.as_posix()),
    )
    if not imgs:
        print(f"[ERR] {args.images_dir} 下没有图片")
        return 2
    name_counts = Counter(p.name for p in imgs)
    duplicate_names = sorted(name for name, count in name_counts.items() if count > 1)
    if duplicate_names:
        print(f"[ERR] 有 {len(duplicate_names)} 个重复图片文件名；当前输出会扁平化，可能互相覆盖。")
        for name in duplicate_names[:10]:
            print(f"      {name}")
        return 2

    try:
        scene_hints = load_scene_hints(args.scene_index)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[ERR] 读取 --scene-index 失败: {exc}")
        return 2
    if scene_hints:
        unknown_stems = sorted(p.stem for p in imgs if p.stem not in scene_hints)
        if unknown_stems:
            print(f"[ERR] --scene-index 未覆盖 {len(unknown_stems)} 张输入图片，拒绝混用猜测场景。")
            for stem in unknown_stems[:10]:
                print(f"      {stem}")
            return 2
        print(f"[T5] 场景索引覆盖 {len(imgs)} 张输入图: {len(args.scene_index)} 个索引")

    lbl_dirs = [Path(path) for path in args.labels_dir]
    if len({path.resolve() for path in lbl_dirs}) != len(lbl_dirs):
        print("[ERR] --labels-dir 含重复目录")
        return 2
    label_paths: list[Path | None] = []
    class_counts = None
    if lbl_dirs:
        class_counts = []
        missing = []
        invalid = []
        for p in imgs:
            matches = [
                directory / f"{p.stem}.txt"
                for directory in lbl_dirs
                if (directory / f"{p.stem}.txt").exists()
            ]
            if len(matches) > 1:
                invalid.append(
                    f"{p.name}: 在多个 --labels-dir 中存在同名标签 "
                    + ", ".join(str(path) for path in matches)
                )
                lp = None
            else:
                lp = matches[0] if matches else None
            counts = Counter()
            if lp is None:
                missing.append(p)
            else:
                for line_no, line in enumerate(
                        lp.read_text(encoding="utf-8").splitlines(), start=1):
                    parts = line.split()
                    if not parts:
                        continue
                    try:
                        cls = int(parts[0])
                    except ValueError:
                        invalid.append(f"{lp.name}:{line_no} 类别不是整数")
                        continue
                    if cls < 0 or cls >= len(names):
                        invalid.append(f"{lp.name}:{line_no} 类别 {cls} 越界")
                        continue
                    counts[cls] += 1
            label_paths.append(lp)
            class_counts.append(counts)
        if invalid:
            print(f"[ERR] 标签中有 {len(invalid)} 个非法类别，已停止。")
            for item in invalid[:10]:
                print(f"      {item}")
            return 2
        if missing and not args.allow_missing_labels:
            print(f"[ERR] {len(missing)} 张图缺 YOLO 标签文件；为防止把漏标误当负样本，已停止。")
            for p in missing[:10]:
                print(f"      {p.name}")
            print("      确认这些全是真负样本时才加 --allow-missing-labels。")
            return 2

    print(f"[T5] {len(imgs)} 张图, 提取 EXIF 特征 …")
    feats = []
    for i, p in enumerate(imgs):
        feats.append(extract_features(p, scene_hints))
        if (i + 1) % 1000 == 0:
            print(f"   {i+1}/{len(imgs)}")

    n_gps = sum(1 for f in feats if f["lat"] is not None)
    n_dt = sum(1 for f in feats if f["dt"])
    print(f"[T5] 有GPS {n_gps} ({100*n_gps/len(feats):.0f}%), 有时间戳 {n_dt} ({100*n_dt/len(feats):.0f}%)")

    scene_of = group_scenes(feats, args.gps_m, args.time_min, args.phash_thr)
    n_scenes = len(set(scene_of))
    sizes = Counter(scene_of)
    big = sizes.most_common(5)
    print(f"[T5] 分出 {n_scenes} 个场景 (每图均摊 {len(feats)/n_scenes:.1f} 张/场景)")
    print(f"     最大场景: {[c for _,c in big]} 张")
    singl = sum(1 for c in sizes.values() if c == 1)
    print(f"     单图场景: {singl} 个 ({100*singl/n_scenes:.0f}%)")

    scene_split, cur = assign_splits(scene_of, class_counts=class_counts)
    split_of = [scene_split[scene_of[i]] for i in range(len(imgs))]

    # 泄漏消解（可选）: 若 val/test 某图与 train pHash 过近，移动它所属的【整个场景】。
    # 绝不能只移动单帧，否则同一路口会被拆到 train 和 val/test，反而制造真泄漏。
    if args.leak_thr > 0:
        phs = [f["phash"] for f in feats]
        vi = [i for i in range(len(feats)) if phs[i] is not None]
        moved_images = 0
        moved_scenes = set()
        for _ in range(10):
            train_arr = np.array([phs[i] for i in vi if split_of[i] == "train"], dtype=np.uint64)
            if not len(train_arr):
                break
            hit_scenes = set()
            for i in vi:
                if split_of[i] == "train":
                    continue
                pc = np.unpackbits((train_arr ^ phs[i]).view(np.uint8)).reshape(-1, 64).sum(axis=1)
                if pc.min() <= args.leak_thr:
                    hit_scenes.add(scene_of[i])
            hit_scenes -= moved_scenes
            if not hit_scenes:
                break
            step = 0
            for i, scene in enumerate(scene_of):
                if scene in hit_scenes and split_of[i] != "train":
                    split_of[i] = "train"
                    step += 1
            moved_images += step
            moved_scenes.update(hit_scenes)
        print(f"[T5] 泄漏消解: 整场景移入train {len(moved_scenes)} 组/{moved_images} 张 "
              f"(pHash汉明<= {args.leak_thr})")

    cur = Counter(split_of)
    print(f"[T5] 最终切分: train={cur['train']} val={cur['val']} test={cur['test']} "
          f"({100*cur['train']/len(imgs):.0f}/{100*cur['val']/len(imgs):.0f}/{100*cur['test']/len(imgs):.0f})")
    if class_counts is not None:
        split_classes = {split: Counter() for split in ("train", "val", "test")}
        for split, counts in zip(split_of, class_counts):
            split_classes[split].update(counts)
        print("[T5] 各切分类别框数: " + ", ".join(
            f"{split}={dict(sorted(split_classes[split].items()))}"
            for split in ("train", "val", "test")
        ))

    if args.dry_run:
        print("[T5] --dry-run: 不复制文件。分组合理即可正式切分。")
        return 0

    # 正式切分：复制/链接/降采样 图片 + 标签
    out = Path(args.out)
    if not lbl_dirs:
        print("[ERR] 正式切分必须给 --labels-dir；负样本也应有明确的空 .txt。")
        return 2
    existing = [
        p for branch in (out / "images", out / "labels")
        if branch.exists() for p in branch.rglob("*") if p.is_file()
    ]
    if existing and not args.allow_existing:
        print(f"[ERR] 输出目录已有 {len(existing)} 个数据文件: {out}")
        print("      请换一个全新 --out；确认要增量写入时才加 --allow-existing。")
        return 2
    counts = Counter()
    for i, p in enumerate(imgs):
        split = split_of[i]
        di = out / "images" / split; di.mkdir(parents=True, exist_ok=True)
        dst = di / p.name
        if not dst.exists():
            if args.resize and args.resize > 0:
                im = Image.open(p).convert("RGB")
                if max(im.size) > args.resize:
                    im.thumbnail((args.resize, args.resize), Image.LANCZOS)
                im.save(dst, quality=92)
            else:
                (shutil.copy2 if not args.link else _link)(p, dst)
        if lbl_dirs:
            lp = label_paths[i]
            dl = out / "labels" / split; dl.mkdir(parents=True, exist_ok=True)
            if lp is not None:
                shutil.copy2(lp, dl / lp.name)
            else:
                # 只有显式 --allow-missing-labels 才会走到这里。
                (dl / f"{p.stem}.txt").write_text("", encoding="utf-8")
        counts[split] += 1
        if (i + 1) % 500 == 0:
            print(f"   写出 {i+1}/{len(imgs)}")

    # data.yaml
    yaml_txt = ("# T4/T5 自动生成\n"
                f"path: {out.resolve().as_posix()}\n"
                "train: images/train\nval: images/val\ntest: images/test\n"
                f"nc: {len(names)}\n"
                "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)))
    (out / "data.yaml").write_text(yaml_txt, encoding="utf-8")
    print(f"[T5] 已写出: {dict(counts)} -> {out}")
    print(f"[T5] data.yaml: {out/'data.yaml'} (nc={len(names)}, names={names})")
    print("[T5] ⚠️ 记得跑泄漏自检: python scripts/check_leakage.py --leak "
          f"--train {out}/images/train --val {out}/images/val")
    return 0


def _link(src, dst):
    import os
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


if __name__ == "__main__":
    raise SystemExit(main())
