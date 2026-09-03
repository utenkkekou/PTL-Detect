#!/usr/bin/env python3
"""T5 — 数据泄漏自检 + 跨源去重（§5 / §5.1）。

两个用途：
1. --leak: 计算 train 与 val 之间图片的 pHash 最小汉明距离分布。
   若存在大量高度相似对(距离很小) → 场景分组切分没做对, 指标会虚高, 必须回去重来。
2. --dedup: 跨数据源全局去重。汉明距离 < 阈值的聚为一组, 每组只留一张,
   输出跨源重复报告(哪两个源之间重复多少张)。**必须在分组切分之前做。**

pHash 自实现(DCT), 只依赖 numpy+Pillow, 不引入 imagehash。

用法:
  # 切分后自检
  python scripts/check_leakage.py --leak \
      --train data/yolo/images/train --val data/yolo/images/val

  # 合并多源后、切分前去重
  python scripts/check_leakage.py --dedup \
      --sources imvisible=data/interim/ptl/images ampel=data/interim/ampel_pilot/images \
      --report docs/dedup_report.md --threshold 6
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def phash(path: Path, hash_size: int = 8, highfreq: int = 4) -> int | None:
    """DCT 感知哈希 -> 64-bit 整数。失败返回 None。"""
    try:
        im = Image.open(path).convert("L").resize(
            (hash_size * highfreq, hash_size * highfreq), Image.LANCZOS)
    except Exception:
        return None
    a = np.asarray(im, dtype=np.float32)
    # 2D DCT-II via scipy 无依赖实现：用 numpy 的 FFT 近似太糙，直接手写 DCT 矩阵
    N = a.shape[0]
    k = np.arange(N)
    dct_mat = np.cos(np.pi * (2 * k[:, None] + 1) * k[None, :] / (2 * N))
    dct_mat[0, :] *= 1 / np.sqrt(2)
    dct = dct_mat @ a @ dct_mat.T
    low = dct[:hash_size, :hash_size]
    med = np.median(low[1:].flatten() if low.size > 1 else low.flatten())
    bits = (low > med).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def list_images(d: Path) -> list[Path]:
    return [p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXT]


def hash_dir(d: Path, label: str) -> dict[Path, int]:
    out = {}
    imgs = list_images(d)
    print(f"[{label}] {len(imgs)} 张, 计算 pHash …")
    for i, p in enumerate(imgs):
        h = phash(p)
        if h is not None:
            out[p] = h
        if (i + 1) % 500 == 0:
            print(f"   {i+1}/{len(imgs)}")
    return out


def cmd_leak(args) -> int:
    tr = hash_dir(Path(args.train), "train")
    va = hash_dir(Path(args.val), "val")
    tr_h = list(tr.values())
    dists = []
    near_pairs = []
    for vp, vh in va.items():
        best = min(((hamming(vh, th), tp) for tp, th in tr.items()),
                   default=(64, None))
        dists.append(best[0])
        if best[0] <= args.threshold:
            near_pairs.append((best[0], vp, best[1]))
    if not dists:
        print("[LEAK] 无数据。"); return 1
    dists = np.array(dists)
    print("\n==== 泄漏自检 (val 每张到 train 的最小 pHash 距离) ====")
    for q in (0, 1, 5, 10, 25, 50):
        print(f"  {q:>3} 分位: {np.percentile(dists, q):.0f}")
    leak_n = int((dists <= args.threshold).sum())
    print(f"\n  距离 <= {args.threshold}(疑似泄漏)的 val 图: {leak_n} / {len(dists)} "
          f"({100*leak_n/len(dists):.1f}%)")
    if leak_n > 0:
        print("  ⚠️ 存在高度相似对，分组切分可能没做对。示例：")
        for d, vp, tp in sorted(near_pairs)[:10]:
            print(f"    dist={d}  {vp.name}  ~  {tp.name if tp else '?'}")
        print("  → 若这些确是同场景连拍跨了 train/val，回 split 重来。")
    else:
        print("  ✅ 未发现明显泄漏。")
    return 0


def cmd_dedup(args) -> int:
    # sources: name=path
    srcs = {}
    for spec in args.sources:
        name, _, path = spec.partition("=")
        srcs[name] = Path(path)
    all_h: list[tuple[str, Path, int]] = []
    for name, d in srcs.items():
        for p, h in hash_dir(d, name).items():
            all_h.append((name, p, h))
    print(f"\n[DEDUP] 共 {len(all_h)} 张, 阈值汉明<{args.threshold} 视为重复")

    # 简单聚类：贪心，O(n^2)（几千~上万张可接受；更大需 LSH）
    n = len(all_h)
    used = [False] * n
    groups = []
    for i in range(n):
        if used[i]:
            continue
        grp = [i]; used[i] = True
        for j in range(i + 1, n):
            if not used[j] and hamming(all_h[i][2], all_h[j][2]) < args.threshold:
                grp.append(j); used[j] = True
        groups.append(grp)

    dup_groups = [g for g in groups if len(g) > 1]
    cross = Counter()
    keep, drop = [], []
    for g in groups:
        # 每组保留优先源(参数顺序靠前=质量高)的第一张
        order = {name: k for k, name in enumerate(srcs)}
        g_sorted = sorted(g, key=lambda idx: order.get(all_h[idx][0], 99))
        keep.append(g_sorted[0])
        drop.extend(g_sorted[1:])
        if len(g) > 1:
            names = sorted({all_h[idx][0] for idx in g})
            if len(names) > 1:
                for a in range(len(names)):
                    for b in range(a + 1, len(names)):
                        cross[(names[a], names[b])] += 1

    lines = ["# 跨源去重报告 (T5 §5.1)", "",
             f"- 输入总图: {n}",
             f"- 去重组数: {len(groups)} (其中含重复的组: {len(dup_groups)})",
             f"- 保留: {len(keep)}  丢弃(重复): {len(drop)}",
             f"- 阈值: 汉明距离 < {args.threshold}", "",
             "## 跨源重复对（源A × 源B : 重复张数）"]
    if cross:
        for (a, b), c in cross.most_common():
            lines.append(f"- {a} × {b} : {c}")
    else:
        lines.append("- 无跨源重复。")
    lines += ["", "## 待丢弃文件(前 50)"]
    for idx in drop[:50]:
        lines.append(f"- [{all_h[idx][0]}] {all_h[idx][1]}")
    Path(args.report).write_text("\n".join(lines), encoding="utf-8")
    print(f"[OUT] 报告: {args.report}")
    for l in lines[:12]:
        print("  " + l)
    # 同时输出待丢弃清单供 split 前剔除
    droplist = Path(args.report).with_suffix(".drop.txt")
    droplist.write_text("\n".join(str(all_h[i][1]) for i in drop), encoding="utf-8")
    print(f"[OUT] 待丢弃清单: {droplist}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leak", action="store_true")
    ap.add_argument("--dedup", action="store_true")
    ap.add_argument("--train"); ap.add_argument("--val")
    ap.add_argument("--sources", nargs="+", help="name=path ...")
    ap.add_argument("--report", default="docs/dedup_report.md")
    ap.add_argument("--threshold", type=int, default=6,
                    help="汉明距离阈值(leak: <=视为泄漏; dedup: <视为重复)")
    args = ap.parse_args()
    if args.leak:
        return cmd_leak(args)
    if args.dedup:
        return cmd_dedup(args)
    ap.error("需指定 --leak 或 --dedup")


if __name__ == "__main__":
    raise SystemExit(main())
