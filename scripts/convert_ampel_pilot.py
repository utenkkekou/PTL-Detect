#!/usr/bin/env python3
"""T2 — Ampel-Pilot MySQL dump -> YOLO detection labels.

设计前提（均已实测确认，非推测）：
- `images` 表：(id, file_name[UUID], file_ext, file_size, img_width, img_height, ...)。
  file_size 恒为 0，图片不在 SQL 内，需从 S3 分卷 zip 下载后放到 --images-dir。
- `objects` 表：(id, image_id, class, x, y, width, height, ...)。
  README 明确：x,y 是框中心，w/h 相对图像尺寸归一化 —— 即已经是 YOLO 格式。
  class 实测直接为 0=Red / 1=Green（与本项目 v1 的 0=ptl_red / 1=ptl_green 一致）。

不起 MySQL，用正则逐条解析 INSERT。
用法：
  python scripts/convert_ampel_pilot.py \
      --sql data/raw/ampel_pilot/database/ampelpilot_data.sql \
      --images-dir data/raw/ampel_pilot/images \
      --out data/interim/ampel_pilot \
      --viz 30
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

# Windows 控制台默认 GBK，中文输出会乱码；强制 UTF-8
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---- SQL 解析 -------------------------------------------------------------

# images: (id,'fname','ext',size,w,h,'ts','ts')
IMG_TUPLE = re.compile(
    r"\((\d+),'([^']*)','([^']*)',(\d+),(\d+),(\d+),'[^']*','[^']*'\)"
)
# objects: (id,image_id,class,x,y,w,h,'ts','ts')
OBJ_TUPLE = re.compile(
    r"\((\d+),(\d+),(-?\d+),"
    r"([-\d.eE]+),([-\d.eE]+),([-\d.eE]+),([-\d.eE]+),"
    r"'[^']*','[^']*'\)"
)


def _insert_blocks(sql_text: str, table: str) -> str:
    """把某表的所有 INSERT ... VALUES ...; 语句拼在一起返回。"""
    chunks = []
    pattern = re.compile(
        r"INSERT INTO `" + re.escape(table) + r"` VALUES (.*?);",
        re.DOTALL,
    )
    for m in pattern.finditer(sql_text):
        chunks.append(m.group(1))
    return "\n".join(chunks)


def parse_images(sql_text: str) -> dict[int, dict]:
    block = _insert_blocks(sql_text, "images")
    out: dict[int, dict] = {}
    for m in IMG_TUPLE.finditer(block):
        iid = int(m.group(1))
        out[iid] = {
            "file_name": m.group(2),
            "ext": m.group(3),
            "w": int(m.group(5)),
            "h": int(m.group(6)),
        }
    return out


def parse_objects(sql_text: str) -> dict[int, list[tuple]]:
    block = _insert_blocks(sql_text, "objects")
    out: dict[int, list[tuple]] = {}
    for m in OBJ_TUPLE.finditer(block):
        image_id = int(m.group(2))
        cls = int(m.group(3))
        x, y, w, h = (float(m.group(i)) for i in range(4, 8))
        out.setdefault(image_id, []).append((cls, x, y, w, h))
    return out


# ---- 图片索引 -------------------------------------------------------------

def index_images_dir(images_dir: Path) -> dict[str, Path]:
    """建立 UUID(不含扩展名) -> 实际文件路径 的索引，兼容大小写扩展名。"""
    idx: dict[str, Path] = {}
    if not images_dir.exists():
        return idx
    for p in images_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            idx[p.stem] = p
    return idx


# ---- 主流程 ---------------------------------------------------------------

def convert(args) -> int:
    sql_path = Path(args.sql)
    if not sql_path.exists():
        print(f"[ERR] SQL 不存在: {sql_path}", file=sys.stderr)
        return 2
    sql_text = sql_path.read_text(encoding="utf-8", errors="replace")

    images = parse_images(sql_text)
    objects = parse_objects(sql_text)
    print(f"[SQL] images 表 {len(images)} 行, objects 表 "
          f"{sum(len(v) for v in objects.values())} 个标注 "
          f"(涉及 {len(objects)} 张图)")

    out_dir = Path(args.out)
    out_img = out_dir / "images"
    out_lbl = out_dir / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    disk_idx = index_images_dir(Path(args.images_dir))
    print(f"[DISK] --images-dir 找到 {len(disk_idx)} 个图片文件")
    if not disk_idx:
        print("[WARN] 未找到任何图片文件。请先下载 Ampel 图片到 --images-dir。\n"
              "       脚本已验证 SQL 解析正确，图片到位后重跑即可。")

    cls_counter = Counter()
    written_labels = 0
    copied_images = 0
    missing = 0
    coord_warn = 0
    neg_samples = 0  # 有图但无标注

    # 遍历所有图片记录
    for iid, meta in images.items():
        stem = meta["file_name"]
        boxes = objects.get(iid, [])
        src = disk_idx.get(stem)
        if src is None:
            missing += 1
            continue

        # 写标签
        lines = []
        for (cls, x, y, w, h) in boxes:
            # 已归一化；仅做防御性 clamp + 合法性检查
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0
                    and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                coord_warn += 1
            xx = min(max(x, 0.0), 1.0)
            yy = min(max(y, 0.0), 1.0)
            ww = min(max(w, 1e-6), 1.0)
            hh = min(max(h, 1e-6), 1.0)
            lines.append(f"{cls} {xx:.6f} {yy:.6f} {ww:.6f} {hh:.6f}")
            cls_counter[cls] += 1

        if not boxes:
            neg_samples += 1  # 空标注 = 负样本

        (out_lbl / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""),
                                             encoding="utf-8")
        written_labels += 1

        dst = out_img / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
        copied_images += 1

    print("\n==== T2 转换结果 ====")
    print(f"写出标签文件      : {written_labels}")
    print(f"复制图片          : {copied_images}")
    print(f"负样本(空标注)图  : {neg_samples}")
    print(f"SQL有记录但磁盘缺 : {missing}")
    print(f"坐标越界警告      : {coord_warn}")
    print(f"类别分布(框数)    : "
          + ", ".join(f"{k}={v}" for k, v in sorted(cls_counter.items()))
          + "  (0=ptl_red, 1=ptl_green)")

    if args.viz and copied_images:
        _visualize(out_img, out_lbl, out_dir / "viz_sample.jpg", args.viz)

    return 0


def _visualize(img_dir: Path, lbl_dir: Path, out_path: Path, n: int) -> None:
    """随机抽 n 张画框拼成网格，肉眼确认坐标没转错（坐标系搞反是最常见 bug）。"""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("[VIZ] 未安装 Pillow，跳过可视化。装 ultralytics 后会自带 Pillow。")
        return
    import random

    lbls = sorted(lbl_dir.glob("*.txt"))
    # 只抽有框的
    lbls = [p for p in lbls if p.read_text().strip()]
    if not lbls:
        print("[VIZ] 没有含框的标签，跳过。")
        return
    random.seed(0)
    pick = random.sample(lbls, min(n, len(lbls)))

    cell = 320
    cols = 6
    rows = (len(pick) + cols - 1) // cols
    grid = Image.new("RGB", (cols * cell, rows * cell), (30, 30, 30))
    colors = {0: (255, 60, 60), 1: (60, 255, 60)}

    for i, lbl in enumerate(pick):
        stem = lbl.stem
        img_p = next((img_dir / f"{stem}{e}" for e in (".jpg", ".JPG", ".jpeg", ".png")
                      if (img_dir / f"{stem}{e}").exists()), None)
        if img_p is None:
            continue
        im = Image.open(img_p).convert("RGB")
        W, H = im.size
        d = ImageDraw.Draw(im)
        for line in lbl.read_text().strip().splitlines():
            c, x, y, w, h = line.split()
            c = int(c); x, y, w, h = map(float, (x, y, w, h))
            x1 = (x - w / 2) * W; y1 = (y - h / 2) * H
            x2 = (x + w / 2) * W; y2 = (y + h / 2) * H
            d.rectangle([x1, y1, x2, y2], outline=colors.get(c, (255, 255, 0)), width=6)
        im.thumbnail((cell, cell))
        gx = (i % cols) * cell; gy = (i // cols) * cell
        grid.paste(im, (gx, gy))

    grid.save(out_path)
    print(f"[VIZ] 抽检网格已存: {out_path}  (红框=red, 绿框=green，请肉眼确认框贴在灯上)")


def main() -> int:
    ap = argparse.ArgumentParser(description="Ampel-Pilot SQL dump -> YOLO")
    ap.add_argument("--sql", default="data/raw/ampel_pilot/database/ampelpilot_data.sql")
    ap.add_argument("--images-dir", default="data/raw/ampel_pilot/images",
                    help="解压后的原图目录（脚本会递归查找 .jpg/.jpeg/.png）")
    ap.add_argument("--out", default="data/interim/ampel_pilot")
    ap.add_argument("--viz", type=int, default=30, help="抽检可视化张数，0=关闭")
    return convert(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
