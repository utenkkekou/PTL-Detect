#!/usr/bin/env python3
"""Create a review manifest from an image folder with an OpenAI-compatible API.

The API key is supplied at runtime. This module never contains or logs secrets.
It can also create an empty manifest for fully manual annotation.
"""
from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageOps

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

CLASSES = [
    {"id": 0, "name": "pedestrian_red", "label": "人行红灯"},
    {"id": 1, "name": "pedestrian_green", "label": "人行绿灯"},
    {"id": 2, "name": "pedestrian_unknown", "label": "人行灯未知"},
    {"id": 3, "name": "vehicle_signal", "label": "车行/自行车灯"},
]

CLASS_ALIASES = {
    "pedestrian_red": 0,
    "red": 0,
    "人行红灯": 0,
    "pedestrian_green": 1,
    "green": 1,
    "人行绿灯": 1,
    "pedestrian_unknown": 2,
    "unknown": 2,
    "人行灯未知": 2,
    "vehicle_signal": 3,
    "vehicle": 3,
    "bicycle_signal": 3,
    "车行灯": 3,
    "自行车灯": 3,
}

DEFAULT_PROMPT = """你是交通信号灯目标检测标注助手。找出图中所有独立灯头，并按以下类别返回：
- pedestrian_red：人行红灯
- pedestrian_green：人行绿灯
- pedestrian_unknown：人行专用灯，但灯色无法确认
- vehicle_signal：车行圆灯、箭头灯或自行车信号灯

不要标注倒计时数字、摄像头、控制盒或普通路灯。
只返回 JSON，格式：
{"boxes":[{"label":"pedestrian_red","bbox":[x1,y1,x2,y2],"confidence":0.95}]}
bbox 使用 0 到 1000 的整数坐标，原点在图片左上角。没有目标时返回 {"boxes":[]}。"""


@dataclass(frozen=True)
class ApiSettings:
    base_url: str
    api_key: str
    model: str
    prompt: str = DEFAULT_PROMPT
    timeout: int = 90

    @property
    def enabled(self) -> bool:
        return bool(self.base_url.strip() and self.api_key.strip() and self.model.strip())


def discover_images(folder: Path, recursive: bool = True) -> list[Path]:
    """Return supported images in stable relative-path order."""
    folder = folder.expanduser().resolve()
    iterator = folder.rglob("*") if recursive else folder.glob("*")
    images = [
        path for path in iterator
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        and ".quick-annotation" not in path.parts
    ]
    return sorted(images, key=lambda path: path.relative_to(folder).as_posix().lower())


def _image_id(relative_path: Path) -> str:
    digest = hashlib.sha1(relative_path.as_posix().encode("utf-8")).hexdigest()[:12]
    stem = re.sub(r"[^0-9A-Za-z_-]+", "_", relative_path.stem).strip("_") or "image"
    return f"{stem[:48]}-{digest}"


def _make_preview(source: Path, destination: Path, max_size: int = 1600) -> tuple[int, int]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
        width, height = image.size
        image.thumbnail((max_size, max_size))
        image.save(destination, "JPEG", quality=88, optimize=True)
    return width, height


def _data_uri(path: Path, max_size: int = 1280) -> str:
    from io import BytesIO

    with Image.open(path) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
        image.thumbnail((max_size, max_size))
        buffer = BytesIO()
        image.save(buffer, "JPEG", quality=85, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _chat_completions_url(base_url: str) -> str:
    base = base_url.strip().rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if not base.endswith("/v1"):
        base += "/v1"
    return base + "/chat/completions"


def _extract_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return {"boxes": value}
    text = str(value or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.I | re.S)
    if fenced:
        text = fenced.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]
    parsed = json.loads(text)
    if isinstance(parsed, list):
        return {"boxes": parsed}
    if not isinstance(parsed, dict):
        raise ValueError("API response is not a JSON object")
    return parsed


def _normalize_box(raw: dict[str, Any]) -> dict[str, Any] | None:
    label = raw.get("label", raw.get("class", raw.get("name", "")))
    if isinstance(label, int) or (isinstance(label, str) and label.isdigit()):
        cls = int(label)
    else:
        cls = CLASS_ALIASES.get(str(label).strip().lower(), -1)
    if cls not in {0, 1, 2, 3}:
        return None

    bbox = raw.get("bbox", raw.get("box"))
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return None

    scale = 1000.0 if max(abs(x1), abs(y1), abs(x2), abs(y2)) > 1.5 else 1.0
    x1, y1, x2, y2 = (value / scale for value in (x1, y1, x2, y2))
    x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
    y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
    if x2 - x1 < 0.001 or y2 - y1 < 0.001:
        return None

    confidence = raw.get("confidence", raw.get("score", 0.5))
    try:
        confidence = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        confidence = 0.5
    return {
        "cls": cls,
        "cx": round((x1 + x2) / 2, 6),
        "cy": round((y1 + y2) / 2, 6),
        "w": round(x2 - x1, 6),
        "h": round(y2 - y1, 6),
        "conf": round(confidence, 4),
        "source": "api",
    }


def parse_api_boxes(payload: Any) -> list[dict[str, Any]]:
    """Parse an API response or message content into review boxes."""
    data = _extract_json(payload)
    boxes = data.get("boxes", data.get("objects", []))
    if not isinstance(boxes, list):
        raise ValueError("API JSON field 'boxes' must be a list")
    return [box for raw in boxes if isinstance(raw, dict)
            if (box := _normalize_box(raw)) is not None]


def call_vision_api(image_path: Path, settings: ApiSettings) -> list[dict[str, Any]]:
    """Call an OpenAI-compatible Chat Completions vision endpoint."""
    payload = {
        "model": settings.model,
        "temperature": 0,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": settings.prompt or DEFAULT_PROMPT},
                {"type": "image_url", "image_url": {"url": _data_uri(image_path)}},
            ],
        }],
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        _chat_completions_url(settings.base_url),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {settings.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=settings.timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"API HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"API connection failed: {exc.reason}") from exc

    try:
        content = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("API response does not contain choices[0].message.content") from exc
    return parse_api_boxes(content)


def build_manifest(
    input_dir: Path,
    output_dir: Path,
    settings: ApiSettings | None = None,
    *,
    recursive: bool = True,
    workers: int = 2,
    progress: Callable[[int, int, str], None] | None = None,
) -> Path:
    """Build previews and a review manifest. API calls are optional."""
    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise ValueError(f"Image folder does not exist: {input_dir}")
    if input_dir == output_dir:
        raise ValueError("Output folder must be different from the image folder")

    images = discover_images(input_dir, recursive=recursive)
    if not images:
        raise ValueError("No supported images were found")

    preview_dir = output_dir / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    total = len(images)

    def process(image_path: Path) -> dict[str, Any]:
        relative = image_path.relative_to(input_dir)
        item_id = _image_id(relative)
        preview_name = f"{item_id}.jpg"
        width, height = _make_preview(image_path, preview_dir / preview_name)
        boxes: list[dict[str, Any]] = []
        error = ""
        if settings and settings.enabled:
            try:
                boxes = call_vision_api(image_path, settings)
            except Exception as exc:  # Keep the image reviewable if one API call fails.
                error = str(exc)
        return {
            "id": item_id,
            "file": relative.as_posix(),
            "preview": f"preview/{preview_name}",
            "width": width,
            "height": height,
            "mode": -1,
            "target_class": -1,
            "boxes": boxes,
            "n_boxes": len(boxes),
            "sort_key": round(min((b["conf"] for b in boxes), default=-1.0), 4),
            "multi": len(boxes) > 1,
            "split": "user",
            "review_note": f"API预标注失败：{error}" if error else "",
            "source": "api" if settings and settings.enabled else "manual",
        }

    manifest: list[dict[str, Any]] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, min(int(workers), 8))) as executor:
        futures = {executor.submit(process, path): path for path in images}
        for future in as_completed(futures):
            manifest.append(future.result())
            completed += 1
            if progress:
                progress(completed, total, futures[future].name)

    manifest.sort(key=lambda item: (item["sort_key"], item["file"].lower()))
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "review_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    metadata = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "total": total,
        "classes": CLASSES,
        "api_enabled": bool(settings and settings.enabled),
    }
    (output_dir / "project.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--no-recursive", action="store_true")
    args = parser.parse_args()

    source = Path(args.input_dir)
    destination = Path(args.output_dir) if args.output_dir else source / ".quick-annotation"
    api = ApiSettings(args.base_url, args.api_key, args.model)
    result = build_manifest(
        source,
        destination,
        api if api.enabled else None,
        recursive=not args.no_recursive,
        workers=args.workers,
        progress=lambda done, total, name: print(f"[{done}/{total}] {name}"),
    )
    print(result)
