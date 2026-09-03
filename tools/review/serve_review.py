#!/usr/bin/env python3
"""Quick Annotation 本地预标注与人工审核服务。

- 设置页选择图片文件夹并配置 OpenAI-compatible 视觉 API。
- 没有 API 时也可创建空白任务，直接人工画框。
- 审核进度写入任务目录，支持中断续做与多人分段。
- API Key 仅保存到用户目录，不返回浏览器、不写入项目。

快速启动：
  python tools/review/serve_review.py
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, unquote, parse_qs

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.api_pre_annotate import (  # noqa: E402
    DEFAULT_PROMPT,
    ApiSettings,
    build_manifest,
    discover_images,
)


class State:
    def __init__(self, manifest: Path, root: Path, progress: Path):
        self.root = root
        self.progress_path = progress
        self.manifest = json.loads(manifest.read_text(encoding="utf-8"))
        self.manifest.sort(key=lambda m: m.get("sort_key", 1.0))
        if progress.exists():
            self.progress = json.loads(progress.read_text(encoding="utf-8"))
        else:
            self.progress = {}
        self.lock = threading.Lock()  # 多人并发保存安全
        self.claims = {}   # 协作模式: id -> (worker, expiry_epoch)
        self.LEASE = 300   # 领取的图 300s 内没标完则释放回池

    def _reviewed(self, iid):
        p = self.progress.get(iid)
        return bool(p and p.get("status") == "reviewed")

    def next_batch(self, worker, n):
        """协作模式: 派 n 张"未复核且没被别人锁定"的图给 worker(并锁定)。返回(items, 全局剩余)。"""
        with self.lock:
            now = time.time()
            out = []
            for m in self.manifest:
                if len(out) >= n:
                    break
                iid = m["id"]
                if self._reviewed(iid):
                    continue
                cl = self.claims.get(iid)
                if cl and cl[0] != worker and cl[1] > now:
                    continue  # 别人锁着且未过期
                self.claims[iid] = (worker, now + self.LEASE)
                item = dict(m)
                p = self.progress.get(iid)
                item["status"] = p.get("status", "pending") if p else "pending"
                if p and "boxes" in p:
                    item["boxes"] = p["boxes"]
                out.append(item)
            remaining = sum(1 for m in self.manifest if not self._reviewed(m["id"]))
            return out, remaining

    def save(self):
        # 原子写: 先写临时文件再替换, 避免并发读到半截 JSON
        tmp = self.progress_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.progress, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        tmp.replace(self.progress_path)

    def stats(self):
        prog = dict(self.progress)  # 快照, 避免并发迭代报错
        # 只统计当前 manifest 的条目；同一进度文件里若残留别的任务 ID，不能污染进度条。
        reviewed = sum(1 for m in self.manifest if self._reviewed(m["id"]))
        cls = {"0": 0, "1": 0, "2": 0, "3": 0}
        neg = 0
        for m in self.manifest:
            p = prog.get(m["id"])
            boxes = p["boxes"] if (p and "boxes" in p) else m["boxes"]
            if not boxes:
                neg += 1
            for b in boxes:
                cls[str(b["cls"])] = cls.get(str(b["cls"]), 0) + 1
        return {"total": len(self.manifest), "reviewed": reviewed,
                "cls_red": cls.get("0", 0), "cls_green": cls.get("1", 0),
                "cls_unknown": cls.get("2", 0), "cls_vehicle": cls.get("3", 0),
                "negatives": neg}


class ConfigStore:
    """Local settings store. The API key is never returned to the browser."""

    DEFAULTS = {
        "input_dir": "",
        "output_dir": "",
        "base_url": "https://api.openai.com/v1",
        "api_key": "",
        "model": "",
        "prompt": DEFAULT_PROMPT,
        "recursive": True,
        "workers": 2,
    }

    def __init__(self, path: Path | None = None):
        config_home = Path(
            os.environ.get("QUICK_ANNOTATION_HOME", Path.home() / ".quick-annotation")
        ).expanduser()
        self.path = path or config_home / "config.json"
        self.lock = threading.Lock()
        self.values = dict(self.DEFAULTS)
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self.values.update(loaded)
            except (OSError, json.JSONDecodeError):
                pass

    def public(self) -> dict:
        with self.lock:
            values = dict(self.values)
        values.pop("api_key", None)
        values["has_api_key"] = bool(self.values.get("api_key"))
        return values

    def update(self, payload: dict) -> dict:
        allowed_text = ("input_dir", "output_dir", "base_url", "model", "prompt")
        with self.lock:
            for key in allowed_text:
                if key in payload:
                    self.values[key] = str(payload.get(key, "")).strip()
            if payload.get("api_key"):
                self.values["api_key"] = str(payload["api_key"]).strip()
            if payload.get("clear_api_key"):
                self.values["api_key"] = ""
            if "recursive" in payload:
                self.values["recursive"] = bool(payload["recursive"])
            if "workers" in payload:
                try:
                    self.values["workers"] = max(1, min(int(payload["workers"]), 8))
                except (TypeError, ValueError):
                    self.values["workers"] = 2
            self._save_unlocked()
        return self.public()

    def _save_unlocked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(".tmp")
        temp_path.write_text(
            json.dumps(self.values, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp_path.replace(self.path)
        try:
            self.path.chmod(0o600)
        except OSError:
            pass


def _validate_review_state(
    manifest: Path,
    root: Path,
    progress: Path,
    *,
    allow_stale: bool,
) -> State:
    state = State(manifest, root, progress)
    manifest_has_vehicle = any(
        int(box.get("cls", -1)) == 3
        for item in state.manifest for box in item.get("boxes", [])
    )
    progress_has_vehicle = any(
        int(box.get("cls", -1)) == 3
        for saved in state.progress.values() for box in saved.get("boxes", [])
    )
    all_reviewed = bool(state.manifest) and all(
        state._reviewed(item["id"]) for item in state.manifest
    )
    stale = (
        progress.exists()
        and progress.stat().st_mtime < manifest.stat().st_mtime
        and manifest_has_vehicle
        and not progress_has_vehicle
        and all_reviewed
    )
    if stale and not allow_stale:
        raise ValueError(
            "检测到新四类清单与较旧三类进度文件的危险组合；"
            "请使用新的进度文件，或显式传入 --allow-stale-progress。"
        )
    return state


class AppState:
    def __init__(self, config: ConfigStore, *, allow_stale: bool = False):
        self.config = config
        self.allow_stale = allow_stale
        self.review: State | None = None
        self.lock = threading.RLock()
        self.job = {
            "status": "idle",
            "done": 0,
            "total": 0,
            "current": "",
            "message": "",
            "error": "",
        }
        self.load_configured_review()

    def load_configured_review(self) -> bool:
        values = self.config.values
        output_text = str(values.get("output_dir", "")).strip()
        if not output_text:
            with self.lock:
                self.review = None
            return False
        output = Path(output_text).expanduser()
        manifest = output / "review_manifest.json"
        if not manifest.exists():
            with self.lock:
                self.review = None
            return False
        progress = output / "review_progress.json"
        with self.lock:
            self.review = _validate_review_state(
                manifest, output, progress, allow_stale=self.allow_stale
            )
        return True

    def load_explicit_review(self, manifest: Path, root: Path, progress: Path) -> None:
        with self.lock:
            self.review = _validate_review_state(
                manifest, root, progress, allow_stale=self.allow_stale
            )

    def config_response(self) -> dict:
        with self.lock:
            review = self.review
            total = len(review.manifest) if review else 0
        return {
            "config": self.config.public(),
            "review_ready": review is not None,
            "total": total,
        }

    def start_preannotation(self, use_api: bool) -> int:
        values = dict(self.config.values)
        input_dir = Path(values.get("input_dir", "")).expanduser()
        if not input_dir.is_dir():
            raise ValueError("请选择有效的图片文件夹")
        output_text = values.get("output_dir", "")
        output_dir = (
            Path(output_text).expanduser()
            if output_text else input_dir / ".quick-annotation"
        )
        if not output_text:
            self.config.update({"output_dir": str(output_dir)})
            values["output_dir"] = str(output_dir)

        api = ApiSettings(
            str(values.get("base_url", "")),
            str(values.get("api_key", "")),
            str(values.get("model", "")),
            str(values.get("prompt", DEFAULT_PROMPT)) or DEFAULT_PROMPT,
        )
        if use_api and not api.enabled:
            raise ValueError("使用 API 预标注前，请填写并保存 API 地址、Key 和模型名称")
        images = discover_images(input_dir, recursive=bool(values.get("recursive", True)))
        if not images:
            raise ValueError("所选文件夹中没有支持的图片")

        with self.lock:
            if self.job["status"] == "running":
                raise ValueError("已有预标注任务正在运行")
            self.job = {
                "status": "running",
                "done": 0,
                "total": len(images),
                "current": "准备中…",
                "message": "",
                "error": "",
            }

        def update(done: int, total: int, current: str) -> None:
            with self.lock:
                self.job.update(done=done, total=total, current=current)

        def worker() -> None:
            try:
                manifest = build_manifest(
                    input_dir,
                    output_dir,
                    api if use_api else None,
                    recursive=bool(values.get("recursive", True)),
                    workers=int(values.get("workers", 2)),
                    progress=update,
                )
                progress_path = output_dir / "review_progress.json"
                state = _validate_review_state(
                    manifest,
                    output_dir,
                    progress_path,
                    allow_stale=self.allow_stale,
                )
                with self.lock:
                    self.review = state
                    self.job.update(
                        status="done",
                        done=len(state.manifest),
                        total=len(state.manifest),
                        current="处理完成",
                        message="任务已生成",
                    )
            except Exception as exc:
                with self.lock:
                    self.job.update(
                        status="error",
                        current="",
                        error=str(exc),
                    )

        threading.Thread(target=worker, daemon=True, name="pre-annotation").start()
        return len(images)


def _pick_directory(initial: str = "") -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as exc:
        raise RuntimeError("当前 Python 未安装 tkinter，请直接粘贴文件夹路径") from exc
    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
        selected = filedialog.askdirectory(
            initialdir=initial if initial and Path(initial).is_dir() else None
        )
        return str(Path(selected).resolve()) if selected else ""
    finally:
        root.destroy()


def make_handler(app: AppState, review_html: str, setup_html: str):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # 安静
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _html(self, html: str):
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(body)

        def _state(self) -> State | None:
            with app.lock:
                state = app.review
            if state is None:
                self._json({"error": "尚未创建审核任务，请先完成设置"}, 409)
            return state

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                self._html(setup_html)
            elif path in ("/review", "/index.html"):
                if app.review is None:
                    self.send_response(302)
                    self.send_header("Location", "/")
                    self.end_headers()
                else:
                    self._html(review_html)
            elif path == "/api/config":
                self._json(app.config_response())
            elif path == "/api/job":
                with app.lock:
                    self._json(dict(app.job))
            elif path == "/api/manifest":
                state = self._state()
                if state is None:
                    return
                # 分段: ?seg=k&segs=N -> 轮转取第 k 段(每人一段, 互不重叠)
                q = parse_qs(urlparse(self.path).query)
                try:
                    seg = int(q.get("seg", ["0"])[0]); segs = int(q.get("segs", ["1"])[0])
                except ValueError:
                    seg, segs = 0, 1
                if segs < 1: segs = 1
                seg = max(0, min(seg, segs - 1))
                out = []
                for idx, m in enumerate(state.manifest):
                    if segs > 1 and idx % segs != seg:
                        continue
                    item = dict(m)
                    p = state.progress.get(m["id"])
                    item["status"] = p.get("status", "pending") if p else "pending"
                    if p and "boxes" in p:
                        item["boxes"] = p["boxes"]
                    out.append(item)
                self._json({"items": out, "stats": state.stats(),
                            "seg": seg, "segs": segs})
            elif path == "/api/next":
                state = self._state()
                if state is None:
                    return
                # 协作模式: 派下一批未复核且未被锁定的图给该 worker
                q = parse_qs(urlparse(self.path).query)
                worker = q.get("worker", ["anon"])[0]
                try:
                    n = int(q.get("n", ["25"])[0])
                except ValueError:
                    n = 25
                n = max(1, min(n, 100))
                items, remaining = state.next_batch(worker, n)
                self._json({"items": items, "remaining": remaining, "stats": state.stats()})
            elif path == "/api/stats":
                state = self._state()
                if state is not None:
                    self._json(state.stats())
            elif path.startswith("/img/"):
                state = self._state()
                if state is None:
                    return
                rel = unquote(path[len("/img/"):])
                fp = (state.root / rel).resolve()
                try:
                    fp.relative_to(state.root.resolve())
                except ValueError:
                    self.send_error(404)
                    return
                if not fp.is_file():
                    self.send_error(404)
                    return
                data = fp.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "max-age=3600")
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_error(404)

        def do_POST(self):
            path = urlparse(self.path).path
            try:
                n = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(n) or b"{}")
                if not isinstance(payload, dict):
                    raise ValueError("请求内容必须是 JSON 对象")
            except (ValueError, json.JSONDecodeError) as exc:
                self._json({"error": f"请求内容无效：{exc}"}, 400)
                return

            if path == "/api/config":
                try:
                    app.config.update(payload)
                    app.load_configured_review()
                    self._json(app.config_response())
                except Exception as exc:
                    self._json({"error": str(exc)}, 400)
            elif path == "/api/select-folder":
                target = str(payload.get("target", "input"))
                current = app.config.values.get(f"{target}_dir", "")
                try:
                    selected = _pick_directory(str(current))
                    suggested = (
                        str(Path(selected) / ".quick-annotation")
                        if selected and target == "input" else ""
                    )
                    self._json({"path": selected, "suggested_output": suggested})
                except Exception as exc:
                    self._json({"error": str(exc)}, 400)
            elif path == "/api/preannotate":
                try:
                    total = app.start_preannotation(bool(payload.get("use_api")))
                    self._json({"ok": True, "total": total})
                except Exception as exc:
                    self._json({"error": str(exc)}, 400)
            elif path == "/api/save":
                state = self._state()
                if state is None:
                    return
                iid = payload["id"]
                with state.lock:  # 并发保存串行化, 防止互相覆盖/文件损坏
                    state.progress[iid] = {
                        "status": payload.get("status", "reviewed"),
                        "boxes": payload.get("boxes", []),
                        "changed": payload.get("changed", False),
                        "by": payload.get("by", ""),  # 哪个分段/复核者(可选)
                    }
                    state.claims.pop(iid, None)  # 已复核 -> 释放锁定
                    state.save()
                    stats = state.stats()
                self._json({"ok": True, "stats": stats})
            else:
                self.send_error(404)
    return H


def main() -> int:
    ap = argparse.ArgumentParser(description="Quick Annotation 本地预标注与审核工具")
    ap.add_argument("--manifest", default="",
                    help="可选：直接打开已有 review_manifest.json")
    ap.add_argument("--root", default="",
                    help="可选：预览图根目录；默认使用 manifest 所在目录")
    ap.add_argument("--progress", default="",
                    help="可选：审核进度文件；默认放在预览图根目录")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1",
                    help="监听地址; 局域网共享用 0.0.0.0")
    ap.add_argument("--segs", type=int, default=1,
                    help="分几段给多人复核(仅用于打印各段网址; 实际分段由 URL ?seg 决定)")
    ap.add_argument("--allow-stale-progress", action="store_true",
                    help="允许较旧进度覆盖较新的候选清单；通常不应使用")
    ap.add_argument("--weak-hints", action="store_true",
                    help="使用右侧弱提示：预标框照常显示，但不弹出改类/删框强确认")
    ap.add_argument("--config", default="",
                    help="本地配置文件路径；默认 ~/.quick-annotation/config.json")
    ap.add_argument("--no-browser", action="store_true",
                    help="启动后不自动打开浏览器")
    args = ap.parse_args()

    config_path = Path(args.config).expanduser() if args.config else None
    app = AppState(
        ConfigStore(config_path),
        allow_stale=args.allow_stale_progress,
    )
    if args.manifest:
        manifest = Path(args.manifest).expanduser().resolve()
        if not manifest.exists():
            print(f"[ERR] manifest 不存在: {manifest}", file=sys.stderr)
            return 2
        root = Path(args.root).expanduser().resolve() if args.root else manifest.parent
        progress = (
            Path(args.progress).expanduser().resolve()
            if args.progress else root / "review_progress.json"
        )
        try:
            app.load_explicit_review(manifest, root, progress)
        except ValueError as exc:
            print(f"[STOP] {exc}", file=sys.stderr)
            return 2

    review_html = (HERE / "index.html").read_text(encoding="utf-8")
    setup_html = (HERE / "setup.html").read_text(encoding="utf-8")
    if args.weak_hints:
        review_html = review_html.replace(
            "<body>",
            '<body data-default-hints="weak">',
            1,
        )
    srv = ThreadingHTTPServer(
        (args.host, args.port),
        make_handler(app, review_html, setup_html),
    )
    print(f"Quick Annotation 已启动 (监听 {args.host}:{args.port})")
    if app.review is not None:
        stats = app.review.stats()
        print(f"  当前任务: {stats['total']} 项, 已复核 {stats['reviewed']}")
        print(f"  进度文件: {app.review.progress_path}")
    else:
        print("  尚未加载任务，请在设置页选择图片文件夹。")
    if args.weak_hints:
        print("  提示模式: 右侧弱提示（预标框照常显示，无改类/删框强确认）")

    if args.host in ("0.0.0.0", "::"):
        lan_ip = _lan_ip()
        base = f"http://{lan_ip}:{args.port}"
        hint_param = "&hints=weak" if args.weak_hints else ""
        local_url = f"{base}/"
        print(f"\n设置页: {local_url}")
        if app.review is not None and args.segs > 1:
            print(f"分成 {args.segs} 段, 每人一段(互不重叠, 自动合并):")
            for k in range(args.segs):
                print(
                    f"  第{k+1}人: "
                    f"{base}/review?seg={k}&segs={args.segs}{hint_param}"
                )
        print("(同事需和你在同一局域网/WiFi; 若打不开检查 Windows 防火墙放行该端口)")
    else:
        local_url = f"http://localhost:{args.port}/"
        print(f"  本机打开: {local_url}")

    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(local_url)).start()

    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        if app.review is not None:
            app.review.save()
        print("\n已保存进度, 退出。")
    return 0


def _lan_ip() -> str:
    """取本机局域网 IP(不实际发包)。"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


if __name__ == "__main__":
    raise SystemExit(main())
