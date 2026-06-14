#!/usr/bin/env python3
"""
Local web GUI for controlling the manga translation pipeline.
"""
import argparse
import json
import mimetypes
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

from main import load_config


REPO_ROOT = Path(__file__).resolve().parent
WEB_ROOT = REPO_ROOT / "webui"


def load_gui_defaults(repo_root: Path) -> dict:
    config_path = repo_root / "config.yaml"
    config = load_config(str(config_path))
    return {
        "mode": "single",
        "config_path": "config.yaml",
        "output_dir": config["output"]["base_dir"],
        "api_key": "",
        "image_path": "",
        "patterns": "",
        "max_pages": 0,
    }


def _split_patterns(text: str) -> list[str]:
    return [line.strip() for line in str(text).splitlines() if line.strip()]


def build_pipeline_command(payload: dict, repo_root: Path) -> tuple[list[str], str]:
    mode = payload.get("mode", "single")
    config_path = str(payload.get("config_path") or "config.yaml")
    output_dir = str(payload.get("output_dir") or load_gui_defaults(repo_root)["output_dir"])
    api_key = str(payload.get("api_key") or "")

    if mode == "batch":
        patterns = _split_patterns(payload.get("patterns", ""))
        if not patterns:
            raise ValueError("Batch mode requires at least one glob pattern.")
        command = [sys.executable, str(repo_root / "run_batch.py"), "--config", config_path, "--output", output_dir]
        if api_key:
            command.extend(["--api-key", api_key])
        max_pages = int(payload.get("max_pages") or 0)
        if max_pages > 0:
            command.extend(["--max-pages", str(max_pages)])
        report_path = str(Path(output_dir) / "batch_report.json")
        command.extend(["--report", report_path])
        command.extend(patterns)
        return command, output_dir

    image_path = str(payload.get("image_path") or "").strip()
    if not image_path:
        raise ValueError("Single mode requires an image path.")
    command = [sys.executable, str(repo_root / "main.py"), "--config", config_path, "--output", output_dir]
    if api_key:
        command.extend(["--api-key", api_key])
    command.append(image_path)
    return command, output_dir


def discover_result_items(output_dir: str | Path) -> list[dict]:
    out_path = Path(output_dir)
    if not out_path.exists():
        return []

    items = []
    for result_path in sorted(out_path.glob("*_result.json")):
        try:
            with open(result_path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        bubbles = data.get("detection", {}).get("bubbles", [])
        final_image = data.get("final_image")
        items.append(
            {
                "page_id": data.get("page_id", result_path.stem.replace("_result", "")),
                "result_path": str(result_path),
                "final_image": final_image,
                "translated_count": sum(1 for bubble in bubbles if bubble.get("translation")),
                "bubble_count": len(bubbles),
                "updated_at": result_path.stat().st_mtime,
            }
        )
    items.sort(key=lambda item: item["updated_at"], reverse=True)
    return items


class PipelineJob:
    def __init__(self, command: list[str], output_dir: str):
        self.id = uuid.uuid4().hex[:10]
        self.command = command
        self.output_dir = output_dir
        self.status = "queued"
        self.started_at = time.time()
        self.finished_at = None
        self.returncode = None
        self.logs = deque(maxlen=2000)
        self.process = None
        self.lock = threading.Lock()

    def add_log(self, line: str):
        with self.lock:
            self.logs.append(line.rstrip("\n"))

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "id": self.id,
                "status": self.status,
                "command": self.command,
                "output_dir": self.output_dir,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "returncode": self.returncode,
                "logs": list(self.logs),
                "results": discover_result_items(self.output_dir),
            }


class JobManager:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.jobs: dict[str, PipelineJob] = {}
        self.current_job_id: str | None = None
        self.lock = threading.Lock()

    def start(self, payload: dict) -> dict:
        command, output_dir = build_pipeline_command(payload, self.repo_root)
        with self.lock:
            if self.current_job_id:
                current = self.jobs[self.current_job_id]
                if current.status in {"queued", "running"}:
                    raise RuntimeError("A pipeline job is already running.")
            job = PipelineJob(command, output_dir)
            self.jobs[job.id] = job
            self.current_job_id = job.id

        thread = threading.Thread(target=self._run_job, args=(job,), daemon=True)
        thread.start()
        return job.snapshot()

    def _run_job(self, job: PipelineJob):
        job.status = "running"
        process = subprocess.Popen(
            job.command,
            cwd=self.repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        job.process = process

        assert process.stdout is not None
        for line in process.stdout:
            job.add_log(line)

        process.wait()
        job.returncode = process.returncode
        job.finished_at = time.time()
        job.status = "done" if process.returncode == 0 else "error"

    def stop(self) -> dict:
        with self.lock:
            if not self.current_job_id:
                raise RuntimeError("No active job.")
            job = self.jobs[self.current_job_id]
        if job.process and job.status == "running":
            job.process.terminate()
            job.status = "stopping"
            return job.snapshot()
        raise RuntimeError("Active job is not running.")

    def status(self) -> dict:
        with self.lock:
            current = self.jobs.get(self.current_job_id) if self.current_job_id else None
            recent_jobs = [job.snapshot() for job in list(self.jobs.values())[-5:]][::-1]
        return {
            "defaults": load_gui_defaults(self.repo_root),
            "current_job": current.snapshot() if current else None,
            "recent_jobs": recent_jobs,
        }

    def allowed_roots(self) -> list[Path]:
        roots = {self.repo_root.resolve()}
        with self.lock:
            for job in self.jobs.values():
                roots.add(Path(job.output_dir).resolve())
        return sorted(roots)


JOB_MANAGER = JobManager(REPO_ROOT)


def _is_allowed_path(path: Path, roots: list[Path]) -> bool:
    resolved = path.resolve()
    for root in roots:
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False


class GuiHandler(BaseHTTPRequestHandler):
    server_version = "MangaPipelineGUI/0.1"

    def _json_response(self, payload: dict, status: int = 200):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_file(self, path: Path):
        mime_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            return self._serve_file(WEB_ROOT / "index.html")
        if parsed.path == "/api/status":
            return self._json_response(JOB_MANAGER.status())
        if parsed.path == "/api/file":
            query = parse_qs(parsed.query)
            raw_path = query.get("path", [""])[0]
            if not raw_path:
                return self._json_response({"error": "Missing path."}, status=400)
            path = Path(unquote(raw_path))
            if not path.exists() or not _is_allowed_path(path, JOB_MANAGER.allowed_roots()):
                return self._json_response({"error": "File not allowed."}, status=404)
            return self._serve_file(path)
        return self._json_response({"error": "Not found."}, status=404)

    def do_POST(self):
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b"{}"
        payload = json.loads(body.decode("utf-8") or "{}")

        try:
            if parsed.path == "/api/run":
                return self._json_response(JOB_MANAGER.start(payload), status=201)
            if parsed.path == "/api/stop":
                return self._json_response(JOB_MANAGER.stop())
        except (ValueError, RuntimeError) as exc:
            return self._json_response({"error": str(exc)}, status=400)

        return self._json_response({"error": "Not found."}, status=404)

    def log_message(self, fmt, *args):
        return


def main():
    parser = argparse.ArgumentParser(description="Local web GUI for the manga pipeline")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), GuiHandler)
    print(f"GUI server running at http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
