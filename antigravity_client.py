"""Safe subprocess adapter for the official Antigravity CLI."""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Mapping, Optional

from PIL import Image

logger = logging.getLogger(__name__)


class AntigravityCliClient:
    """Run one-shot model prompts through the user's Antigravity subscription."""

    _lock = threading.Lock()

    def __init__(self, model: str, timeout: int = 180, binary: Optional[str] = None):
        self.model = model
        self.timeout = max(10, int(timeout))
        self.binary = binary or shutil.which("agy") or ""
        self.last_error = ""

    @property
    def available(self) -> bool:
        return bool(self.binary and Path(self.binary).is_file())

    @staticmethod
    def _safe_environment() -> dict[str, str]:
        allowed = {
            "HOME",
            "LANG",
            "LC_ALL",
            "LC_CTYPE",
            "LOGNAME",
            "NO_PROXY",
            "PATH",
            "SHELL",
            "SSL_CERT_DIR",
            "SSL_CERT_FILE",
            "TERM",
            "TMPDIR",
            "USER",
        }
        return {key: value for key, value in os.environ.items() if key in allowed}

    @staticmethod
    def _safe_filename(label: str, index: int) -> str:
        normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(label)).strip("._")
        return f"{index:03d}_{normalized or 'image'}.png"

    def generate(
        self,
        prompt: str,
        images: Optional[Mapping[str, Image.Image]] = None,
    ) -> Optional[str]:
        """Return the model's plain-text response, or None on any CLI failure."""
        self.last_error = ""
        if not self.available:
            self.last_error = "Antigravity CLI executable 'agy' was not found"
            logger.error(self.last_error)
            return None

        with tempfile.TemporaryDirectory(prefix="manga-trans-agy-") as temp_name:
            temp_dir = Path(temp_name)
            image_lines = []
            for index, (label, image) in enumerate((images or {}).items(), start=1):
                image_path = temp_dir / self._safe_filename(label, index)
                image.convert("RGB").save(image_path, format="PNG")
                image_lines.append(f"- {label}: {image_path}")

            full_prompt = prompt.strip()
            if image_lines:
                full_prompt += (
                    "\n\nIMAGE FILES (authoritative; read every file with the image-capable file tool):\n"
                    + "\n".join(image_lines)
                    + "\nIf an image cannot be opened, return exactly [IMAGE_READ_FAILED]. "
                    "Never guess or substitute content from another image."
                )

            command = [
                self.binary,
                "-p",
                full_prompt,
                "--agent",
                self.model,
                "--sandbox",
                "--print-timeout",
                f"{self.timeout}s",
            ]
            if image_lines:
                command.extend(["--add-dir", str(temp_dir)])

            try:
                with self._lock:
                    completed = subprocess.run(
                        command,
                        cwd=temp_dir,
                        env=self._safe_environment(),
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=self.timeout + 30,
                        check=False,
                    )
            except subprocess.TimeoutExpired:
                self.last_error = f"Antigravity CLI timed out after {self.timeout + 30}s"
                logger.error(self.last_error)
                return None
            except OSError as exc:
                self.last_error = f"Antigravity CLI failed to start: {exc}"
                logger.error(self.last_error)
                return None

            stdout = completed.stdout.strip()
            stderr = completed.stderr.strip()
            if completed.returncode != 0:
                self.last_error = stderr or stdout or f"Antigravity CLI exited with {completed.returncode}"
                logger.error(self.last_error[:500])
                return None
            if not stdout or "no output produced" in stdout.lower():
                self.last_error = stderr or "Antigravity CLI returned no output"
                logger.error(self.last_error[:500])
                return None
            if "[IMAGE_READ_FAILED]" in stdout:
                self.last_error = "Antigravity CLI could not read a staged image"
                logger.error(self.last_error)
                return None
            return stdout
