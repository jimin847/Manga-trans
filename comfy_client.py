"""
ComfyUI API client — ONLY for LaMa inpainting via proven workflow
"""
import io
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import requests
from PIL import Image

logger = logging.getLogger(__name__)

# Proven LaMa workflow (from existing manga-localization)
WORKFLOW_LAMA = {
    "1": {
        "inputs": {
            "image": "__PLACEHOLDER_IMAGE__",
            "upload": "image",
        },
        "class_type": "LoadImage",
        "_meta": {"title": "Load Manga Image"},
    },
    "2": {
        "inputs": {
            "image": "__PLACEHOLDER_MASK__",
            "channel": "red",
            "upload": "image",
        },
        "class_type": "LoadImageMask",
        "_meta": {"title": "Load Mask"},
    },
    "3": {
        "inputs": {
            "images": ["1", 0],
            "masks": ["2", 0],
            "mask_threshold": 250,
            "gaussblur_radius": 12,
            "invert_mask": False,
        },
        "class_type": "LamaRemover",
        "_meta": {"title": "Lama Remover"},
    },
    "4": {
        "inputs": {
            "filename_prefix": "manga_clean_v2",
            "images": ["3", 0],
        },
        "class_type": "SaveImage",
        "_meta": {"title": "Save Clean Image"},
    },
}


class ComfyClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8188"):
        self.base_url = base_url.rstrip("/")

    def _upload_image(self, image_data: bytes, filename: str) -> str:
        """Upload image to ComfyUI."""
        resp = requests.post(
            f"{self.base_url}/upload/image",
            files={"image": (filename, io.BytesIO(image_data), "image/png")},
            data={"overwrite": "true"},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["name"]

    def _queue_prompt(self, workflow: dict) -> str:
        """Queue workflow and get prompt_id."""
        client_id = str(uuid.uuid4())
        resp = requests.post(
            f"{self.base_url}/prompt",
            json={"prompt": workflow, "client_id": client_id},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["prompt_id"]

    def _wait_for_result(self, prompt_id: str, timeout: int = 120) -> dict:
        """Poll until workflow completes. Checks queue to abort early on failure."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            # Check history
            resp = requests.get(f"{self.base_url}/history/{prompt_id}", timeout=10)
            if resp.status_code == 200 and resp.json().get(prompt_id):
                return resp.json()[prompt_id]

            # Check queue to see if it errored out (not in queue, not in history)
            try:
                queue_resp = requests.get(f"{self.base_url}/queue", timeout=10)
                if queue_resp.status_code == 200:
                    q_data = queue_resp.json()
                    in_queue = False
                    for pending in q_data.get("queue_pending", []):
                        if pending[1] == prompt_id:
                            in_queue = True
                            break
                    if not in_queue:
                        for running in q_data.get("queue_running", []):
                            if running[1] == prompt_id:
                                in_queue = True
                                break
                    
                    if not in_queue:
                        # Give it a tiny grace period to move from queue to history
                        time.sleep(1)
                        hist_resp = requests.get(f"{self.base_url}/history/{prompt_id}", timeout=10)
                        if hist_resp.status_code == 200 and not hist_resp.json().get(prompt_id):
                            raise RuntimeError(f"ComfyUI prompt {prompt_id} failed and was removed from queue.")
            except requests.RequestException:
                pass

            time.sleep(2)
        raise TimeoutError(f"ComfyUI timeout ({timeout}s)")

    def _download_image(self, history: dict) -> Optional["Image.Image"]:
        """Download result image from workflow history."""
        for node_id, node_out in history.get("outputs", {}).items():
            for img_info in node_out.get("images", []):
                resp = requests.get(
                    f"{self.base_url}/view",
                    params={
                        "filename": img_info["filename"],
                        "subfolder": img_info.get("subfolder", ""),
                        "type": img_info["type"],
                    },
                    timeout=60,
                )
                resp.raise_for_status()
                return Image.open(io.BytesIO(resp.content)).convert("RGB")
        return None

    def run_inpaint_lama(self, image: Image.Image, mask: Image.Image) -> Optional[Image.Image]:
        """Run LaMa inpainting via ComfyUI."""
        try:
            ts = str(int(time.time()))
            img_buf = io.BytesIO()
            mask_buf = io.BytesIO()
            image.save(img_buf, format="PNG")
            mask.convert("L").save(mask_buf, format="PNG")
            img_buf.seek(0)
            mask_buf.seek(0)

            img_name = self._upload_image(img_buf.read(), f"inp_img_{ts}.png")
            mask_name = self._upload_image(mask_buf.read(), f"inp_mask_{ts}.png")

            workflow = json.loads(json.dumps(WORKFLOW_LAMA))
            workflow["1"]["inputs"]["image"] = img_name
            workflow["2"]["inputs"]["image"] = mask_name

            prompt_id = self._queue_prompt(workflow)
            logger.info(f"LaMa prompt queued: {prompt_id}")
            history = self._wait_for_result(prompt_id)
            result = self._download_image(history)
            return result
        except Exception as e:
            logger.error(f"LaMa inpainting failed: {e}")
            return None


def flat_fill(image: Image.Image, mask: Image.Image,
              bubbles: Optional[list] = None) -> tuple[Image.Image, Optional[Image.Image]]:
    """
    Smart flat fill.
    When bubbles are provided: only fills white-background bubble regions;
    returns remaining mask for tone/screentone bubbles.
    When bubbles=None: old behavior, fills everything (backward compat).
    """
    import numpy as np
    from scipy import ndimage

    img = np.array(image.convert("RGB"))
    m = np.array(mask.convert("L"))
    result = img.copy()
    remaining = m.copy()  # mask not handled by flat fill → LaMa

    if bubbles:
        for bubble in bubbles:
            bx1, by1, bx2, by2 = [int(v) for v in bubble["bbox"]]
            bx1, bx2 = max(0, bx1), min(img.shape[1], bx2)
            by1, by2 = max(0, by1), min(img.shape[0], by2)
            if bx2 - bx1 < 4 or by2 - by1 < 4:
                continue

            roi_mask = m[by1:by2, bx1:bx2]

            # Sample non-masked interior → estimate background brightness
            bg_pixels = img[by1:by2, bx1:bx2][roi_mask <= 128]
            if len(bg_pixels) < 10:
                # Tiny region — remove from remaining (ignore)
                remaining[by1:by2, bx1:bx2][roi_mask > 128] = 0
                continue

            mean_brightness = np.mean(bg_pixels.astype(np.float32))

            if mean_brightness > 220:  # ← WHITE / near-white background
                border = ndimage.binary_dilation(roi_mask > 128, iterations=5) & ~(roi_mask > 128)
                fill_color = (255, 255, 255,)
                if border.sum() > 5:
                    fill_color = tuple(np.median(img[by1:by2, bx1:bx2][border], axis=0).astype(int).tolist())
                result[by1:by2, bx1:bx2][roi_mask > 128] = fill_color
                remaining[by1:by2, bx1:bx2][roi_mask > 128] = 0
            # tone → leave in remaining for LaMa

    else:
        # Legacy path: fill entire mask
        border = ndimage.binary_dilation(m > 128, iterations=5) & ~(m > 128)
        fill_color = (255, 255, 255)
        if border.sum() > 5:
            fill_color = tuple(np.median(img[border], axis=0).astype(int).tolist())
        result[m > 128] = fill_color
        remaining = np.zeros_like(m)  # nothing left

    remaining_img = Image.fromarray(remaining) if np.any(remaining > 128) else None
    return Image.fromarray(result), remaining_img
