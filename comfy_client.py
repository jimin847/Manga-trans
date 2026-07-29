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
    def __init__(self, base_url: str = "http://127.0.0.1:8000", timeout: int = 180):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def check_available(self, timeout: int = 2) -> bool:
        """Return True if ComfyUI responds quickly."""
        try:
            resp = requests.get(f"{self.base_url}/system_stats", timeout=timeout)
            return resp.status_code == 200
        except requests.RequestException:
            return False

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
            history = self._wait_for_result(prompt_id, timeout=self.timeout)
            result = self._download_image(history)
            return result
        except Exception as e:
            logger.error(f"LaMa inpainting failed: {e}")
            return None


def opencv_inpaint(image: Image.Image, mask: Image.Image, radius: int = 3, dilate_iterations: int = 2) -> Image.Image:
    """Small local fallback for remaining text masks when ComfyUI is unavailable."""
    import cv2
    import numpy as np

    mask_l = mask.convert("L")
    if mask_l.getbbox() is None:
        return image.copy()

    img = np.array(image.convert("RGB"))
    mask_arr = np.array(mask_l)
    if dilate_iterations > 0:
        kernel = np.ones((3, 3), np.uint8)
        mask_arr = cv2.dilate(mask_arr, kernel, iterations=dilate_iterations)
    if not np.any(mask_arr > 128):
        return image.copy()
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    inpainted = cv2.inpaint(bgr, mask_arr, radius, cv2.INPAINT_TELEA)
    return Image.fromarray(cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)).convert("RGB")


def flat_fill(image: Image.Image, mask: Image.Image,
              bubbles: Optional[list] = None, texts: Optional[list] = None) -> tuple[Image.Image, Optional[Image.Image]]:
    """
    Smart flat fill.
    When bubbles are provided: judges background brightness using inner text bounding boxes (if available)
    or median brightness to avoid interference from dark speech bubble borders.
    For white bubbles: fills text boxes and dilated text masks completely.
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

            # 1. 배경 밝기 판별: 말풍선 안쪽에 위치한 파란색 글자 박스(texts) 안의 픽셀을 우선 샘플링
            bg_pixels = []
            matched_texts = []
            if texts:
                for t in texts:
                    tx1, ty1, tx2, ty2 = [int(v) for v in t["bbox"]]
                    ix1, iy1 = max(bx1, tx1), max(by1, ty1)
                    ix2, iy2 = min(bx2, tx2), min(by2, ty2)
                    if ix2 > ix1 and iy2 > iy1:
                        matched_texts.append([tx1, ty1, tx2, ty2])
                        t_mask = m[iy1:iy2, ix1:ix2]
                        t_bg = img[iy1:iy2, ix1:ix2][t_mask <= 128]
                        if len(t_bg) > 0:
                            bg_pixels.extend(t_bg)

            # 매칭된 글자 박스 픽셀이 부족하거나 없을 시, 기존 말풍선 박스 내 픽셀 사용
            if len(bg_pixels) < 10:
                bg_pixels = img[by1:by2, bx1:bx2][roi_mask <= 128]

            if len(bg_pixels) < 10:
                remaining[by1:by2, bx1:bx2][roi_mask > 128] = 0
                continue

            # 어두운 외곽선 노이즈 영향을 안 받는 중앙값(median)으로 배경 판별
            median_brightness = np.median(bg_pixels)

            background_std = float(np.std(bg_pixels))
            if median_brightness > 210 and background_std < 35:  # flat white / near-white only
                fill_color = (255, 255, 255)
                # 안티에일리싱(계단현상) 회색 픽셀 잔상을 완전히 제거하기 위해 여유롭게 확장된 마스크 클리닝
                clean_mask = ndimage.binary_dilation(roi_mask > 20, iterations=4)
                border = ndimage.binary_dilation(clean_mask, iterations=3) & ~clean_mask
                if border.sum() > 5:
                    fill_color = tuple(np.median(img[by1:by2, bx1:bx2][border], axis=0).astype(int).tolist())
                result[by1:by2, bx1:bx2][clean_mask] = fill_color
                remaining[by1:by2, bx1:bx2][clean_mask] = 0

                # Never fill the whole text rectangle: it creates visible
                # blocks on gradients and screentones. Threshold augmentation
                # upstream is responsible for adding missed glyph pixels.
            # tone → leave in remaining for LaMa

    else:
        # Legacy path: fill entire mask
        clean_mask = ndimage.binary_dilation(m > 20, iterations=4)
        border = ndimage.binary_dilation(clean_mask, iterations=3) & ~clean_mask
        fill_color = (255, 255, 255)
        if border.sum() > 5:
            fill_color = tuple(np.median(img[border], axis=0).astype(int).tolist())
        result[clean_mask] = fill_color
        remaining = np.zeros_like(m)  # nothing left

    remaining_img = Image.fromarray(remaining) if np.any(remaining > 128) else None
    return Image.fromarray(result), remaining_img
