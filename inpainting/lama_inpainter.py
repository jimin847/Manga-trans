import os
import logging
import urllib.request
import numpy as np
from PIL import Image
from scipy import ndimage

from .base import BaseInpainter

logger = logging.getLogger(__name__)

DEFAULT_MODEL_URL = "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama.onnx"


class LaMaONNXInpainter(BaseInpainter):
    """Standalone local LaMa inpainter using ONNX Runtime with 512x512 patch/crop execution."""

    def __init__(self, config: dict = None):
        super().__init__(config)
        self.model_path = str(self.config.get("model_path", "models/lama.onnx"))
        self.dilate_iterations = int(self.config.get("dilate_iterations", 4))
        self._session = None

    def _ensure_model(self):
        if self._session is not None:
            return
        if not os.path.exists(self.model_path):
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            logger.info(f"Downloading standalone LaMa ONNX model to {self.model_path}...")
            urllib.request.urlretrieve(DEFAULT_MODEL_URL, self.model_path)
            logger.info("Download completed.")
        import onnxruntime
        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._session = onnxruntime.InferenceSession(self.model_path, providers=providers)

    def inpaint(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        self._ensure_model()
        res_arr = np.array(image.convert("RGB"))
        mask_arr = np.array(mask.convert("L")) > 128

        if not np.any(mask_arr):
            return image

        if self.dilate_iterations > 0:
            mask_arr = ndimage.binary_dilation(mask_arr, iterations=self.dilate_iterations)

        labels, num = ndimage.label(mask_arr)
        H, W, _ = res_arr.shape

        for i in range(1, num + 1):
            ys, xs = np.where(labels == i)
            if len(xs) == 0:
                continue
            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # 512x512 centered window around component
            x_start = max(0, min(W - 512, cx - 256))
            y_start = max(0, min(H - 512, cy - 256))
            x_end = min(W, x_start + 512)
            y_end = min(H, y_start + 512)
            x_start = max(0, x_end - 512)
            y_start = max(0, y_end - 512)

            crop_img = res_arr[y_start:y_end, x_start:x_end].astype(np.float32) / 255.0
            crop_mask = mask_arr[y_start:y_end, x_start:x_end].astype(np.float32)

            # Pad to 512x512 if image dimensions are smaller than 512
            pad_h, pad_w = 512 - crop_img.shape[0], 512 - crop_img.shape[1]
            if pad_h > 0 or pad_w > 0:
                crop_img = np.pad(crop_img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
                crop_mask = np.pad(crop_mask, ((0, pad_h), (0, pad_w)), mode="constant")

            in_img = np.transpose(crop_img, (2, 0, 1))[None, ...]
            in_mask = crop_mask[None, None, ...]

            out = self._session.run(["clamp"], {"l_image_": in_img, "l_mask_": in_mask})[0][0]
            out_img = (np.clip(np.transpose(out, (1, 2, 0)), 0, 1) * 255.0).astype(np.uint8)

            ch, cw = min(512, y_end - y_start), min(512, x_end - x_start)
            m_3d = crop_mask[:ch, :cw, None]
            res_arr[y_start:y_end, x_start:x_end] = np.where(m_3d > 0, out_img[:ch, :cw], res_arr[y_start:y_end, x_start:x_end])

        return Image.fromarray(res_arr)
