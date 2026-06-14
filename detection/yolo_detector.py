"""
YOLO detector — 직접 Python ultralytics 호출 (ComfyUI 우회)
"""
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from ultralytics import YOLO

logger = logging.getLogger(__name__)

DetectionResult = dict
"""
{
  "page_id": str,
  "width": int,
  "height": int,
  "bubbles": [
    {
      "id": "b001",
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.95,
      "type": "bubble",
    }
  ],
  "texts": [
    {
      "id": "t001",
      "bbox": [x1, y1, x2, y2],
      "mask": np.ndarray (H, W, uint8) or None,
      "confidence": 0.85,
    }
  ],
  "text_mask": np.ndarray | None,   # full-page text pixel mask (H, W, uint8)
}
"""


class YoloDetector:
    def __init__(
        self,
        text_segmenter_path: str,
        bubble_detector_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "mps",
    ):
        self.text_model = YOLO(text_segmenter_path)
        self.bubble_model = YOLO(bubble_detector_path)
        self.conf = conf_threshold
        self.iou = iou_threshold
        self.device = device

    def detect(self, image: Image.Image, page_id: str = "000") -> DetectionResult:
        w, h = image.size

        # --- 1. 텍스트 세그멘테이션 ---
        text_results = self.text_model(
            image,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            retina_masks=True,
        )
        tr = text_results[0]

        # 전체 페이지 텍스트 마스크 누적
        text_mask = None
        texts = []
        if tr.masks is not None:
            masks = tr.masks.data.cpu().numpy()  # (N, orig_h, orig_w)
            boxes = tr.boxes.xyxy.cpu().numpy() if tr.boxes is not None else None
            confs = tr.boxes.conf.cpu().numpy() if tr.boxes is not None else None

            # 마스크를 원본 해상도로 리사이즈
            mask_accum = np.zeros((h, w), dtype=np.uint8)
            for i, mask in enumerate(masks):
                # mask shape: (model_h, model_w) — bilinear resize to (h, w)
                mask_img = Image.fromarray((mask * 255).astype(np.uint8))
                mask_resized = np.array(mask_img.resize((w, h), Image.BILINEAR))
                mask_accum = np.maximum(mask_accum, mask_resized)

                bbox = [float(x) for x in boxes[i]] if boxes is not None else None
                conf = float(confs[i]) if confs is not None else 0.0
                texts.append({
                    "id": f"t{i+1:03d}",
                    "bbox": bbox,
                    "confidence": conf,
                })

            text_mask = mask_accum

        # --- 2. 말풍선 검출 ---
        bubble_results = self.bubble_model(
            image,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
        )
        br = bubble_results[0]

        bubbles = []
        if br.boxes is not None:
            boxes = br.boxes.xyxy.cpu().numpy()
            confs = br.boxes.conf.cpu().numpy()
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                # 0-clamp
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if (x2 - x1) * (y2 - y1) < 500:
                    continue
                bubbles.append({
                    "id": f"b{i+1:03d}",
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": float(confs[i]),
                    "type": "bubble",
                })

        result: DetectionResult = {
            "page_id": page_id,
            "width": w,
            "height": h,
            "bubbles": bubbles,
            "texts": texts,
            "text_mask": text_mask,
        }
        return result

    def detect_path(self, image_path: str, page_id: Optional[str] = None) -> DetectionResult:
        if page_id is None:
            page_id = Path(image_path).stem
        image = Image.open(image_path).convert("RGB")
        return self.detect(image, page_id=page_id)
