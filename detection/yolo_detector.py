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
        image_size: int = 1280,
    ):
        self.text_model = YOLO(text_segmenter_path)
        self.bubble_model = YOLO(bubble_detector_path)
        self.conf = conf_threshold
        self.iou = iou_threshold
        self.device = device
        self.image_size = image_size

        # 다중 클래스 모델(예: frame, text, balloon) 사용 시 text 클래스만 추출
        self.text_classes = None
        if hasattr(self.text_model, "names") and isinstance(self.text_model.names, dict):
            if len(self.text_model.names) > 1:
                t_cls = [k for k, v in self.text_model.names.items() if "text" in str(v).lower()]
                if t_cls:
                    self.text_classes = t_cls

    def detect(self, image: Image.Image, page_id: str = "000") -> DetectionResult:
        w, h = image.size

        # --- 1. 말풍선 검출 ---
        bubble_results = self.bubble_model(
            image,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            imgsz=self.image_size,
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
                    "id": f"b{len(bubbles)+1:03d}",
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": float(confs[i]),
                    "type": "bubble",
                })

        # --- 2. 텍스트 영역 감지 및 분류 ---
        text_kwargs = {
            "conf": self.conf,
            "iou": self.iou,
            "device": self.device,
            "imgsz": self.image_size,
            "retina_masks": True,
        }
        if self.text_classes is not None:
            text_kwargs["classes"] = self.text_classes

        text_results = self.text_model(image, **text_kwargs)
        tr = text_results[0]

        bubble_mask_accum = np.zeros((h, w), dtype=np.uint8)
        floating_mask_accum = np.zeros((h, w), dtype=np.uint8)
        all_mask_accum = np.zeros((h, w), dtype=np.uint8)
        texts = []
        floating_texts = []

        if tr.masks is not None:
            masks = tr.masks.data.cpu().numpy()  # (N, orig_h, orig_w)
            boxes = tr.boxes.xyxy.cpu().numpy() if tr.boxes is not None else None
            confs = tr.boxes.conf.cpu().numpy() if tr.boxes is not None else None

            for i, mask in enumerate(masks):
                mask_img = Image.fromarray((mask * 255).astype(np.uint8))
                mask_resized = np.array(mask_img.resize((w, h), Image.BILINEAR))
                all_mask_accum = np.maximum(all_mask_accum, mask_resized)

                bbox = [float(x) for x in boxes[i]] if boxes is not None else [0.0, 0.0, float(w), float(h)]
                conf = float(confs[i]) if confs is not None else 0.0

                tx1, ty1, tx2, ty2 = bbox
                tarea = max(1.0, (tx2 - tx1) * (ty2 - ty1))

                # Check intersection with detected bubbles
                in_bubble = False
                for b in bubbles:
                    bx1, by1, bx2, by2 = b["bbox"]
                    ix1 = max(tx1, bx1)
                    iy1 = max(ty1, by1)
                    ix2 = min(tx2, bx2)
                    iy2 = min(ty2, by2)
                    if ix2 > ix1 and iy2 > iy1:
                        iarea = (ix2 - ix1) * (iy2 - iy1)
                        if iarea / tarea >= 0.35:
                            in_bubble = True
                            break

                texts.append({
                    "id": f"t{i+1:03d}",
                    "bbox": bbox,
                    "confidence": conf,
                    "in_bubble": in_bubble,
                })

                if in_bubble:
                    bubble_mask_accum = np.maximum(bubble_mask_accum, mask_resized)
                else:
                    floating_mask_accum = np.maximum(floating_mask_accum, mask_resized)
                    if tarea >= 400:
                        floating_texts.append({
                            "id": f"f{len(floating_texts)+1:03d}",
                            "bbox": bbox,
                            "confidence": conf,
                            "type": "floating_text",
                        })

        result: DetectionResult = {
            "page_id": page_id,
            "width": w,
            "height": h,
            "bubbles": bubbles,
            "texts": texts,
            "floating_texts": floating_texts,
            "text_mask": bubble_mask_accum if tr.masks is not None else None,
            "floating_text_mask": floating_mask_accum if tr.masks is not None else None,
            "all_text_mask": all_mask_accum if tr.masks is not None else None,
        }
        return result

    def detect_path(self, image_path: str, page_id: Optional[str] = None) -> DetectionResult:
        if page_id is None:
            page_id = Path(image_path).stem
        image = Image.open(image_path).convert("RGB")
        return self.detect(image, page_id=page_id)
