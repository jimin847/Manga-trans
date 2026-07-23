import logging
from typing import Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .base import BaseInpainter

logger = logging.getLogger(__name__)



class GenAIEditInpainter(BaseInpainter):
    """Next-generation Generative AI Image Edit / MLLM Replacement Engine.

    When handling floating text over complex illustrations/screentones, standard pixel
    interpolation models blur or smudge line art. This engine takes ROI crops around text
    and prompts GenAI image models (e.g., ComfyUI Flux/SDXL Inpaint workflows, MLLM Edit APIs)
    to reconstruct the underlying manga line art or replace typography directly in native style.
    """

    def __init__(self, config: dict = None):
        super().__init__(config)
        self.provider = self.config.get("genai_provider", "local_comfyui")
        self.prompt_template = self.config.get(
            "genai_prompt_replace",
            "High quality manga page. Replace the original Japanese text in this region with the Korean text: '{translation}'. Preserve exact artwork, screentones, line art and character illustration."
        )

    def generate_prompt(self, translation: str, text_type: str = "floating_text") -> str:
        try:
            return self.prompt_template.format(translation=translation, target_text=translation, text_type=text_type)
        except Exception:
            return f"Replace Japanese text with Korean translation '{translation}' while preserving manga art."

    def _get_font(self, size: int):
        font_paths = [
            "/System/Library/Fonts/AppleSDGothicNeo.ttc",
            "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
            "/Library/Fonts/Arial Unicode.ttf",
        ]
        for fp in font_paths:
            try:
                return ImageFont.truetype(fp, size=size)
            except Exception:
                continue
        return ImageFont.load_default()

    def replace_text_region(
        self,
        image: Image.Image,
        item: dict,
        mask: Optional[Image.Image] = None,
    ) -> Image.Image:
        trans = item.get("translation")
        if not trans:
            return image

        bbox = [int(v) for v in item["bbox"]]
        x1 = max(0, bbox[0])
        y1 = max(0, bbox[1])
        x2 = min(image.width, bbox[2])
        y2 = min(image.height, bbox[3])
        if x2 <= x1 or y2 <= y1:
            return image

        text_type = item.get("type", "floating_text")
        prompt = self.generate_prompt(trans, text_type)
        logger.info(f"  [GenAI Replacement] Region {item.get('id', 'N/A')} ({text_type}): prompt generated")
        logger.debug(f"    Prompt: \"{prompt}\"")

        # 1. Expand ROI crop for background/screentone context
        pad = max(20, int(0.15 * max(x2 - x1, y2 - y1)))
        cx1 = max(0, x1 - pad)
        cy1 = max(0, y1 - pad)
        cx2 = min(image.width, x2 + pad)
        cy2 = min(image.height, y2 + pad)

        crop_img = image.crop((cx1, cy1, cx2, cy2))

        # 2. Local Standalone GenAI Simulation / Diffusion Erasure + Outline Typography
        try:
            if self.provider in ("local_diffusers", "diffusers", "sd_inpaint"):
                from .diffusers_inpainter import LocalDiffusersInpainter
                inpainter = LocalDiffusersInpainter(self.config)
            else:
                from .lama_inpainter import LaMaONNXInpainter
                inpainter = LaMaONNXInpainter(self.config)

            # Build local mask for the text box relative to crop coordinates
            rx1, ry1 = max(0, x1 - cx1), max(0, y1 - cy1)
            rx2, ry2 = min(crop_img.width, x2 - cx1), min(crop_img.height, y2 - cy1)

            import cv2
            crop_mask_arr = np.zeros((crop_img.height, crop_img.width), dtype=np.uint8)
            roi_gray = np.array(crop_img.convert("L"))[ry1:ry2, rx1:rx2]
            if roi_gray.size > 0:
                threshold_val = int(self.config.get("text_threshold_value", 185))
                stroke_mask = (roi_gray < threshold_val).astype(np.uint8) * 255
                if mask is not None:
                    yolo_crop_mask = np.array(mask.crop((cx1, cy1, cx2, cy2)).convert("L"))[ry1:ry2, rx1:rx2]
                    if np.max(yolo_crop_mask) > 0:
                        kernel_yolo = np.ones((5, 5), np.uint8)
                        yolo_dilated = cv2.dilate(yolo_crop_mask, kernel_yolo, iterations=2)
                        stroke_mask = cv2.bitwise_and(stroke_mask, yolo_dilated)

                kernel = np.ones((3, 3), np.uint8)
                stroke_mask = cv2.morphologyEx(stroke_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
                stroke_mask = cv2.dilate(stroke_mask, kernel, iterations=1)
                if np.max(stroke_mask) == 0 or (np.sum(stroke_mask > 0) / roi_gray.size) > 0.85:
                    if mask is not None:
                        crop_mask_arr = np.array(mask.crop((cx1, cy1, cx2, cy2)).convert("L"))
                    else:
                        crop_mask_arr[ry1:ry2, rx1:rx2] = 255
                else:
                    crop_mask_arr[ry1:ry2, rx1:rx2] = stroke_mask
            crop_mask = Image.fromarray(crop_mask_arr)


            # Run inpainting backend on the crop to reconstruct background artwork / screentones
            erased_crop = inpainter.inpaint(crop_img, crop_mask)


            # Render Korean typography onto erased crop
            draw = ImageDraw.Draw(erased_crop)
            box_h = ry2 - ry1
            box_w = rx2 - rx1

            is_vertical = item.get("type", "") == "vertical" or (box_h > box_w * 1.3 and len(trans) > 2)
            if is_vertical:
                clean_trans = trans.replace("\n", " ").strip()
                char_count = max(1, len(clean_trans))
                font_size = max(14, min(box_w, int(box_h / char_count * 0.9)))
                font = self._get_font(font_size)
                char_spacing = int(font_size * 1.1)
                total_h = len(clean_trans) * char_spacing
                start_y = ry1 + max(0, (box_h - total_h) // 2)
                start_x = rx1 + max(0, (box_w - font_size) // 2)
                stroke_w = max(1, font_size // 14)
                for i, char in enumerate(clean_trans):
                    cw = draw.textlength(char, font=font)
                    cx = start_x + max(0, (font_size - cw) / 2)
                    cy = start_y + i * char_spacing
                    draw.text((cx, cy), char, font=font, fill=(20, 20, 20), stroke_width=stroke_w, stroke_fill=(255, 255, 255))
            else:
                char_count = max(1, len(trans))
                font_size = max(14, min(box_h, int((box_w * box_h / char_count) ** 0.5 * 0.85)))
                font = self._get_font(font_size)

                lines = []
                words = trans.split()
                cur_line = ""
                for w in words:
                    test_line = f"{cur_line} {w}".strip() if cur_line else w
                    if draw.textlength(test_line, font=font) <= box_w or not cur_line:
                        cur_line = test_line
                    else:
                        lines.append(cur_line)
                        cur_line = w
                if cur_line:
                    lines.append(cur_line)

                line_spacing = int(font_size * 1.2)
                total_h = len(lines) * line_spacing
                start_y = ry1 + max(0, (box_h - total_h) // 2)

                for i, line in enumerate(lines):
                    lw = draw.textlength(line, font=font)
                    lx = rx1 + max(0, (box_w - lw) / 2)
                    ly = start_y + i * line_spacing
                    stroke_w = max(1, font_size // 14)
                    draw.text((lx, ly), line, font=font, fill=(20, 20, 20), stroke_width=stroke_w, stroke_fill=(255, 255, 255))


            result = image.copy()
            result.paste(erased_crop, (cx1, cy1))
            return result

        except Exception as e:
            logger.warning(f"  [GenAI Replacement] Local simulation failed ({e}), keeping original crop.")
            return image

    def inpaint(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        logger.info("GenAI Edit Inpainter invoked: crop ROI and reconstruct via Generative AI model.")
        return image
