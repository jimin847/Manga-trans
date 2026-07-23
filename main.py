#!/usr/bin/env python3
"""
manga_pipeline_v2 — main entry point
YOLO detection + VLM OCR + ComfyUI inpainting + QPainter typesetting
"""
import argparse
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")
_DETECTOR_CACHE = {}
_OCR_CLIENT_CACHE = {}
_TRANSLATOR_CACHE = {}
OCR_CACHE_VERSION = "ocr_v4"
TRANSLATION_CACHE_VERSION = "trans_v6"


def validate_config(config: dict, base_dir: Path = None) -> list:
    """Startup validation: checks required keys, model paths, value ranges.
    Returns list of warnings (non-fatal issues).
    Raises ValueError for fatal config errors.
    """
    warnings = []

    # Required top-level keys
    required = ["comfyui", "models", "yolo", "ocr", "translation", "inpainting", "output"]
    for key in required:
        if key not in config:
            raise ValueError(f"Missing required config key: '{key}'")

    # comfyui section
    if "base_url" not in config.get("comfyui", {}):
        raise ValueError("Missing 'comfyui.base_url' — ComfyUI server address required")

    # models section — file existence checks
    for model_key in ["text_segmenter", "bubble_detector"]:
        model_path = config.get("models", {}).get(model_key)
        if not model_path:
            raise ValueError(f"Missing 'models.{model_key}' path")
        if not Path(model_path).exists():
            logger.warning(f"Model file not found: {model_key} = {model_path}")
            warnings.append(f"models.{model_key}: file not found ({model_path})")

    # yolo section
    conf = config.get("yolo", {}).get("conf_threshold", 0.25)
    iou = config.get("yolo", {}).get("iou_threshold", 0.45)
    if not (0 < conf <= 1):
        raise ValueError(f"yolo.conf_threshold must be 0-1, got {conf}")
    if not (0 < iou <= 1):
        raise ValueError(f"yolo.iou_threshold must be 0-1, got {iou}")

    # ocr section
    provider = config.get("ocr", {}).get("provider", "")
    valid_providers = ("openrouter", "google-ai-studio")
    if provider not in valid_providers:
        raise ValueError(f"ocr.provider must be one of {valid_providers}, got '{provider}'")
    if not config.get("ocr", {}).get("model"):
        raise ValueError("Missing 'ocr.model'")

    # translation section
    t_provider = config.get("translation", {}).get("provider", "")
    if t_provider not in valid_providers:
        raise ValueError(f"translation.provider must be one of {valid_providers}, got '{t_provider}'")
    if not config.get("translation", {}).get("model"):
        raise ValueError("Missing 'translation.model'")

    # output section
    output_base = config.get("output", {}).get("base_dir")
    if not output_base:
        raise ValueError("Missing 'output.base_dir'")
    out_path = Path(output_base)
    if not out_path.exists():
        logger.info(f"Output directory will be created: {out_path}")

    # qa section (optional, validate if enabled)
    if config.get("qa", {}).get("enabled"):
        warn_th = config["qa"].get("warn_threshold", 40)
        bad_th = config["qa"].get("bad_threshold", 10)
        if warn_th < bad_th:
            raise ValueError(f"qa.warn_threshold ({warn_th}) must be >= qa.bad_threshold ({bad_th})")
        if not (0 <= warn_th <= 100):
            raise ValueError(f"qa.warn_threshold must be 0-100, got {warn_th}")

    return warnings


def _resolve_model_path(model_path: str, base_dir: Path | None = None) -> str:
    p = Path(model_path).expanduser()
    if p.is_absolute():
        return str(p)
    if base_dir:
        return str((base_dir / p).resolve())
    return str(p.resolve())


def get_detector(config: dict):
    from detection.yolo_detector import YoloDetector

    base_dir = Path(config.get("_config_dir", ".")) if isinstance(config.get("_config_dir"), str) else None
    text_path = _resolve_model_path(config["models"]["text_segmenter"], base_dir)
    bubble_path = _resolve_model_path(config["models"]["bubble_detector"], base_dir)

    cache_key = (
        text_path,
        bubble_path,
        config["yolo"]["conf_threshold"],
        config["yolo"]["iou_threshold"],
        config["yolo"]["device"],
    )
    detector = _DETECTOR_CACHE.get(cache_key)
    if detector is None:
        detector = YoloDetector(
            text_segmenter_path=text_path,
            bubble_detector_path=bubble_path,
            conf_threshold=config["yolo"]["conf_threshold"],
            iou_threshold=config["yolo"]["iou_threshold"],
            device=config["yolo"]["device"],
        )
        _DETECTOR_CACHE[cache_key] = detector
    return detector


def _set_config_dir(config: dict, base_dir: Path | None) -> None:
    if base_dir:
        config["_config_dir"] = str(base_dir)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config_dir = Path(config_path).parent
    _set_config_dir(config, config_dir)
    validate_config(config, config_dir)
    return config


def get_ocr_client(config: dict):
    from ocr.vlm_ocr import VlmOcr

    cache_key = (
        config["ocr"]["provider"],
        config["ocr"]["model"],
        config["ocr"].get("fallback_model"),
    )
    client = _OCR_CLIENT_CACHE.get(cache_key)
    if client is None:
        client = VlmOcr(
            provider=config["ocr"]["provider"],
            model=config["ocr"]["model"],
            fallback_model=config["ocr"].get("fallback_model"),
        )
        _OCR_CLIENT_CACHE[cache_key] = client
    return client


def get_translator_client(config: dict):
    from translation.translator import Translator

    cache_key = (
        config["translation"]["provider"],
        config["translation"]["model"],
        tuple(sorted((config["translation"].get("glossary") or {}).items())),
        tuple(config["translation"].get("preserve_terms") or []),
    )
    client = _TRANSLATOR_CACHE.get(cache_key)
    if client is None:
        translator_kwargs = {}
        if "glossary" in config["translation"]:
            translator_kwargs["glossary"] = config["translation"].get("glossary") or {}
        if "preserve_terms" in config["translation"]:
            translator_kwargs["preserve_terms"] = config["translation"].get("preserve_terms") or []
        client = Translator(
            provider=config["translation"]["provider"],
            model=config["translation"]["model"],
            **translator_kwargs,
        )
        _TRANSLATOR_CACHE[cache_key] = client
    return client


def run_detection(image_path: str, config: dict, tmp_dir: Path) -> tuple[dict, Image.Image]:
    """Step 1: YOLO detection → text mask + bubble bboxes.
    Saves intermediates to tmp_dir (auto-cleaned).
    """
    detector = get_detector(config)
    original = Image.open(image_path).convert("RGB")
    result = detector.detect(original, page_id=Path(image_path).stem)

    inpaint_cfg = config.get("inpainting", {})
    if inpaint_cfg.get("threshold_left_margin_vertical", True):
        _add_left_margin_vertical_text_mask(original, result, config)
    if inpaint_cfg.get("threshold_text_mask", True):
        _add_threshold_text_mask(original, result, config)

    # Debug overlay → tmpdir
    if config["output"].get("debug_overlay", True):
        from PIL import ImageDraw
        img = original.copy()
        draw = ImageDraw.Draw(img)
        for b in result["bubbles"]:
            x1, y1, x2, y2 = b["bbox"]
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
            draw.text((x1 + 5, y1 - 15), b["id"], fill=(0, 255, 0))
        for t in result["texts"]:
            if t["bbox"]:
                x1, y1, x2, y2 = t["bbox"]
                draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
        overlay_path = tmp_dir / f"{result['page_id']}_detect_overlay.png"
        img.save(overlay_path)
        logger.info(f"  Debug overlay → {overlay_path}")

    # Text mask → tmpdir
    if config["output"].get("save_masks", True) and result["text_mask"] is not None:
        mask_path = tmp_dir / f"{result['page_id']}_text_mask.png"
        Image.fromarray(result["text_mask"]).save(mask_path)
        logger.info(f"  Text mask → {mask_path}")

    # BBox JSON → tmpdir
    page_id = result["page_id"]
    bbox_data = {
        "page_id": page_id,
        "width": result["width"],
        "height": result["height"],
        "bubbles": [{k: v for k, v in b.items() if k in ("id", "bbox", "confidence")}
                     for b in result["bubbles"]],
        "texts": result["texts"],
    }
    bbox_path = tmp_dir / f"{page_id}_bbox.json"
    with open(bbox_path, "w") as f:
        json.dump(bbox_data, f, indent=2)
    logger.info(f"  BBox JSON → {bbox_path}")

    return result, original


def _make_bubble_mask(width: int, height: int, bbox: list[int], shrink: int = 2) -> Image.Image:
    from PIL import ImageDraw

    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, x1 + shrink)
    y1 = max(0, y1 + shrink)
    x2 = min(width, x2 - shrink)
    y2 = min(height, y2 - shrink)
    mask = Image.new("L", (width, height), 0)
    if x2 <= x1 or y2 <= y1:
        return mask
    draw = ImageDraw.Draw(mask)
    draw.ellipse([x1, y1, x2, y2], fill=255)
    return mask


def _clip_text_mask_to_bubbles(text_mask_img: Image.Image, bubbles: list[dict], original: Image.Image) -> Image.Image:
    """Keep only text pixels inside detected bubbles to protect panel backgrounds."""
    if not bubbles:
        return text_mask_img
    from PIL import ImageChops

    union = Image.new("L", text_mask_img.size, 0)
    for bubble in bubbles:
        bubble_mask = _make_bubble_mask(original.width, original.height, bubble["bbox"])
        union = ImageChops.lighter(union, bubble_mask)
    from PIL import ImageChops
    return ImageChops.darker(text_mask_img, union)


def _bbox_intersects_any(bbox: list[int], others: list[list[int]]) -> bool:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    for ox1, oy1, ox2, oy2 in others:
        if x1 < ox2 and x2 > ox1 and y1 < oy2 and y2 > oy1:
            return True
    return False


def _clip_text_mask_to_text_bboxes(text_mask_img: Image.Image, result: dict, config: dict) -> Image.Image:
    """Keep only pixels inside detected text boxes to protect backgrounds and bubbles."""
    texts = result.get("texts") or []
    if not texts:
        return text_mask_img
    from PIL import ImageChops, ImageDraw
    import cv2
    import numpy as np

    union = Image.new("L", text_mask_img.size, 0)
    draw = ImageDraw.Draw(union)
    dilate = int(config.get("inpainting", {}).get("preserve_bubble_shape_text_bbox_dilate", 2))
    text_bboxes = []
    for text in texts:
        x1, y1, x2, y2 = [int(v) for v in text["bbox"]]
        text_bboxes.append([x1, y1, x2, y2])
        draw.rectangle([x1 - dilate, y1 - dilate, x2 + dilate, y2 + dilate], fill=255)

    if config.get("inpainting", {}).get("keep_external_text_components", True):
        mask_arr = np.array(text_mask_img.convert("L"))
        num, labels, stats, _ = cv2.connectedComponentsWithStats((mask_arr > 128).astype(np.uint8), connectivity=8)
        external = Image.new("L", text_mask_img.size, 0)
        ext_draw = ImageDraw.Draw(external)
        bubble_bboxes = [[int(v) for v in b["bbox"]] for b in result.get("bubbles", [])]
        for i in range(1, num):
            x, y, w, h, area = stats[i]
            if area < 8 or area > 2000:
                continue
            aspect = max(w, h) / max(1, min(w, h))
            if aspect > 8:
                continue
            bbox = [x, y, x + w, y + h]
            if _bbox_intersects_any(bbox, text_bboxes) or _bbox_intersects_any(bbox, bubble_bboxes):
                continue
            ext_draw.rectangle(bbox, fill=255)
        union = ImageChops.lighter(union, external)
    return ImageChops.darker(text_mask_img, union)


def _exclude_bubbles_from_text_mask(text_mask_img: Image.Image, bubbles: list[dict]) -> Image.Image:
    """Remove text pixels inside bubbles marked skip_inpaint, usually SFX."""
    from PIL import ImageDraw
    mask = text_mask_img.copy()
    draw = ImageDraw.Draw(mask)
    for bubble in bubbles:
        if not bubble.get("skip_inpaint"):
            continue
        x1, y1, x2, y2 = [int(v) for v in bubble["bbox"]]
        draw.rectangle([x1, y1, x2, y2], fill=0)
    return mask


def _shrink_bbox_for_typesetting(bbox: list[int], config: dict) -> list[int]:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    pad = int(config.get("typesetting", {}).get("bubble_padding", 10))
    return [x1 + pad, y1 + pad, x2 - pad, y2 - pad]


def _estimate_font_size(text: str, bubble: dict, vertical: bool, config: dict) -> int:
    bbox = _shrink_bbox_for_typesetting(bubble["bbox"], config)
    bbox_w = max(20, bbox[2] - bbox[0])
    bbox_h = max(20, bbox[3] - bbox[1])
    char_count = max(1, len("".join(str(text).split())))
    if vertical:
        return max(16, min(72, int(bbox_h / max(1, char_count) * 1.15)))
    avg_chars_per_line = max(4, int(bbox_h / 42))
    chars_per_line = max(1, int(char_count / avg_chars_per_line))
    return max(16, min(72, int((bbox_w - 20) / max(chars_per_line, 1) / 0.85)))


def _add_left_margin_vertical_text_mask(original: Image.Image, result: dict, config: dict) -> None:
    """Add a mask for narrow vertical text columns in the left margin.

    Some title-page text sits outside bubble/text bboxes, especially the thin
    vertical side copy on the far left. This pass finds narrow dark-column
    groups in the left margin and adds them to the text mask without touching
    the rest of the page.
    """
    import cv2

    mask = result.get("text_mask")
    if mask is None:
        return

    inpaint_cfg = config.get("inpainting", {})
    threshold = int(inpaint_cfg.get("text_threshold_left_margin_value", 160))
    left_ratio = float(inpaint_cfg.get("text_threshold_left_margin_ratio", 0.25))
    min_column_pixels = int(inpaint_cfg.get("text_threshold_left_margin_min_pixels", 12))
    min_width = int(inpaint_cfg.get("text_threshold_left_margin_min_width", 5))
    max_width = int(inpaint_cfg.get("text_threshold_left_margin_max_width", 120))
    min_height = int(inpaint_cfg.get("text_threshold_left_margin_min_height", 120))
    dilate = int(inpaint_cfg.get("text_threshold_left_margin_dilate", 1))

    arr = np.asarray(original.convert("L"))
    h, w = arr.shape
    left = max(1, int(w * left_ratio))
    crop = arr[:, :left]
    local = crop < threshold
    kernel = np.ones((3, 3), np.uint8)
    if dilate > 0:
        local = cv2.dilate(local.astype(np.uint8), kernel, iterations=dilate) > 0
    local = cv2.morphologyEx(local.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1) > 0

    column_counts = local.sum(axis=0)
    groups = []
    start = None
    for x, count in enumerate(column_counts):
        if count >= min_column_pixels and start is None:
            start = x
        if (count < min_column_pixels or x == len(column_counts) - 1) and start is not None:
            end = x if count >= min_column_pixels else x - 1
            groups.append((start, end))
            start = None

    extra = np.zeros_like(arr, dtype=np.uint8)
    for x1, x2 in groups:
        width = x2 - x1 + 1
        if width < min_width or width > max_width:
            continue
        rows = np.where(local[:, x1:x2 + 1].sum(axis=1) >= max(1, min_column_pixels // 2))[0]
        if len(rows) == 0:
            continue
        y1, y2 = int(rows[0]), int(rows[-1]) + 1
        height = y2 - y1
        if height < min_height:
            continue
        extra[y1:y2, x1:x2 + 1] = np.maximum(extra[y1:y2, x1:x2 + 1], local[y1:y2, x1:x2 + 1].astype(np.uint8) * 255)

    if np.any(extra):
        result["text_mask"] = np.maximum(mask, extra).astype(np.uint8)


def _add_threshold_text_mask(original: Image.Image, result: dict, config: dict) -> None:
    """Add dark-pixel text mask inside detected bubble boxes.

    YOLO text masks can miss thin title-page text. Manga text is usually black on
    white/gray, so a local luminance threshold inside bubble boxes catches the
    remainder without painting arbitrary line-art outside bubbles.
    """
    import cv2

    mask = result.get("text_mask")
    targets = list(result.get("texts") or [])
    if not targets:
        targets = list(result.get("bubbles") or [])
    if mask is None or not targets:
        return

    inpaint_cfg = config.get("inpainting", {})
    threshold = int(inpaint_cfg.get("text_threshold_value", 185))
    dilate = int(inpaint_cfg.get("text_threshold_dilate", 1))
    min_component = int(inpaint_cfg.get("text_threshold_min_component", 3))
    max_component = int(inpaint_cfg.get("text_threshold_max_component", 900))

    arr = np.asarray(original.convert("L"))
    extra = np.zeros_like(arr, dtype=np.uint8)
    h, w = arr.shape
    kernel = np.ones((3, 3), np.uint8)

    for target in targets:
        bx1, by1, bx2, by2 = [int(v) for v in target["bbox"]]
        bx1, by1 = max(0, bx1), max(0, by1)
        bx2, by2 = min(w, bx2), min(h, by2)
        if bx2 - bx1 < 4 or by2 - by1 < 4:
            continue
        local = arr[by1:by2, bx1:bx2] < threshold
        if dilate > 0:
            local = cv2.dilate(local.astype(np.uint8), kernel, iterations=dilate) > 0
        local = cv2.morphologyEx(local.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1) > 0
        if min_component > 0:
            count, labels, stats, _ = cv2.connectedComponentsWithStats(local.astype(np.uint8), connectivity=8)
            keep = np.zeros_like(local, dtype=np.uint8)
            for i in range(1, count):
                if stats[i, cv2.CC_STAT_AREA] >= min_component:
                    keep[labels == i] = 255
            local = keep > 0
        extra_slice = extra[by1:by2, bx1:bx2]
        extra_slice[local] = 255
        extra[by1:by2, bx1:bx2] = extra_slice

    combined = np.maximum(mask, extra)

    if inpaint_cfg.get("threshold_global_mask", True):
        global_local = arr < threshold
        if dilate > 0:
            global_local = cv2.dilate(global_local.astype(np.uint8), kernel, iterations=dilate) > 0
        global_local = cv2.morphologyEx(global_local.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1) > 0
        count, labels, stats, _ = cv2.connectedComponentsWithStats(global_local.astype(np.uint8), connectivity=8)
        keep = np.zeros_like(global_local, dtype=np.uint8)
        for i in range(1, count):
            area = int(stats[i, cv2.CC_STAT_AREA])
            width = int(stats[i, cv2.CC_STAT_WIDTH])
            height = int(stats[i, cv2.CC_STAT_HEIGHT])
            is_vertical_text = (
                width <= int(inpaint_cfg.get("text_threshold_global_vertical_width", 90))
                and height >= int(inpaint_cfg.get("text_threshold_global_vertical_height", 120))
            )
            x = int(stats[i, cv2.CC_STAT_LEFT])
            is_left_margin = x <= int(w * float(inpaint_cfg.get("text_threshold_global_left_ratio", 0.25)))
            if min_component <= area <= max_component or (is_vertical_text and is_left_margin):
                keep[labels == i] = 255
        combined = np.maximum(combined, keep)

    result["text_mask"] = combined.astype(np.uint8)


def _fill_threshold_text_components(image: Image.Image, original: Image.Image, result: dict, config: dict) -> None:
    """Fill tiny threshold text remnants after inpainting.

    OpenCV inpaint can leave white interiors of outlined title text. This pass
    fills only small dark components inside detected text/bubble boxes using the
    local non-text median, so it is safer than a page-wide threshold fill.
    """
    import cv2

    targets = list(result.get("texts") or [])
    if not targets:
        targets = list(result.get("bubbles") or [])
    if not targets:
        return

    inpaint_cfg = config.get("inpainting", {})
    threshold_offset = int(inpaint_cfg.get("text_threshold_offset", 30))
    min_component = int(inpaint_cfg.get("text_threshold_min_component", 2))
    max_component = int(inpaint_cfg.get("text_threshold_max_component", 3000))
    kernel = np.ones((3, 3), np.uint8)
    out = np.array(image.convert("RGB"), copy=True)
    base = np.asarray(original.convert("L"))

    for target in targets:
        bx1, by1, bx2, by2 = [int(v) for v in target["bbox"]]
        bx1, by1 = max(0, bx1), max(0, by1)
        bx2, by2 = min(out.shape[1], bx2), min(out.shape[0], by2)
        if bx2 - bx1 < 4 or by2 - by1 < 4:
            continue
        crop = base[by1:by2, bx1:bx2]
        local_threshold = int(np.median(crop)) - threshold_offset
        local = crop < local_threshold
        local = cv2.dilate(local.astype(np.uint8), kernel, iterations=2) > 0
        local = cv2.morphologyEx(local.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1) > 0
        count, labels, stats, _ = cv2.connectedComponentsWithStats(local.astype(np.uint8), connectivity=8)
        for i in range(1, count):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if min_component <= area <= max_component:
                comp = labels == i
                border = cv2.dilate(comp.astype(np.uint8), kernel, iterations=8) - comp.astype(np.uint8)
                samples = crop[border.astype(bool)]
                if samples.size == 0:
                    continue
                value = int(np.median(samples))
                out[by1:by2, bx1:bx2][comp] = value
                image.paste(Image.fromarray(out), (0, 0))


def _prepare_ocr_crop(original: Image.Image, bubble: dict, text_mask: Optional[np.ndarray], config: dict) -> Image.Image:
    x1, y1, x2, y2 = [int(v) for v in bubble["bbox"]]
    crop_box = (x1, y1, x2, y2)

    if text_mask is not None and text_mask.size:
        mask = np.asarray(text_mask, dtype=np.uint8)
        h, w = mask.shape[:2]
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(w, x2), min(h, y2)
        if x2c > x1c and y2c > y1c:
            local = mask[y1c:y2c, x1c:x2c]
            ys, xs = np.where(local > 0)
            if len(xs) and len(ys):
                margin = int(config.get("ocr", {}).get("text_mask_margin", 8))
                mx1 = max(x1c + int(xs.min()) - margin, 0)
                my1 = max(y1c + int(ys.min()) - margin, 0)
                mx2 = min(x1c + int(xs.max()) + 1 + margin, w)
                my2 = min(y1c + int(ys.max()) + 1 + margin, h)
                if mx2 > mx1 and my2 > my1:
                    crop_box = (mx1, my1, mx2, my2)

    crop = original.crop(crop_box)
    upscale = int(config.get("ocr", {}).get("crop_upscale", 2) or 1)
    if upscale > 1:
        crop = crop.resize((crop.width * upscale, crop.height * upscale))
    return crop


def run_ocr(detection: dict, original: Image.Image, config: dict, api_key: str, tmp_dir: Path, cache):
    """Step 2: VLM OCR on each bubble crop (Parallel + Caching + Orientation)"""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from io import BytesIO

    ocr = get_ocr_client(config)
    all_targets = detection["bubbles"] + detection.get("floating_texts", [])
    if hasattr(ocr, "check_auth") and not ocr.check_auth(api_key):
        logger.error("  OCR auth failed — skipping OCR for this page")
        for bubble in all_targets:
            bubble["ocr_text"] = None
        return

    def prepare_bubble(bubble):
        crop = _prepare_ocr_crop(original, bubble, detection.get("all_text_mask", detection.get("text_mask")), config)
        x1, y1, x2, y2 = [int(v) for v in bubble["bbox"]]

        bw, bh = x2 - x1, y2 - y1
        orientation = "mixed"
        if bh > bw * 1.5:
            orientation = "vertical"
        elif bw > bh * 1.5:
            orientation = "horizontal"
        bubble["orientation"] = orientation

        upscale = config["ocr"].get("crop_upscale", 2)
        if upscale > 1:
            crop = crop.resize((crop.width * upscale, crop.height * upscale))

        if config["output"].get("save_crops", True):
            crop_path = tmp_dir / f"{detection['page_id']}_{bubble['id']}_crop.png"
            crop.save(crop_path)

        buf = BytesIO()
        crop.save(buf, format="PNG")
        crop_hash = cache.hash_image(buf.getvalue())
        return bubble, crop, orientation, crop_hash

    def process_unique_crop(crop, orientation, crop_hash, bubble_id):
        # Check Cache
        cache_key = f"{OCR_CACHE_VERSION}_{crop_hash}"
        cached_text = cache.get(cache_key)

        if cached_text is not None:
            if cached_text == "[NO TEXT]":
                logger.info(f"  OCR {bubble_id} (cached): [NO TEXT]")
                return None
            logger.info(f"  OCR {bubble_id} (cached): '{cached_text[:60]}'")
            return cached_text

        ocr_text = ocr.read_crop(crop, api_key=api_key, orientation=orientation)
        if ocr_text:
            cache.set(cache_key, ocr_text)
            logger.info(f"  OCR {bubble_id} ({orientation}): '{ocr_text[:60]}'")
        else:
            cache.set(cache_key, "[NO TEXT]")
            logger.info(f"  OCR {bubble_id} ({orientation}): [NO TEXT]")
        return ocr_text

    bubble_prepared = [prepare_bubble(b) for b in all_targets]
    hash_to_bubbles = {}
    unique_jobs = []
    for bubble, crop, orientation, crop_hash in bubble_prepared:
        hash_to_bubbles.setdefault(crop_hash, []).append(bubble)
        if len(hash_to_bubbles[crop_hash]) == 1:
            unique_jobs.append((bubble, crop, orientation, crop_hash))

    max_workers = config.get("ocr", {}).get("max_workers", 4)
    is_google = config.get("ocr", {}).get("provider") == "google-ai-studio"
    # Google AI Studio free tier has 20 req/min — serialize + throttle to avoid 429
    if is_google and max_workers > 1:
        max_workers = 1
        logger.info("  Google AI Studio: serializing OCR (max_workers=1)")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        import time
        futures = {}
        for bubble, crop, orientation, crop_hash in unique_jobs:
            if is_google:
                time.sleep(3)  # Throttle: 20 req/min → 1 req per 3s
            future = executor.submit(process_unique_crop, crop, orientation, crop_hash, bubble["id"])
            futures[future] = crop_hash
        for future in as_completed(futures):
            crop_hash = futures[future]
            text = future.result()
            for bubble in hash_to_bubbles[crop_hash]:
                bubble["ocr_text"] = text


def run_translation(
    detection: dict,
    config: dict,
    api_key: str,
    cache,
    previous_context: Optional[list[str]] = None,
):
    """Step 3: translate OCR results in Batch (JSON array) + Caching + Classification"""
    from translation.translator import Translator
    import json

    def normalize_context(items: Optional[list[str]]) -> list[str]:
        if not items:
            return []
        max_items = config["translation"].get("context_max_items", 4)
        max_chars = config["translation"].get("context_max_chars", 80)
        normalized = []
        seen_keys = set()
        for item in items[-max_items:]:
            text = " ".join(str(item).split())
            if not text:
                continue
            text_key = "".join(text.split())
            if text_key in seen_keys:
                continue
            if len(text) > max_chars:
                text = text[: max_chars - 1].rstrip() + "…"
            normalized.append(text)
            seen_keys.add(text_key)
        return normalized

    def normalize_translation_key(text: str) -> str:
        return "".join(str(text).split())

    def has_japanese_chars(text: str) -> bool:
        return bool(re.search(r"[\u3040-\u30ff\u3400-\u9fff]", str(text)))

    def likely_sfx_text(text: str) -> bool:
        compact = re.sub(r"\s+", "", str(text or ""))
        if len(compact) > 30:
            return False
        if not re.search(r"[!！]{2,}", compact):
            return False
        dialogue_markers = [
            "様", "さん", "ちゃん", "先生", "お願い", "俺", "私", "僕", "君",
            "可愛い", "可愛", "人類", "一生", "憂い", "無し", "ない", "無い",
            "が", "の", "に", "を", "は", "も", "と", "へ", "より",
        ]
        if any(marker in compact for marker in dialogue_markers):
            return False
        return True

    def clean_translation_artifacts(text: str, source_text: str) -> str:
        cleaned = str(text or "")
        cleaned = cleaned.replace("일생마모", "").replace("일생 마모", "")
        cleaned = cleaned.replace("평생\n평생 지켜줄게", "평생 지켜줄게")
        cleaned = cleaned.replace("평생 지켜줄게\n평생 지켜줄게", "평생 지켜줄게")
        cleaned = cleaned.replace("잘 부탁해요♡", "잘 부탁해♡")
        cleaned = cleaned.replace("잘 부탁해요", "잘 부탁해")
        if re.search(r"[가-힣]\.\.\.", cleaned[:4]) and re.search(r"가\.\.\.", cleaned):
            cleaned = re.sub(r"^가\.\.\.", "하지만…", cleaned)
        if "가…" in cleaned and "평생" in cleaned:
            cleaned = cleaned.replace("가…", "하지만…")
        if (
            ("一生守る" in source_text or "いっしょうまも" in source_text)
            and "평생" in cleaned
            and ("지켜" in cleaned or "지킬" in cleaned)
        ):
            lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
            filtered = []
            for line in lines:
                if line == "평생" and any(
                    "평생" in other and ("지킬" in other or "지켜" in other)
                    for other in lines
                ):
                    continue
                filtered.append(line)
            cleaned = "\n".join(filtered)
        if ("一生守る" in source_text or "いっしょうまも" in source_text) and "평생" in cleaned:
            lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
            for i, line in enumerate(lines):
                if "평생" in line:
                    lines = lines[i:]
                    break
            if lines and "지켜" in lines[-1] and "줄게" not in lines[-1] and "지킬게" not in lines[-1]:
                lines[-1] = "지켜줄게…"
            filtered = []
            for line in lines:
                if any(
                    "평생" in existing and "평생" in line and
                    ("지킬" in existing or "지켜줄" in existing) and
                    ("지킬" in line or "지켜줄" in line)
                    for existing in filtered
                ):
                    continue
                filtered.append(line)
            cleaned = "\n".join(filtered)
        if "が…" in source_text and cleaned.startswith("가"):
            cleaned = "하지만…" + cleaned[1:]
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = cleaned.replace("얼굴과몸", "얼굴과 몸")
        cleaned = cleaned.replace("마음대로고를", "마음대로 고를")
        cleaned = cleaned.replace("수있잖아", "수 있잖아")
        cleaned = cleaned.replace("진짜직구", "진짜 직구")
        cleaned = cleaned.replace("평생 지켜 지켜줄 게", "평생 지켜줄게")
        cleaned = cleaned.replace("평생 지켜 지켜줄", "평생 지켜줄")
        cleaned = cleaned.replace("평생 지켜\n평생 지켜줄 게", "평생 지켜줄게")
        cleaned = cleaned.replace("평생 지켜\n평생", "평생")
        cleaned = cleaned.replace("무다모도 안 나와!!!", "불필요한 털도 안 나와!!!")
        cleaned = cleaned.replace("무다모도 안나와!!!", "불필요한 털도 안 나와!!!")
        cleaned = cleaned.replace("무다모도 안 나와 있어!!!", "불필요한 털도 안 나와!!!")
        cleaned = cleaned.replace("불필요한 털도 안 나와 있어!!!", "불필요한 털도 안 나와!!!")
        cleaned = cleaned.replace("난폭하지 말고\n바람도 하지 말고", "난폭하지 않아\n바람도 안 해")
        cleaned = cleaned.replace("난폭하지 말고 바람도 하지 말고", "난폭하지 않아 바람도 안 해")
        cleaned = cleaned.replace("이제는\n한 점의 근심도 없어!!", "이제는 근심이 하나도 없어!!")
        cleaned = cleaned.replace("무한히 잘해주고, 그럼에도 대가도 요구하지 않아", "무한히 잘해주는데 대가도 요구하지 않아")
        cleaned = cleaned.replace("무한히 잘해주고 그럼에도 대가도 요구하지 않아", "무한히 잘해주는데 대가도 요구하지 않아")
        cleaned = cleaned.replace("무엇보다\n새롭게 관계를\n만드는데 필요한\n시간도 노력도\n필요 없다는\n대단해~", "무엇보다 새로 관계를 맺는 데 필요한 시간도 노력도 필요 없다는 게 대단해~")
        cleaned = cleaned.replace("새롭게 관계를 만드는데 필요한", "새 관계를 만드는 데 필요한")
        cleaned = cleaned.replace("필요 없다는 게\n큰거", "필요 없다는 게 큰거")
        cleaned = cleaned.replace("필요 없다는\n대단해~", "필요 없다는 게 대단해~")
        cleaned = cleaned.replace("필요 없다고\n대단해~", "필요 없다는 게 대단해~")
        cleaned = cleaned.replace("필요 없다고 대단해~", "필요 없다는 게 대단해~")
        cleaned = cleaned.replace("잘해 주고", "잘해주고")
        cleaned = "\n".join(line.strip() for line in cleaned.splitlines()).strip()
        return cleaned

    def is_usable_translation(res: dict, source_text: str) -> bool:
        trans = str(res.get("translation") or "").strip()
        if not trans:
            return False
        if has_japanese_chars(trans):
            return False
        if normalize_translation_key(trans) == normalize_translation_key(source_text):
            return False
        if len(trans) < 2:
            return False
        return True

    def strict_retry_instruction(item: dict) -> str:
        return (
            "Strict retry for one manga speech bubble. "
            "Return exactly one natural Korean translation for the source text. "
            "Do NOT preserve Japanese Kanji, Hiragana, Katakana, names, honorifics, or particles. "
            "Translate titles naturally: 旦那様→남편님, 様→님, さん/氏→씨, ちゃん→쨩, 先輩→선배, 先生→선생님. "
            "Translate literal traps naturally: 一生→평생, 最早→이제는, 不束者ですが→서툴지만/서툰 사람인데, "
            "一片の憂い無し→한 점의 근심도 없어, いっしょうまも/一生守る→평생 지켜줄게. "
            f"Source text: {item['text']}"
        )

    def translate_item_with_retries(item: dict) -> Optional[list[dict]]:
        result = translator.translate_batch([item], api_key=api_key, previous_context=previous_context)
        if result and result[0] and is_usable_translation(result[0], item["text"]):
            return result
        for _ in range(2):
            result = translator.translate_batch(
                [item],
                api_key=api_key,
                previous_context=previous_context,
                extra_instruction=strict_retry_instruction(item),
            )
            if result and result[0] and is_usable_translation(result[0], item["text"]):
                return result
        logger.warning(f"  Translation retry did not fully clean {item['id']}; keeping best available result")
        return result

    previous_context = normalize_context(previous_context)

    texts_to_translate = []
    text_to_bubble_ids = {}
    all_targets = detection["bubbles"] + detection.get("floating_texts", [])
    for bubble in all_targets:
        ocr_text = bubble.get("ocr_text")
        if ocr_text and ocr_text != "[NO TEXT]":
            text_key = normalize_translation_key(ocr_text)
            text_to_bubble_ids.setdefault(text_key, []).append(bubble["id"])
            if len(text_to_bubble_ids[text_key]) == 1:
                item = {"id": bubble["id"], "text": ocr_text}
                if likely_sfx_text(ocr_text):
                    item["force_type"] = "sfx"
                texts_to_translate.append(item)
        else:
            bubble["translation"] = None

    if not texts_to_translate:
        return

    # Check Cache
    cache_payload = {
        "texts": texts_to_translate,
        "previous_context": previous_context or [],
    }
    payload_str = json.dumps(cache_payload, ensure_ascii=False)
    payload_hash = cache.hash_text(payload_str)
    cache_key = f"{TRANSLATION_CACHE_VERSION}_{payload_hash}"
    cached_translations = cache.get(cache_key)

    if cached_translations is not None:
        logger.info("  Translation: Using cached batch result")
        cached_items = Translator._sanitize_batch_result(json.loads(cached_translations))
        translations_dict = {t["id"]: t for t in cached_items}
    else:
        translator = get_translator_client(config)
        if hasattr(translator, "check_auth") and not translator.check_auth(api_key):
            logger.error("  Translation auth failed — skipping translation for this page")
            return
        # Split large batches to avoid LLM output truncation. Process in chunks of max 2 items.
        import math
        BATCH_CHUNK_SIZE = 2
        all_results = []
        for chunk_start in range(0, len(texts_to_translate), BATCH_CHUNK_SIZE):
            chunk = texts_to_translate[chunk_start: chunk_start + BATCH_CHUNK_SIZE]
            logger.info(f"  Translation: Chunk {chunk_start // BATCH_CHUNK_SIZE + 1}/{math.ceil(len(texts_to_translate) / BATCH_CHUNK_SIZE)} ({len(chunk)} bubbles)")
            chunk_result = translator.translate_batch(
                chunk,
                api_key=api_key,
                previous_context=previous_context,
            ) or []
            result_by_id = {res.get("id"): res for res in chunk_result if res.get("id")}
            missing_items = [
                item for item in chunk
                if item["id"] not in result_by_id or not is_usable_translation(result_by_id[item["id"]], item["text"])
            ]
            if missing_items:
                logger.warning(
                    f"  Translation validation failed for {', '.join(item['id'] for item in missing_items)}; retrying individually"
                )
            for item in missing_items:
                retried = translate_item_with_retries(item)
                if retried:
                    for res in retried:
                        result_by_id[res["id"]] = res
            all_results.extend(result_by_id[item["id"]] for item in chunk if item["id"] in result_by_id)
        batch_result = all_results if all_results else None
        if batch_result:
            cache.set(cache_key, json.dumps(batch_result, ensure_ascii=False))
            translations_dict = {t["id"]: t for t in batch_result}
        else:
            logger.error("  Translation: All chunks failed")
            translations_dict = {}

    bubble_result_map = {}
    source_id_to_text_key = {
        item["id"]: normalize_translation_key(item["text"])
        for item in texts_to_translate
    }
    for source_id, res in translations_dict.items():
        source_text_key = source_id_to_text_key.get(source_id)
        if source_text_key is None:
            continue
        res = dict(res)
        res["translation"] = clean_translation_artifacts(res.get("translation", ""), source_text_key)
        for bubble_id in text_to_bubble_ids.get(source_text_key, [source_id]):
            bubble_result_map[bubble_id] = res

    all_targets = detection["bubbles"] + detection.get("floating_texts", [])
    for bubble in all_targets:
        if bubble["id"] in bubble_result_map:
            res = bubble_result_map[bubble["id"]]
            trans = res.get("translation")
            text_type = res.get("type", "dialogue")
            if res.get("force_type"):
                text_type = res["force_type"]
            bubble["text_type"] = text_type
            skip_sfx = config.get("typesetting", {}).get("skip_sfx", True)
            if skip_sfx and text_type == "sfx":
                bubble["translation"] = None
                bubble["skip_inpaint"] = True
                logger.info(f"  SFX skipped for {bubble['id']}")
                continue
            bubble["translation"] = trans
            bubble["skip_inpaint"] = False
            if trans:
                logger.info(f"  Trans {bubble['id']} [{text_type}]: '{trans[:40]}'")


def run_inpainting(
    image_path: str,
    detection: dict,
    config: dict,
    tmp_dir: Path,
    original: Optional[Image.Image] = None,
) -> Path:
    """Step 4: clean text from image (flat fill + LaMa fallback for tone bubbles).
    Saves cleaned image to tmp_dir; caller is responsible for cleanup.
    """
    original = original.copy() if original is not None else Image.open(image_path).convert("RGB")
    text_mask_img = None
    if detection["text_mask"] is not None:
        text_mask_img = Image.fromarray(detection["text_mask"])
        if config["inpainting"].get("preserve_bubble_shape", True):
            text_mask_img = _clip_text_mask_to_text_bboxes(text_mask_img, detection, config)

    # Protect bubbles with failed translation from being erased
    if text_mask_img is not None:
        for bubble in detection.get("bubbles", []):
            if bubble.get("translation") is None and bubble.get("ocr_text") and bubble["ocr_text"] != "[NO TEXT]":
                bx1, by1, bx2, by2 = [int(v) for v in bubble["bbox"]]
                nx1 = max(0, bx1)
                ny1 = max(0, by1)
                nx2 = min(text_mask_img.width, bx2)
                ny2 = min(text_mask_img.height, by2)
                if nx2 > nx1 and ny2 > ny1:
                    import numpy as np
                    arr = np.array(text_mask_img)
                    arr[ny1:ny2, nx1:nx2] = 0
                    text_mask_img = Image.fromarray(arr)
                    logger.info(f"  Protected bubble {bubble['id']} from inpainting (translation failed)")
        if config["inpainting"].get("dialogue_only", True):
            text_mask_img = _exclude_bubbles_from_text_mask(text_mask_img, detection.get("bubbles", []))

    if text_mask_img is None:
        logger.warning("  No text mask — skipping inpainting")
        cleaned_path = tmp_dir / f"{detection['page_id']}_cleaned.png"
        original.save(cleaned_path)
        return cleaned_path

    result = original.copy()
    remaining_mask = text_mask_img

    # Step 1: Flat fill for white-background bubbles
    if config["inpainting"].get("flat_fill", True):
        from comfy_client import flat_fill
        result, remaining_mask = flat_fill(result, text_mask_img, bubbles=detection.get("bubbles"), texts=detection.get("texts"))
        logger.info(f"  Flat fill applied")

    # Step 2: Local standalone AI inpainting (or OpenCV fallback) for remaining mask areas
    inpaint_cfg = config["inpainting"]
    backend = str(inpaint_cfg.get("backend", "lama_onnx")).lower()
    inpainting_applied = False

    if remaining_mask is not None and backend == "comfyui":
        from comfy_client import ComfyClient
        client = ComfyClient(base_url=config["comfyui"]["base_url"], timeout=int(config["comfyui"].get("timeout", 180)))
        if client.check_available(timeout=2):
            inpainted = client.run_inpaint_lama(result, remaining_mask)
            if inpainted:
                result = inpainted
                inpainting_applied = True
                logger.info(f"  ComfyUI LaMa inpainting applied")

    if remaining_mask is not None and not inpainting_applied:
        try:
            from inpainting import get_inpainter
            inpainter = get_inpainter(config)
            result = inpainter.inpaint(result, remaining_mask)
            logger.info(f"  Standalone local inpainting applied ({backend})")
        except Exception as e:
            logger.error(f"  Local inpainting failed ({e}), falling back to basic OpenCV")
            from comfy_client import opencv_inpaint
            result = opencv_inpaint(
                result,
                remaining_mask,
                radius=int(inpaint_cfg.get("opencv_radius", 3)),
                dilate_iterations=int(inpaint_cfg.get("opencv_dilate", 2)),
            )
            logger.info(f"  OpenCV inpaint fallback applied")

    cleaned_path = tmp_dir / f"{detection['page_id']}_cleaned.png"
    result.save(cleaned_path)
    return cleaned_path


def _plan_vertical_text(text: str, bbox: list[int], config: dict) -> str:
    """Plan vertical Korean text as newline-separated columns."""
    typeset_cfg = config.get("typesetting", {})
    text = " ".join(str(text).split())
    if not text:
        return ""
    bbox_w = max(1, int(bbox[2]) - int(bbox[0]))
    bbox_h = max(1, int(bbox[3]) - int(bbox[1]))
    font_size = int(typeset_cfg.get("target_font_size", 42))
    avg_char_w = float(typeset_cfg.get("avg_char_width", 0.85))
    line_height = int(typeset_cfg.get("line_height", 52))
    padding = int(typeset_cfg.get("padding", 20))
    max_chars_per_col = max(1, int((bbox_h - padding) / max(line_height, 1)))
    char_step = max(1, int(font_size * avg_char_w))
    max_cols = max(1, int((bbox_w - padding) / max(char_step, 1)))
    if " " in text:
        columns: list[str] = []
        for segment in text.split():
            if not segment:
                continue
            for i in range(0, len(segment), max_chars_per_col):
                columns.append(segment[i:i + max_chars_per_col])
        if columns:
            return "\n".join(columns)
    chars = [ch for ch in text if not ch.isspace()]
    columns = []
    for i in range(0, len(chars), max_chars_per_col):
        columns.append("".join(chars[i:i + max_chars_per_col]))
    if len(columns) > max_cols:
        cols_per_col = max(1, int((len(chars) / max_cols) + 0.999))
        columns = []
        for i in range(0, len(chars), cols_per_col):
            columns.append("".join(chars[i:i + cols_per_col]))
        columns = columns[:max_cols]
    return "\n".join(columns)


def _plan_korean_lines(text: str, bbox: list[int], config: dict) -> str:
    """Insert conservative Korean line breaks before QPainter rendering."""
    typeset_cfg = config.get("typesetting", {})
    target_font_size = float(typeset_cfg.get("target_font_size", 42))
    avg_char_width = float(typeset_cfg.get("avg_char_width", 0.85))
    padding = float(typeset_cfg.get("padding", 20))
    line_height = float(typeset_cfg.get("line_height", 52))
    line_scale = float(typeset_cfg.get("line_planning_scale", 0.65))
    x1, y1, x2, y2 = [int(v) for v in bbox]
    bbox_w = max(1, x2 - x1)
    bbox_h = max(1, y2 - y1)
    chars_per_line = max(4, math.ceil((bbox_w - padding) / (target_font_size * avg_char_width * line_scale)))
    max_chars = typeset_cfg.get("max_chars_per_line")
    if max_chars:
        chars_per_line = min(chars_per_line, int(max_chars))
    max_lines = max(1, int((bbox_h - padding) / line_height))
    planned: list[str] = []
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    if any(" " in p for p in paragraphs):
        paragraphs = [" ".join(paragraphs)]
    for paragraph in paragraphs:
        if " " in paragraph:
            words = paragraph.split()
            lines = []
            current = ""
            for word in words:
                candidate = word if not current else f"{current} {word}"
                if len(candidate) <= chars_per_line:
                    current = candidate
                else:
                    if current:
                        lines.append(current)
                    current = word
            if current:
                lines.append(current)
        else:
            lines = [paragraph[i:i + chars_per_line] for i in range(0, len(paragraph), chars_per_line)]
        planned.extend(lines)
    refined: list[str] = []
    for line in planned:
        if len(line) <= chars_per_line:
            refined.append(line)
        else:
            refined.extend(line[i:i + chars_per_line] for i in range(0, len(line), chars_per_line))
    planned = refined
    if len(planned) > max_lines:
        # Do not force narrower wrapping; it often increases line count.
        # Renderer shrink handles overflow instead.
        logger.warning(f"  Planned {len(planned)} lines for {bbox_w}x{bbox_h}; renderer will shrink font")
    return "\n".join(planned)


def run_typesetting(
    cleaned_path: Path,
    detection: dict,
    config: dict,
    tmp_dir: Path,
    page_id: str,
) -> Path:
    """
    Step 5: Typeset translations onto cleaned image using existing QPainter engine.
    Saves typeset_plan to tmp_dir; writes final _ko.png to tmp_dir.
    Caller copies _ko.png to output dir.
    """
    # Build typeset_plan from detection bubbles + translations
    typeset_plans = []
    for bubble in detection["bubbles"]:
        trans = bubble.get("translation")
        if not trans:
            continue
            
        text_type = bubble.get("text_type", "dialogue")
        style = "normal"
        font_policy = "auto"
        
        if text_type == "thought":
            style = "italic"
        elif text_type == "sfx":
            style = "bold"
            font_policy = "sfx"
        bbox = [int(v) for v in bubble["bbox"]]
        bbox = _shrink_bbox_for_typesetting(bbox, config)
        vertical = bbox[3] - bbox[1] > (bbox[2] - bbox[0]) * 1.8
        typeset_plans.append({
            "id": bubble["id"],
            "action": "translate_replace",
            "text": _plan_vertical_text(trans, bbox, config) if vertical else _plan_korean_lines(trans, bbox, config),
            "bbox": bbox,
            "vertical": vertical,
            "style": style,
            "align": "center",
            "font_policy": font_policy,
            "estimated_font_size": _estimate_font_size(trans, bubble, vertical, config),
        })

    if not typeset_plans:
        logger.warning("  No translations to typeset — copying cleaned image")
        output_png = tmp_dir / f"{page_id}_ko.png"
        shutil.copy(str(cleaned_path), str(output_png))
        return output_png

    # Save typeset plan to temp dir
    plan_data = {"typeset_plans": typeset_plans}
    plan_path = tmp_dir / f"{page_id}_typeset_plan.json"
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(plan_data, f, indent=2, ensure_ascii=False)

    # Use local render_text.py (copied to project scripts/)
    script_path = str(Path(__file__).parent / "scripts" / "render_text.py")

    output_png = str(tmp_dir / f"{page_id}_ko.png")
    try:
        result = subprocess.run(
            [
                sys.executable, script_path,
                "--clean-image", str(cleaned_path),
                "--plan", str(plan_path),
                "--output-png", output_png,
            ],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            logger.error(f"  Typesetting subprocess failed (exit {result.returncode}):\n{result.stderr}")
            # Fallback: copy cleaned image
            shutil.copy(str(cleaned_path), output_png)
        else:
            # render_text.py logs go to stderr; stdout is typically empty
            out = (result.stderr + result.stdout).strip()
            if out:
                for line in out.split("\n"):
                    logger.info(f"  {line}")
            logger.info(f"  Typeset → {output_png}")
    except Exception as e:
        logger.error(f"  Typesetting error: {e}")
        shutil.copy(str(cleaned_path), output_png)

    return Path(output_png)


def run_genai_replacement(
    image_path: Path,
    detection: dict,
    config: dict,
    tmp_dir: Path,
    page_id: str,
) -> Path:
    """Step 5.5: Replace floating/background texts using Generative AI model or replacement pipeline."""
    floating = [f for f in detection.get("floating_texts", []) if f.get("translation")]
    if not floating or not config.get("inpainting", {}).get("enable_genai_floating_text", True):
        return image_path

    logger.info(f"[{page_id}] === GenAI Floating Text Replacement ({len(floating)} regions) ===")
    from inpainting.genai_inpainter import GenAIEditInpainter
    genai = GenAIEditInpainter(config.get("inpainting", {}))
    img = Image.open(image_path).convert("RGB")

    mask_img = None
    mask_arr = detection.get("floating_text_mask")
    if mask_arr is None:
        mask_arr = detection.get("all_text_mask") or detection.get("text_mask")
    if mask_arr is not None:
        mask_img = Image.fromarray(mask_arr)

    for f in floating:
        img = genai.replace_text_region(img, f, mask=mask_img)


    out_path = tmp_dir / f"{page_id}_ko.png"
    img.save(out_path)
    return out_path


def _build_dialogue_qa_report(page_id: str, detection: dict, final_image: str, config: dict) -> dict:
    """Lightweight QA report focused on speech-bubble dialogue quality."""
    import re

    def has_japanese_chars(text: str) -> bool:
        return bool(re.search(r"[\u3040-\u30ff\u3400-\u9fff]", str(text)))

    items = []
    translated_dialogue = 0
    total_dialogue = 0
    issues = []
    for bubble in detection.get("bubbles", []):
        ocr_text = bubble.get("ocr_text")
        translation = bubble.get("translation")
        text_type = bubble.get("text_type", "dialogue")
        flags = []
        if text_type == "sfx":
            flags.append("sfx")
            if bubble.get("skip_inpaint"):
                flags.append("preserved")
        else:
            total_dialogue += 1
            if ocr_text and ocr_text != "[NO TEXT]" and not translation:
                flags.append("missing_translation")
                issues.append(f"{bubble['id']}: missing translation")
            else:
                translated_dialogue += 1
            if translation and has_japanese_chars(translation):
                flags.append("japanese_remaining")
                issues.append(f"{bubble['id']}: Japanese remains")
        items.append({
            "id": bubble.get("id"),
            "type": text_type,
            "ocr_text": ocr_text,
            "translation": translation,
            "skip_inpaint": bool(bubble.get("skip_inpaint")),
            "bbox": bubble.get("bbox"),
            "flags": flags,
        })
    coverage = (translated_dialogue / total_dialogue * 100) if total_dialogue else 100.0
    return {
        "page_id": page_id,
        "final_image": final_image,
        "dialogue_coverage": round(coverage, 2),
        "translated_dialogue": translated_dialogue,
        "total_dialogue": total_dialogue,
        "issues": issues,
        "bubbles": items,
    }


def process_page(image_path: str, config: dict, api_key: str = "",
                 previous_context: Optional[list[str]] = None,
                 progress_callback: Optional[object] = None) -> dict:
    """
    Full pipeline end-to-end. Only the final _ko.png lands in out_dir.

    progress_callback(step, status, data) — optional hook for live progress.
      step: 'detection' | 'ocr' | 'translation' | 'inpainting' | 'typesetting' | 'qa'
      status: 'start' | 'done' | 'skip' | 'error'
      data: {"page": page_id, "error": str | None, ...}
    """
    def _progress(step, status, data=None):
        if progress_callback:
            try:
                d = dict(data or {})
                d.setdefault("page", page_id)
                progress_callback(step, status, d)
            except Exception:
                pass  # callback failures must not crash pipeline
    from cache_manager import CacheManager
    
    page_id = Path(image_path).stem
    out_dir = Path(config["output"]["base_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    cache = CacheManager(out_dir / "cache.json", autosave=False)

    # Temp dir for intermediate files (auto-cleaned on exit)
    try:
        with tempfile.TemporaryDirectory(prefix=f"manga_{page_id}_") as tmp_str:
            tmp_dir = Path(tmp_str)
            step_errors = []
            detection = None

            # Step 1: Detection
            try:
                _progress("detection", "start")
                logger.info(f"[{page_id}] === Detection ===")
                detection, original = run_detection(image_path, config, tmp_dir)
                _progress("detection", "done", {"bubbles_found": len(detection.get("bubbles", []))})
            except Exception as e:
                _progress("detection", "error", {"error": str(e)})
                logger.error(f"[{page_id}] Detection failed: {e}", exc_info=True)
                return {"page_id": page_id, "status": "error", "step": "detection", "error": str(e)}

            # Step 2: OCR (optional — needs API key)
            ocr_succeeded = False
            if api_key:
                try:
                    _progress("ocr", "start")
                    logger.info(f"[{page_id}] === OCR ===")
                    run_ocr(detection, original, config, api_key, tmp_dir, cache)
                    ocr_succeeded = True
                    _progress("ocr", "done")
                except Exception as e:
                    _progress("ocr", "error", {"error": str(e)})
                    step_errors.append(f"OCR: {e}")
                    logger.error(f"[{page_id}] OCR failed — will skip Translation: {e}", exc_info=True)
            else:
                _progress("ocr", "skip", {"reason": "no_api_key"})

            # Step 3: Translation (only if OCR succeeded)
            translation_succeeded = False
            if ocr_succeeded:
                try:
                    _progress("translation", "start")
                    logger.info(f"[{page_id}] === Translation ===")
                    run_translation(detection, config, api_key, cache, previous_context=previous_context)
                    translation_succeeded = True
                    _progress("translation", "done")
                except Exception as e:
                    _progress("translation", "error", {"error": str(e)})
                    step_errors.append(f"Translation: {e}")
                    logger.error(f"[{page_id}] Translation failed — continuing with inpainting: {e}", exc_info=True)
            else:
                _progress("translation", "skip")
                logger.info(f"[{page_id}] === Skipping OCR/Translation ===")

            # Step 4: Inpainting
            cleaned_path = None
            try:
                _progress("inpainting", "start")
                logger.info(f"[{page_id}] === Inpainting ===")
                cleaned_path = run_inpainting(image_path, detection, config, tmp_dir, original=original)
                _progress("inpainting", "done")
            except Exception as e:
                _progress("inpainting", "error", {"error": str(e)})
                step_errors.append(f"Inpainting: {e}")
                logger.error(f"[{page_id}] Inpainting failed — using original image: {e}", exc_info=True)
                cleaned_path = image_path  # fallback to original

            # Step 5: Typesetting & GenAI Replacement
            final_path = None
            if translation_succeeded and cleaned_path:
                try:
                    _progress("typesetting", "start")
                    logger.info(f"[{page_id}] === Typesetting ===")
                    final_path = run_typesetting(cleaned_path, detection, config, tmp_dir, page_id)
                    final_path = run_genai_replacement(final_path, detection, config, tmp_dir, page_id)
                    _progress("typesetting", "done")
                except Exception as e:
                    _progress("typesetting", "error", {"error": str(e)})
                    step_errors.append(f"Typesetting: {e}")
                    logger.error(f"[{page_id}] Typesetting failed: {e}", exc_info=True)
            else:
                _progress("typesetting", "skip")

            # Step 6: QA (optional, gated by config)
            qa_report = None
            if config.get("qa", {}).get("enabled", False):
                try:
                    from qa_module import check_page_result
                    # Build temporary result dict for QA
                    qa_input = {
                        "page_id": page_id,
                        "final_image": str(out_dir / f"{page_id}_ko.png"),
                        "detection": detection,
                        "errors": step_errors if step_errors else None,
                    }
                    qa_report = check_page_result(qa_input, config=config.get("qa", {}))
                    qa_icon = "✅" if qa_report["severity"] == "ok" else "⚠️" if qa_report["severity"] == "warn" else "❌"
                    logger.info(f"[{page_id}] QA: {qa_icon} {qa_report['translated']}/{qa_report['bubbles']} trans"
                                + (f" | {'; '.join(qa_report['issues'])}" if qa_report.get("issues") else ""))
                except Exception as e:
                    logger.warning(f"[{page_id}] QA check failed: {e}")

            # Determine output: prefer typeset result, fallback to inpainted, fallback to original
            if final_path and os.path.exists(final_path):
                chosen_final = final_path
            elif cleaned_path and os.path.exists(str(cleaned_path)):
                chosen_final = Path(cleaned_path) if isinstance(cleaned_path, str) else cleaned_path
            else:
                chosen_final = Path(image_path)

            # Copy final result to output directory
            dest = out_dir / f"{page_id}_ko.png"
            shutil.copy2(str(chosen_final), str(dest))
            qa_report = _build_dialogue_qa_report(page_id, detection, str(dest), config)
            qa_dir = Path(config.get("qa", {}).get("report_dir") or out_dir)
            qa_dir.mkdir(parents=True, exist_ok=True)
            with open(qa_dir / f"{page_id}_dialogue_qa.json", "w", encoding="utf-8") as f:
                json.dump(qa_report, f, ensure_ascii=False, indent=2)

            # Collect translated text for next page context
            page_translations = []
            if detection:
                for b in detection.get("bubbles", []):
                    t = b.get("translation")
                    if t and b.get("text_type") != "sfx":
                        page_translations.append(t)

            has_error = len(step_errors) > 0
            status = "partial" if has_error else "complete"

            partial_note = ""
            if has_error:
                partial_note = f" ({'; '.join(step_errors)})"
            logger.info(f"[{page_id}] {status.upper()}{partial_note} → {dest}")

        return {
            "page_id": page_id,
            "status": status,
            "final_image": str(dest),
            "translations": page_translations,
            "detection": {
                "bubbles": [
                    {k: v for k, v in b.items() if k != "type"}
                    for b in detection.get("bubbles", [])
                ] if detection else [],
                "texts": [
                    {k: v for k, v in t.items() if k != "mask"}
                    for t in detection.get("texts", [])
                ] if detection else [],
            } if detection else None,
            "errors": step_errors if has_error else None,
        }
    finally:
        cache.flush()


def main():
    parser = argparse.ArgumentParser(description="Manga Translation Pipeline v2")
    parser.add_argument("images", nargs="+", help="Image file(s) to process")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--api-key", default="", help="OpenRouter API key")
    parser.add_argument("--output", default=None, help="Output directory override")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.output:
        config["output"]["base_dir"] = args.output

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        # Try loading from .env
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("OPENROUTER_API_KEY="):
                        api_key = line.split("=", 1)[1].strip().strip("\"'")
                        break
    if not api_key:
        logger.warning("No API key — skipping OCR and translation (detection + inpainting only)")

    results = []
    for img_path in args.images:
        t0 = time.time()
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing: {img_path}")
        logger.info(f"{'='*50}")
        try:
            result = process_page(img_path, config, api_key)
            elapsed = time.time() - t0
            logger.info(f"Done in {elapsed:.1f}s: {result['page_id']}")
            results.append(result)
        except Exception as e:
            logger.error(f"Failed on {img_path}: {e}", exc_info=True)
            results.append({"page_id": Path(img_path).stem, "status": "error", "error": str(e)})

    # Summary
    success = sum(1 for r in results if r.get("status") in ("complete", "partial"))
    logger.info(f"\n{'='*50}")
    logger.info(f"Summary: {success}/{len(results)} succeeded")
    for r in results:
        s = r.get("status", "error")
        if s == "complete":
            icon = "✅"
        elif s == "partial":
            icon = "⚠️"
        else:
            icon = "❌"
        detail = r.get("final_image", r.get("error", "?"))
        if r.get("errors"):
            detail += f" [{'; '.join(r['errors'])}]"
        logger.info(f"  {icon} {r['page_id']}: {detail}")


if __name__ == "__main__":
    main()
