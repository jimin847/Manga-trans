#!/usr/bin/env python3
"""
Manga localization typesetting engine — PyQt5 QPainter direct rendering.
Subprocess entry point called by main.py run_typesetting().

Usage:
    python3 call_krita.py --clean-image <path> --plan <json_path> --output-png <path>
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("typeset")

# ── style presets (v12: base=48, Qt-measured 0.85 avg_char_w) ─────────
STYLE_PRESETS = {
    "normal": {
        "base_font_size": 48,
        "min_font_size": 22,
        "max_font_size": 72,
        "color": "#000000",
        "stroke_color": "#ffffff",
        "stroke_width": 5,
        "font_weight": "Bold",
        "padding": 10,
    },
    "shout": {
        "base_font_size": 56,
        "min_font_size": 34,
        "max_font_size": 84,
        "color": "#000000",
        "stroke_color": "#ffffff",
        "stroke_width": 5,
        "font_weight": "Bold",
        "padding": 10,
    },
    "whisper": {
        "base_font_size": 36,
        "min_font_size": 22,
        "max_font_size": 48,
        "color": "#555555",
        "stroke_color": "#ffffff",
        "stroke_width": 4,
        "font_weight": "Normal",
        "padding": 8,
    },
    "thought": {
        "base_font_size": 40,
        "min_font_size": 26,
        "max_font_size": 60,
        "color": "#000000",
        "stroke_color": "#ffffff",
        "stroke_width": 4,
        "font_weight": "Normal",
        "padding": 10,
    },
    "narration": {
        "base_font_size": 40,
        "min_font_size": 26,
        "max_font_size": 60,
        "color": "#333333",
        "stroke_color": "#ffffff",
        "stroke_width": 4,
        "font_weight": "Bold",
        "padding": 10,
    },
    "small_note": {
        "base_font_size": 30,
        "min_font_size": 18,
        "max_font_size": 42,
        "color": "#000000",
        "stroke_color": "#ffffff",
        "stroke_width": 3,
        "font_weight": "Normal",
        "padding": 6,
    },
}


def scale_vlm_bboxes(plans, img_w, img_h):
    """Scale VLM bbox coordinates from 0-1000 range to image pixels. X-only."""
    if not plans:
        return
    max_vlm_x = max((p.get("bbox") or [0])[2] for p in plans if p.get("bbox"))
    if max_vlm_x == 0:
        return
    scale_x = img_w / max_vlm_x
    for p in plans:
        b = p["bbox"]
        b[0] = max(0, min(img_w, int(b[0] * scale_x)))
        b[2] = max(0, min(img_w, int(b[2] * scale_x)))
        # Y stays as-is


def calculate_font_size(bbox_w: int, bbox_h: int, text: str, preset: dict, preferred_font_size: Optional[int] = None) -> int:
    """
    Iterative font-size calculator that finds the largest size where
    word-wrapped text fits within bbox dimensions.
    """
    from PyQt5.QtGui import QFont, QFontMetrics, QTextOption
    from PyQt5.QtCore import Qt, QRectF
    from PyQt5.QtWidgets import QApplication

    base = preset.get("base_font_size", 48)
    min_fs = preset.get("min_font_size", 28)
    max_fs = preset.get("max_font_size", 72)
    preferred = preferred_font_size if preferred_font_size else None
    if preferred is not None:
        max_fs = max(min_fs, min(max_fs, int(preferred)))
    padding = preset.get("padding", 10)
    stroke_width = preset.get("stroke_width", 5)

    if not text or len(text) == 0:
        return base

    # Ensure QApplication exists
    app = QApplication.instance() or QApplication(["hermes_typeset"])

    text_img_w = bbox_w - padding * 2
    text_img_h = bbox_h - padding * 2
    margin = stroke_width + padding
    usable_w = text_img_w - margin * 2
    usable_h = text_img_h - margin * 2

    if usable_w <= 0 or usable_h <= 0:
        return min_fs

    # Binary search for max font size that fits
    lo, hi = min_fs, max_fs
    best = min_fs

    font_family = "Apple SD Gothic Neo"

    while lo <= hi:
        mid = (lo + hi) // 2
        font = QFont(font_family, mid)
        font.setWeight(QFont.Bold if preset.get("font_weight") == "Bold" else QFont.Normal)
        fm = QFontMetrics(font)

        # Simulate word-wrap: count lines needed
        line_h = fm.lineSpacing()
        words = text.replace("\n", " \n ").split()

        total_lines = 0
        current_w = 0.0
        for word in words:
            if word == "\n":
                total_lines += 1
                current_w = 0.0
                continue
            word_w = fm.horizontalAdvance(word)
            if total_lines == 0:
                total_lines = 1
            if current_w and current_w + fm.horizontalAdvance(" ") + word_w > usable_w:
                total_lines += 1
                current_w = word_w
            else:
                current_w += word_w + (fm.horizontalAdvance(" ") if current_w else 0)
        if current_w == 0 and total_lines == 0:
            total_lines = 1

        needed_h = total_lines * line_h

        # Check if longest individual line fits
        longest_line = ""
        for line in text.split("\n"):
            if len(line) > len(longest_line):
                longest_line = line
        longest_w = fm.horizontalAdvance(longest_line)

        if needed_h <= usable_h and longest_w <= usable_w:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    # Clamp
    result = max(min_fs, min(max_fs, best))
    result = max(result, min_fs)

    return result


def render_text_direct(clean_image_path, typeset_plans, output_png_path):
    """
    Render typeset plans onto cleaned image using QPainter.drawText with
    automatic word-wrap, then tight-crop to actual text bounds.
    """
    from PyQt5.QtGui import QImage, QPainter, QFont, QColor, QPen, QTextOption
    from PyQt5.QtCore import Qt, QPointF, QRectF
    from PyQt5.QtWidgets import QApplication
    from PIL import Image as PILImage

    app = QApplication.instance() or QApplication([])

    pil_img = PILImage.open(clean_image_path).convert("RGBA")
    img_w, img_h = pil_img.size

    img_bytes = pil_img.tobytes()
    main_qimage = QImage(img_bytes, img_w, img_h, QImage.Format_RGBA8888)
    main_painter = QPainter(main_qimage)

    font_family = "Apple SD Gothic Neo"

    # Render lower bubbles first so overlapping detected bboxes don't cover text above them.
    typeset_plans = sorted(
        typeset_plans,
        key=lambda p: (int((p.get("bbox") or [0])[1]), int((p.get("bbox") or [0])[0])),
        reverse=True,
    )

    for plan in typeset_plans:
        bbox = plan.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        if bbox_w < 10 or bbox_h < 10:
            continue

        text = plan.get("text", "").strip()
        if not text:
            continue

        style_name = plan.get("style", "normal")
        preset = STYLE_PRESETS.get(style_name, STYLE_PRESETS["normal"])

        # Font size from iterative binary-search calculator
        font_size = calculate_font_size(bbox_w, bbox_h, text, preset, plan.get("estimated_font_size"))

        padding = preset.get("padding", 10)
        stroke_width = preset.get("stroke_width", 5)

        font = QFont(font_family, font_size)
        font.setWeight(QFont.Bold if preset.get("font_weight") == "Bold" else QFont.Normal)
        font.setStyleHint(QFont.SansSerif)

        # Temp image: intentionally taller than bbox so QPainter never clips long text.
        # After rendering we tight-crop and scale the actual glyph bounds down if needed.
        margin = stroke_width + padding
        text_img_w = bbox_w
        text_img_h = max(bbox_h, int(font_size * 20) + margin * 4)

        text_img = QImage(text_img_w, text_img_h, QImage.Format_ARGB32)
        text_img.fill(Qt.transparent)

        tp = QPainter(text_img)
        tp.setRenderHint(QPainter.Antialiasing)
        tp.setRenderHint(QPainter.TextAntialiasing)
        tp.setFont(font)

        # The text rect should be WIDE enough that text doesn't wrap unnecessarily,
        # but we'll measure actual bounds after rendering
        text_rect = QRectF(margin, margin,
                          text_img_w - margin * 2,
                          text_img_h - margin * 2)

        # Stroke
        stroke_color = QColor(preset.get("stroke_color", "#ffffff"))
        stroke_pen = QPen(stroke_color, max(stroke_width, 0.5))
        stroke_pen.setJoinStyle(Qt.RoundJoin)

        # Fill
        fill_color = QColor(preset.get("color", "#000000"))

        if plan.get("vertical", False):
            _render_vertical_text(tp, text_rect, text, font, stroke_pen, fill_color)
        else:
            # Configure drawText for auto word-wrap + center alignment
            option = QTextOption()
            option.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            option.setWrapMode(QTextOption.WrapMode.NoWrap)

            # Stroke pass
            tp.setPen(stroke_pen)
            tp.drawText(text_rect, text, option)

            # Fill pass
            tp.setPen(fill_color)
            tp.drawText(text_rect, text, option)

        tp.end()

        # Tight crop: scan for non-transparent pixels
        cropped = _crop_transparent(text_img, stroke_width)
        if cropped is None:
            logger.warning(f"  Typeset {plan.get('id', '?')}: empty text, skipping")
            continue

        crop_w = cropped.width()
        crop_h = cropped.height()

        # Downscale if exceeds bbox dimensions
        if crop_w > bbox_w or crop_h > bbox_h:
            scale = min(bbox_w / crop_w, bbox_h / crop_h)
            new_w = max(1, int(crop_w * scale))
            new_h = max(1, int(crop_h * scale))
            cropped = cropped.scaled(new_w, new_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            crop_w, crop_h = new_w, new_h

        # Center the cropped text in the bbox on main image
        dest_x = x1 + (bbox_w - crop_w) // 2
        dest_y = y1 + (bbox_h - crop_h) // 2
        main_painter.drawImage(QPointF(dest_x, dest_y), cropped)

        logger.info(f"  Typeset {plan.get('id', '?')} ({style_name}, {font_size}pt, {bbox_w}x{bbox_h}): '{text[:40]}'")

    main_painter.end()
    main_qimage.save(output_png_path)
    logger.info(f"Typesetting complete -> {output_png_path}")
    return True


def _render_vertical_text(painter, rect, text, font, stroke_pen, fill_color):
    from PyQt5.QtCore import Qt
    painter.setFont(font)
    columns = [col for col in text.split("\n") if col.strip()] or [text]
    fm = painter.fontMetrics()
    char_step = max(1, int(fm.height() * 0.9))
    col_gap = max(1, int(fm.horizontalAdvance(" ") * 1.2))
    max_col_len = max(len(col) for col in columns)
    total_w = len(columns) * char_step + max(0, len(columns) - 1) * col_gap
    start_x = int(rect.right()) - max(0, int((rect.width() - total_w) / 2))
    start_y = int(rect.top()) + max(0, int((rect.height() - (max_col_len - 1) * fm.height() * 0.9) / 2))
    for col_i, col in enumerate(columns):
        x = start_x - col_i * (char_step + col_gap)
        for j, ch in enumerate(col):
            y = start_y + int(j * fm.height() * 0.9)
            char_rect = rect.__class__(x - char_step // 2, y - fm.height() // 2, char_step, fm.height())
            painter.setPen(stroke_pen)
            painter.drawText(char_rect, Qt.AlignmentFlag.AlignCenter, ch)
            painter.setPen(fill_color)
            painter.drawText(char_rect, Qt.AlignmentFlag.AlignCenter, ch)


def _crop_transparent(image, margin=0):
    """Crop ARGB32 QImage to non-transparent pixel bounds. Returns cropped QImage or None."""
    from PyQt5.QtCore import QRect

    w, h = image.width(), image.height()
    if w == 0 or h == 0:
        return None

    # Quick scan: find min/max non-transparent pixels using scanLine
    top, bottom = h, 0
    for y in range(h):
        ptr = image.scanLine(y)
        # ptr is a sip.voidptr; each pixel = 4 bytes (ARGB), alpha at offset 3
        ptr.setsize(w * 4)
        alpha_data = bytes(ptr)[3::4]  # every 4th byte starting at index 3
        if any(b != 0 for b in alpha_data):
            if y < top:
                top = y
            bottom = y

    if top > bottom:
        return None

    # Horizontal scan within [top, bottom]
    left, right = w, 0
    for y in range(top, bottom + 1):
        ptr = image.scanLine(y)
        ptr.setsize(w * 4)
        rgba = bytes(ptr)
        for x in range(w):
            alpha = rgba[x * 4 + 3]
            if alpha != 0:
                if x < left:
                    left = x
                if x > right:
                    right = x

    if left > right:
        return None

    # Add margin
    extra = max(margin, 2)
    left = max(0, left - extra)
    top = max(0, top - extra)
    right = min(w - 1, right + extra)
    bottom = min(h - 1, bottom + extra)

    crop_w = right - left + 1
    crop_h = bottom - top + 1
    if crop_w <= 0 or crop_h <= 0:
        return None

    return image.copy(left, top, crop_w, crop_h)


def main():
    parser = argparse.ArgumentParser(description="Manga typesetting renderer (PyQt5)")
    parser.add_argument("--clean-image", required=True, help="Path to cleaned image (PNG)")
    parser.add_argument("--plan", required=True, help="Path to typeset_plan.json")
    parser.add_argument("--output-png", required=True, help="Output PNG path")
    args = parser.parse_args()

    if not os.path.exists(args.clean_image):
        logger.error(f"Clean image not found: {args.clean_image}")
        sys.exit(1)
    if not os.path.exists(args.plan):
        logger.error(f"Typeset plan not found: {args.plan}")
        sys.exit(1)

    with open(args.plan, encoding="utf-8") as f:
        plan_data = json.load(f)
    typeset_plans = plan_data.get("typeset_plans", [])

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    success = render_text_direct(args.clean_image, typeset_plans, args.output_png)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
