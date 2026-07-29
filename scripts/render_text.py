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
        "min_font_size": 14,
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
    "floating": {
        "base_font_size": 38,
        "min_font_size": 22,
        "max_font_size": 54,
        "color": "#000000",
        "stroke_color": "#ffffff",
        "stroke_width": 10,
        "font_weight": "Bold",
        "padding": 0,
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
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QFont, QFontMetrics, QTextDocument, QTextOption
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
        if "\n" in text:
            metrics = QFontMetrics(font)
            lines = text.splitlines() or [text]
            measured_w = max(metrics.horizontalAdvance(line) for line in lines) + stroke_width * 2
            measured_h = metrics.lineSpacing() * len(lines) + stroke_width * 2
        else:
            option = QTextOption()
            option.setAlignment(Qt.AlignmentFlag.AlignCenter)
            option.setWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
            document = QTextDocument()
            document.setDocumentMargin(0)
            document.setDefaultFont(font)
            document.setDefaultTextOption(option)
            document.setPlainText(text)
            document.setTextWidth(usable_w)
            measured = document.size()
            measured_w = measured.width()
            measured_h = measured.height()

        if measured_h <= usable_h and measured_w <= usable_w:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    # Clamp
    result = max(min_fs, min(max_fs, best))
    result = max(result, min_fs)

    return result


def calculate_vertical_font_size(
    bbox_w: int,
    bbox_h: int,
    text: str,
    preset: dict,
    preferred_font_size: Optional[int] = None,
) -> int:
    """Fit multi-column vertical text using the same geometry as the renderer."""
    from PyQt5.QtGui import QFont, QFontMetrics
    from PyQt5.QtWidgets import QApplication

    QApplication.instance() or QApplication(["hermes_typeset"])
    min_fs = preset.get("min_font_size", 28)
    max_fs = preset.get("max_font_size", 72)
    if preferred_font_size is not None:
        max_fs = max(min_fs, min(max_fs, int(preferred_font_size)))
    margin = preset.get("padding", 10) + preset.get("stroke_width", 5)
    usable_w = bbox_w - margin * 2
    usable_h = bbox_h - margin * 2
    columns = [column for column in text.split("\n") if column.strip()] or [text]
    max_col_len = max(len(column) for column in columns)

    best = min_fs
    for size in range(min_fs, max_fs + 1):
        font = QFont("Apple SD Gothic Neo", size)
        font.setWeight(QFont.Bold if preset.get("font_weight") == "Bold" else QFont.Normal)
        metrics = QFontMetrics(font)
        char_step = max(1, int(metrics.height() * 0.9))
        col_gap = max(1, int(metrics.horizontalAdvance(" ") * 1.2))
        measured_w = len(columns) * char_step + max(0, len(columns) - 1) * col_gap
        measured_h = max_col_len * char_step
        if measured_w <= usable_w and measured_h <= usable_h:
            best = size
        else:
            break
    return best


def render_text_direct(clean_image_path, typeset_plans, output_png_path):
    """
    Render typeset plans onto cleaned image using QPainter.drawText with
    automatic word-wrap, then tight-crop to actual text bounds.
    """
    from PyQt5.QtGui import QImage, QPainter, QFont, QColor, QPen, QTextOption
    from PyQt5.QtCore import Qt, QRectF
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

        # Detected speech-bubble boxes are usually oval or irregular. Keep text in
        # a narrower centered rectangle so strokes do not touch the curved sides.
        safe_width_ratio = max(0.5, min(1.0, float(plan.get("safe_width_ratio", 0.82))))
        safe_bbox_w = max(10, int(bbox_w * safe_width_ratio))

        # Font size from iterative binary-search calculator
        if plan.get("vertical", False):
            font_size = calculate_vertical_font_size(
                safe_bbox_w, bbox_h, text, preset, plan.get("estimated_font_size")
            )
        else:
            font_size = calculate_font_size(
                safe_bbox_w, bbox_h, text, preset, plan.get("estimated_font_size")
            )

        padding = preset.get("padding", 10)
        stroke_width = preset.get("stroke_width", 5)

        font = QFont(font_family, font_size)
        font.setWeight(QFont.Bold if preset.get("font_weight") == "Bold" else QFont.Normal)
        font.setStyleHint(QFont.SansSerif)

        margin = stroke_width + padding
        safe_x1 = x1 + (bbox_w - safe_bbox_w) / 2
        text_rect = QRectF(
            safe_x1 + margin,
            y1 + margin,
            max(1, safe_bbox_w - margin * 2),
            max(1, bbox_h - margin * 2),
        )

        # Stroke
        stroke_color = QColor(preset.get("stroke_color", "#ffffff"))
        stroke_pen = QPen(stroke_color, max(stroke_width, 0.5))
        stroke_pen.setJoinStyle(Qt.RoundJoin)

        # Fill
        fill_color = QColor(preset.get("color", "#000000"))

        if plan.get("vertical", False):
            _render_vertical_text(main_painter, text_rect, text, font, stroke_pen, fill_color)
        else:
            option = QTextOption()
            option.setAlignment(Qt.AlignmentFlag.AlignCenter)
            option.setWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)

            # Stroke pass
            main_painter.setFont(font)
            main_painter.setPen(stroke_pen)
            main_painter.drawText(text_rect, text, option)

            # Fill pass
            main_painter.setPen(fill_color)
            main_painter.drawText(text_rect, text, option)

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
