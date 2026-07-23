from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def load(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def is_dark(rgb: np.ndarray, threshold: int = 90) -> np.ndarray:
    return np.mean(rgb, axis=2) < threshold


def diff_dark_pixels(original: Image.Image, output: Image.Image, threshold: int = 90) -> tuple[int, int]:
    orig = np.asarray(original)
    out = np.asarray(output)
    orig_dark = is_dark(orig, threshold)
    out_dark = is_dark(out, threshold)
    removed = int(np.sum(orig_dark & ~out_dark))
    added = int(np.sum(~orig_dark & out_dark))
    return removed, added


def mask_from_bboxes(size: tuple[int, int], bboxes: list[list[int]], padding: int = 0) -> np.ndarray:
    mask = np.zeros(size[::-1], dtype=bool)
    h, w = size
    for bbox in bboxes:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = True
    return mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple manga localization preservation QA")
    parser.add_argument("--original", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bbox-json")
    parser.add_argument("--bubble-padding", type=int, default=0)
    parser.add_argument("--text-padding", type=int, default=6)
    parser.add_argument("--threshold", type=int, default=90)
    args = parser.parse_args()

    original = load(args.original)
    output = load(args.output)
    if original.size != output.size:
        raise SystemExit(f"size mismatch: {original.size} vs {output.size}")

    removed, added = diff_dark_pixels(original, output, args.threshold)

    bubble_leak = None
    text_bbox_coverage = None
    if args.bbox_json:
        data = json.loads(Path(args.bbox_json).read_text())
        bubble_mask = mask_from_bboxes(original.size, [b["bbox"] for b in data.get("bubbles", [])], args.bubble_padding)
        text_mask = mask_from_bboxes(original.size, [t["bbox"] for t in data.get("texts", [])], args.text_padding)
        orig_dark = is_dark(np.asarray(original), args.threshold)
        out_dark = is_dark(np.asarray(output), args.threshold)
        new_dark = ~orig_dark & out_dark
        bubble_leak = int(np.sum(new_dark & ~bubble_mask)) if np.any(~bubble_mask) else 0
        text_bbox_coverage = float(np.sum(out_dark & text_mask) / max(1, np.sum(text_mask)))

    report = {
        "removed_dark_pixels": removed,
        "added_dark_pixels": added,
        "bubble_outside_added_dark_pixels": bubble_leak,
        "text_bbox_dark_coverage": text_bbox_coverage,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
