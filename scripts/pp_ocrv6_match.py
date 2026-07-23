#!/usr/bin/env python3
"""Minimal PP-OCRv6 row to YOLO bubble matcher used by tests/experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_vlm_json(path: Path) -> dict[str, str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return {str(item["id"]): str(item.get("text", "")) for item in data}
    return {str(k): str(v) for k, v in data.items()}


def _bbox(poly: Any) -> tuple[int, int, int, int]:
    pts = np.asarray(poly, dtype=float).reshape((-1, 2))
    x1, y1 = np.floor(pts.min(axis=0)).astype(int)
    x2, y2 = np.ceil(pts.max(axis=0)).astype(int)
    return int(x1), int(y1), int(x2), int(y2)


def _area(box: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def _intersection(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    return max(0, x2 - x1) * max(0, y2 - y1)


def _center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def _contains(bubble: tuple[int, int, int, int], point: tuple[float, float], margin: int) -> bool:
    x1, y1, x2, y2 = bubble
    px, py = point
    return x1 - margin <= px <= x2 + margin and y1 - margin <= py <= y2 + margin


def _match_score(
    bubble: tuple[int, int, int, int],
    text_box: tuple[int, int, int, int],
    image_w: int,
    image_h: int,
    margin: int,
) -> float:
    bx1, by1, bx2, by2 = bubble
    tx1, ty1, tx2, ty2 = text_box
    bubble_area = _area(bubble)
    text_area = _area(text_box)
    if bubble_area <= 0 or text_area <= 0:
        return -1.0

    overlap = _intersection(bubble, text_box)
    overlap_score = overlap / min(bubble_area, text_area)
    center = _center(text_box)
    inside = 1.0 if _contains(bubble, center, margin) else 0.0
    bcx, bcy = _center(bubble)
    diag = (image_w**2 + image_h**2) ** 0.5
    dist = ((center[0] - bcx) ** 2 + (center[1] - bcy) ** 2) ** 0.5 / max(diag, 1)
    return overlap_score * 2.0 + inside - dist * 2.0


def match_pp_ocr_to_bubbles(
    detection: dict[str, Any],
    pp_rows: list[dict[str, Any]],
    score_threshold: float = 0.5,
    match_score_threshold: float = 2.0,
) -> dict[str, Any]:
    image_w = int(detection.get("width") or 0)
    image_h = int(detection.get("height") or 0)
    vlm_by_id = {str(b["id"]): str(b.get("ocr_text") or b.get("vlm_text") or "") for b in detection.get("bubbles", [])}
    matches: dict[str, dict[str, Any]] = {
        str(b["id"]): {"pp": "", "vlm": vlm_by_id.get(str(b["id"]), ""), "score": -1.0}
        for b in detection.get("bubbles", [])
    }
    filtered: list[dict[str, Any]] = []

    for row in pp_rows:
        score = float(row.get("score") or 0.0)
        if score < score_threshold:
            continue
        text_box = _bbox(row.get("poly", []))
        best_id: str | None = None
        best_score = -1.0
        for bubble in detection.get("bubbles", []):
            bubble_id = str(bubble["id"])
            bubble_bbox = [int(v) for v in bubble["bbox"]]
            bubble_box: tuple[int, int, int, int] = (
                bubble_bbox[0],
                bubble_bbox[1],
                bubble_bbox[2],
                bubble_bbox[3],
            )
            candidate = _match_score(bubble_box, text_box, image_w, image_h, margin=5)
            if candidate > best_score:
                best_score = candidate
                best_id = bubble_id
        if best_id is None:
            continue
        filtered_row = dict(row)
        filtered_row.update({"bubble_id": best_id, "match_score": best_score})
        if best_score >= match_score_threshold:
            filtered.append(filtered_row)
            matches[best_id]["pp"] = str(row.get("text", ""))
            matches[best_id]["score"] = best_score

    return {
        "pp_rows_total": len(pp_rows),
        "pp_rows_filtered": filtered,
        "matches": matches,
    }
