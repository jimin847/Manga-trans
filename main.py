#!/usr/bin/env python3
"""
manga_pipeline_v2 — main entry point
YOLO detection + VLM OCR + ComfyUI inpainting + QPainter typesetting
"""
import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

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
OCR_CACHE_VERSION = "ocr_v2"
TRANSLATION_CACHE_VERSION = "trans_v2"


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    validate_config(config, Path(config_path).parent)
    return config


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


def get_detector(config: dict):
    from detection.yolo_detector import YoloDetector

    cache_key = (
        config["models"]["text_segmenter"],
        config["models"]["bubble_detector"],
        config["yolo"]["conf_threshold"],
        config["yolo"]["iou_threshold"],
        config["yolo"]["device"],
    )
    detector = _DETECTOR_CACHE.get(cache_key)
    if detector is None:
        detector = YoloDetector(
            text_segmenter_path=config["models"]["text_segmenter"],
            bubble_detector_path=config["models"]["bubble_detector"],
            conf_threshold=config["yolo"]["conf_threshold"],
            iou_threshold=config["yolo"]["iou_threshold"],
            device=config["yolo"]["device"],
        )
        _DETECTOR_CACHE[cache_key] = detector
    return detector


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
    )
    client = _TRANSLATOR_CACHE.get(cache_key)
    if client is None:
        client = Translator(
            provider=config["translation"]["provider"],
            model=config["translation"]["model"],
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


def run_ocr(detection: dict, original: Image.Image, config: dict, api_key: str, tmp_dir: Path, cache):
    """Step 2: VLM OCR on each bubble crop (Parallel + Caching + Orientation)"""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from io import BytesIO

    ocr = get_ocr_client(config)

    def prepare_bubble(bubble):
        x1, y1, x2, y2 = [int(v) for v in bubble["bbox"]]
        crop = original.crop((x1, y1, x2, y2))

        bw, bh = x2 - x1, y2 - y1
        orientation = "mixed"
        if bh > bw * 1.5:
            orientation = "vertical"
        elif bw > bh * 1.5:
            orientation = "horizontal"
        bubble["orientation"] = orientation

        upscale = config["ocr"].get("crop_upscale", 2)
        if upscale > 1:
            crop = crop.resize((crop.width * upscale, crop.height * upscale), Image.LANCZOS)

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

    bubble_prepared = [prepare_bubble(b) for b in detection["bubbles"]]
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

    previous_context = normalize_context(previous_context)

    texts_to_translate = []
    text_to_bubble_ids = {}
    for bubble in detection["bubbles"]:
        ocr_text = bubble.get("ocr_text")
        if ocr_text and ocr_text != "[NO TEXT]":
            text_key = normalize_translation_key(ocr_text)
            text_to_bubble_ids.setdefault(text_key, []).append(bubble["id"])
            if len(text_to_bubble_ids[text_key]) == 1:
                texts_to_translate.append({"id": bubble["id"], "text": ocr_text})
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
        # Split large batches to avoid LLM output truncation (Nemotron often returns
        # partial results for 5+ items). Process in chunks of max 3 items.
        import math
        BATCH_CHUNK_SIZE = 3
        all_results = []
        for chunk_start in range(0, len(texts_to_translate), BATCH_CHUNK_SIZE):
            chunk = texts_to_translate[chunk_start: chunk_start + BATCH_CHUNK_SIZE]
            logger.info(f"  Translation: Chunk {chunk_start // BATCH_CHUNK_SIZE + 1}/{math.ceil(len(texts_to_translate) / BATCH_CHUNK_SIZE)} ({len(chunk)} bubbles)")
            chunk_result = translator.translate_batch(
                chunk,
                api_key=api_key,
                previous_context=previous_context,
            )
            if chunk_result:
                all_results.extend(chunk_result)
            else:
                logger.error(f"  Translation chunk failed for items {chunk[0]['id']}..{chunk[-1]['id']}")
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
        for bubble_id in text_to_bubble_ids.get(source_text_key, [source_id]):
            bubble_result_map[bubble_id] = res

    for bubble in detection["bubbles"]:
        if bubble["id"] in bubble_result_map:
            res = bubble_result_map[bubble["id"]]
            trans = res.get("translation")
            text_type = res.get("type", "dialogue")
            bubble["translation"] = trans
            bubble["text_type"] = text_type
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
        result, remaining_mask = flat_fill(result, text_mask_img, bubbles=detection.get("bubbles"))
        logger.info(f"  Flat fill applied")

    # Step 2: LaMa for remaining (tone/screentone) mask areas
    if remaining_mask is not None and config["inpainting"].get("lama_fallback", True):
        from comfy_client import ComfyClient
        client = ComfyClient(base_url=config["comfyui"]["base_url"])
        inpainted = client.run_inpaint_lama(result, remaining_mask)
        if inpainted:
            result = inpainted
            logger.info(f"  LaMa inpainting applied")
        else:
            logger.warning("  LaMa failed, keeping flat fill result")

    cleaned_path = tmp_dir / f"{detection['page_id']}_cleaned.png"
    result.save(cleaned_path)
    return cleaned_path


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
            
        typeset_plans.append({
            "id": bubble["id"],
            "action": "translate_replace",
            "text": trans,
            "bbox": [int(v) for v in bubble["bbox"]],
            "style": style,
            "align": "center",
            "font_policy": font_policy,
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

            # Step 5: Typesetting (only if translation exists)
            final_path = None
            if translation_succeeded and cleaned_path:
                try:
                    _progress("typesetting", "start")
                    logger.info(f"[{page_id}] === Typesetting ===")
                    final_path = run_typesetting(cleaned_path, detection, config, tmp_dir, page_id)
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
        env_path = os.path.expanduser("~/.hermes/skills/manga-localization/.env")
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
