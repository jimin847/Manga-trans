#!/usr/bin/env python3
"""
run_batch.py — 배치 처리 실행기 (+ 재시작/이어하기 지원)

Usage:
    python3 run_batch.py --api-key KEY ch15/*.jpg
    python3 run_batch.py --resume ch15/*.jpg          # 기존 완료건 skip
    python3 run_batch.py --report batch.json ch16/*.png
"""
import argparse
import glob
import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("batch")


def resolve_images(patterns: list[str]) -> list[str]:
    """Glob 패턴을 실제 파일 경로 리스트로 변환 (정렬 포함)"""
    files = []
    seen = set()
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            if path not in seen:
                seen.add(path)
                files.append(path)
    return files


def format_eta(seconds: float) -> str:
    """초 → HH:MM:SS"""
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    elif m:
        return f"{m}m{s:02d}s"
    else:
        return f"{s}s"


# ── 진행 상태 파일 관리 ──────────────────────────────────────────────────

RESUME_FILE = ".batch_progress.json"


def load_progress() -> dict:
    """저장된 진행 상태 불러오기"""
    if os.path.exists(RESUME_FILE):
        try:
            with open(RESUME_FILE, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.warning(f"Corrupted progress file ({RESUME_FILE}), starting fresh.")
    return {"completed": [], "failed": [], "partial": [], "results": [], "context": []}


def save_progress(progress: dict):
    """진행 상태 저장"""
    with open(RESUME_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Manga Pipeline v2 — Batch Runner")
    parser.add_argument("patterns", nargs="+", help="Glob patterns (e.g. ch15/*.jpg)")
    parser.add_argument("--api-key", default="", help="OpenRouter API key (default: from .env)")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--output", default=None, help="Output directory override")
    parser.add_argument("--max-pages", type=int, default=0, help="Max pages (0=unlimited)")
    parser.add_argument("--report", default="batch_report.json", help="Batch report output path")
    parser.add_argument("--resume", action="store_true", help="Skip completed pages & continue from last progress")
    parser.add_argument("--skip-existing", action="store_true", help="Skip pages with _ko.png already present")
    args = parser.parse_args()

    # ── Config & images ──────────────────────────────────────────────────

    images = resolve_images(args.patterns)
    if not images:
        logger.error("No matching images found.")
        sys.exit(1)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from main import process_page, load_config

    config = load_config(args.config)
    if args.output:
        config["output"]["base_dir"] = args.output

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        env_path = os.path.expanduser("~/.hermes/skills/manga-localization/.env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    if line.startswith("OPENROUTER_API_KEY="):
                        api_key = line.split("=", 1)[1].strip().strip("\"'")
                        break
    if not api_key:
        logger.warning("No API key — skipping OCR and translation (detection + inpainting only)")

    # ── Resume / skip-existing ───────────────────────────────────────────

    progress = load_progress() if args.resume else {}
    out_dir = Path(config["output"]["base_dir"])

    filtered = []
    skipped_reason = {}
    for img in images:
        page_id = Path(img).stem
        # --resume: skip if in progress.completed
        if args.resume and page_id in progress.get("completed", []):
            skipped_reason[page_id] = "already completed (from progress)"
            continue
        # --skip-existing: skip if _ko.png exists
        if args.skip_existing and (out_dir / f"{page_id}_ko.png").exists():
            skipped_reason[page_id] = "output already exists"
            continue
        filtered.append(img)

    if skipped_reason:
        logger.info(f"Skipped {len(skipped_reason)} page(s) via {'--resume' if args.resume else '--skip-existing'}:")
        for pid, reason in skipped_reason.items():
            logger.info(f"  - {pid} ({reason})")

    images = filtered
    if not images:
        logger.info("All pages already processed. Nothing to do.")
        sys.exit(0)

    if args.max_pages > 0:
        images = images[:args.max_pages]

    logger.info(f"Processing {len(images)} image(s)")
    logger.info(f"Output dir: {args.output or '(config default)'}")
    logger.info(f"{'='*50}")

    # ── Batch loop ───────────────────────────────────────────────────────

    batch_start = time.time()
    previous_results = list(progress.get("results", []))
    results = list(previous_results)
    remaining_count = len(images)
    total_count = len(previous_results) + remaining_count
    global_context = list(progress.get("context", []))

    for idx, img_path in enumerate(images, 1):
        t0 = time.time()
        page_id = Path(img_path).stem

        elapsed = time.time() - batch_start
        avg_per_img = elapsed / idx if idx > 0 else 0
        remaining = (remaining_count - idx) * avg_per_img if avg_per_img else 0
        logger.info(f"\n{'─'*50}")
        logger.info(f"[{idx}/{remaining_count}] {page_id}  |  elapsed: {format_eta(elapsed)}  |  ETA: {format_eta(remaining)}")
        logger.info(f"{'─'*50}")

        try:
            result = process_page(img_path, config, api_key, previous_context=list(global_context))
            page_time = time.time() - t0
            result["_batch_time_s"] = round(page_time, 1)
            status = result.get("status", "complete")
            icon = "✅" if status == "complete" else "⚠️" if status == "partial" else "✅"
            logger.info(f"  {icon} Done in {page_time:.1f}s")
            results.append(result)

            # Update global context
            if "translations" in result:
                global_context.extend(result["translations"])
                if len(global_context) > 10:
                    global_context = global_context[-10:]

            # Save progress periodically
            progress["completed"] = list(set(progress.get("completed", []) + [page_id]))
            progress["results"] = results
            progress["context"] = list(global_context)
            save_progress(progress)

        except Exception as e:
            page_time = time.time() - t0
            logger.error(f"  ❌ FAILED ({page_time:.1f}s): {e}", exc_info=True)
            result = {
                "page_id": page_id,
                "status": "error",
                "error": str(e),
                "_batch_time_s": round(page_time, 1),
            }
            results.append(result)
            progress["failed"] = list(set(progress.get("failed", []) + [page_id]))
            progress["results"] = results
            save_progress(progress)

    # ── Summary ──────────────────────────────────────────────────────────

    total_time = time.time() - batch_start
    success = sum(1 for r in results if r.get("status") in ("complete", "partial"))

    logger.info(f"\n{'='*50}")
    logger.info(f"BATCH COMPLETE — {success}/{total_count} succeeded ({format_eta(total_time)})")
    logger.info(f"{'='*50}")
    for r in results:
        s = r.get("status", "error")
        icon = "✅" if s == "complete" else "⚠️" if s == "partial" else "❌"
        ts = r.get("_batch_time_s", 0)
        note = f" [{'; '.join(r.get('errors', []))}]" if r.get("errors") else ""
        logger.info(f"  {icon} [{ts:.1f}s] {r['page_id']}{note}")

    # ── Report ───────────────────────────────────────────────────────────

    report = {
        "batch": {
            "total": total_count,
            "success": success,
            "failed": total_count - success,
            "total_time_s": round(total_time, 1),
            "avg_time_s": round(total_time / total_count, 1) if total_count else 0,
            "resumed": args.resume,
        },
        "pages": [
            {
                "page_id": r["page_id"],
                "status": r.get("status", "error"),
                "time_s": r.get("_batch_time_s", 0),
                "final_image": r.get("final_image"),
                "error": r.get("error"),
                "bubble_count": len(r.get("detection", {}).get("bubbles", [])),
                "translated_count": sum(
                    1 for b in r.get("detection", {}).get("bubbles", [])
                    if b.get("translation")
                ),
            }
            for r in results
        ],
    }

    report_path = args.report
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"  Batch report: {report_path}")

    # Clean up progress file on successful completion
    if os.path.exists(RESUME_FILE) and success == total_count:
        try:
            os.remove(RESUME_FILE)
            logger.info(f"  Progress file cleaned up (all {total_count} succeeded)")
        except OSError:
            pass


if __name__ == "__main__":
    main()
