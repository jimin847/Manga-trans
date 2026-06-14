#!/usr/bin/env python3
"""
qa_module.py — 파이프라인 내장 QA + 독립 실행형 검증

Pipeline mode:
    from qa_module import check_page_result
    qa = check_page_result(page_result, config=config.get("qa", {}))

Standalone mode:
    python3 qa_module.py --dir ./output
    python3 qa_module.py --report batch_report.json --dir ./output
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)
logger = logging.getLogger("qa")


def check_page_result(result: dict, config: dict | None = None) -> dict:
    """
    process_page()의 반환 dict를 직접 검증 (in-memory, 디스크 불필요).

    v2 호환: _result.json / _text_mask.png 필요 없음.
    result dict는 {page_id, final_image, detection: {bubbles: [...]}, errors, status} 형태.
    """
    page_id = result.get("page_id", "?")
    issues = []
    cfg = config or {}

    warn_threshold = cfg.get("warn_threshold", 40)
    bad_threshold = cfg.get("bad_threshold", 10)

    # 1. Detection stats
    bubbles = result.get("detection", {}).get("bubbles", [])
    total_bubbles = len(bubbles)
    text_detected = sum(1 for b in bubbles if b.get("ocr_text"))
    translated = sum(1 for b in bubbles if b.get("translation"))

    ocr_rate = (text_detected / total_bubbles * 100) if total_bubbles else 0
    trans_rate = (translated / total_bubbles * 100) if total_bubbles else 0

    if total_bubbles == 0:
        issues.append("detect: no bubbles found")
    elif trans_rate < bad_threshold:
        issues.append(f"trans: very low coverage ({trans_rate:.0f}%)")
    elif trans_rate < warn_threshold:
        issues.append(f"trans: low coverage ({trans_rate:.0f}%)")

    # 2. OCR → translation gap
    gap = text_detected - translated
    if gap > 0:
        issues.append(f"ocr>trans: {gap} text(s) not translated")

    # 3. Step errors from pipeline
    step_errors = result.get("errors")
    if step_errors:
        for err in step_errors:
            issues.append(f"step: {err}")

    # 4. Final image exists
    final_img = result.get("final_image")
    if final_img:
        final_path = Path(final_img)
        if not final_path.exists():
            issues.append(f"missing final image: {final_img}")
    else:
        issues.append("no final image path")

    severity = "ok"
    if len(issues) >= 3:
        severity = "bad"
    elif len(issues) >= 1:
        severity = "warn"

    return {
        "page_id": page_id,
        "severity": severity,
        "bubbles": total_bubbles,
        "ocr_ok": text_detected,
        "translated": translated,
        "ocr_rate_pct": round(ocr_rate, 0),
        "trans_rate_pct": round(trans_rate, 0),
        "issues": issues,
    }


def check_result_json(result: dict, page_dir: Path) -> dict:
    """
    레거시 호환: result.json dict + 페이지 디렉토리 기반 검증.
    v2에서는 process_page() 반환 dict를 check_page_result()로 검증 권장.
    """
    page_id = result.get("page_id", "?")
    issues = []

    # 1. Detection stats from result.json
    bubbles = result.get("detection", {}).get("bubbles", [])
    total_bubbles = len(bubbles)
    text_detected = sum(1 for b in bubbles if b.get("ocr_text"))
    translated = sum(1 for b in bubbles if b.get("translation"))

    ocr_rate = (text_detected / total_bubbles * 100) if total_bubbles else 0
    trans_rate = (translated / total_bubbles * 100) if total_bubbles else 0

    if total_bubbles == 0:
        issues.append("detect: no bubbles found")
    elif trans_rate < 10:
        issues.append(f"trans: very low coverage ({trans_rate:.0f}%)")
    elif trans_rate < 40:
        issues.append(f"trans: low coverage ({trans_rate:.0f}%)")

    gap = text_detected - translated
    if gap > 0:
        issues.append(f"ocr>trans: {gap} text(s) not translated")

    # 2. Final image check
    final_img = result.get("final_image")
    if final_img:
        final_path = Path(final_img)
        if not final_path.is_absolute():
            final_path = page_dir / final_path
        if not final_path.exists():
            issues.append(f"missing final image: {final_img}")
    else:
        issues.append("no final image path")

    # 3. Inpainting quality (optional, needs mask file on disk)
    if final_img and final_path.exists():
        mask_fn = page_id + "_text_mask.png"
        mask_path = page_dir / mask_fn
        if mask_path.exists():
            try:
                mask = np.array(Image.open(mask_path).convert("L"))
                inpainted_area = mask > 128
                inpaint_px = inpainted_area.sum()
                if inpaint_px > 100:
                    gray = np.mean(np.array(Image.open(final_path).convert("RGB")), axis=2)
                    region = gray[inpainted_area]
                    if len(region) > 10:
                        region_var = np.var(region.astype(np.float32))
                        if region_var < 20:
                            issues.append(f"inpaint: low variance ({region_var:.0f}) in masked area")
            except Exception as e:
                issues.append(f"inpaint check error: {e}")

    severity = "ok"
    if len(issues) >= 3:
        severity = "bad"
    elif len(issues) >= 1:
        severity = "warn"

    return {
        "page_id": page_id,
        "severity": severity,
        "bubbles": total_bubbles,
        "ocr_ok": text_detected,
        "translated": translated,
        "ocr_rate_pct": round(ocr_rate, 0),
        "trans_rate_pct": round(trans_rate, 0),
        "issues": issues,
    }


def scan_output_dir(output_dir: str, report_path: str = None) -> list:
    """output 디렉토리의 모든 result.json 스캔 (레거시 모드)"""
    out_path = Path(output_dir)
    result_files = sorted(out_path.glob("*_result.json"))
    nested = list(out_path.glob("*/**/*_result.json"))
    result_files.extend(nested)

    if not result_files:
        logger.error(f"No result files found in {output_dir}")
        return []

    reports = []
    for rf in result_files:
        try:
            with open(rf, encoding="utf-8") as f:
                result = json.load(f)
        except Exception as e:
            logger.warning(f"  Skip {rf.name}: {e}")
            continue

        report = check_result_json(result, rf.parent)
        reports.append(report)
        icon = "✅" if report["severity"] == "ok" else "⚠️" if report["severity"] == "warn" else "❌"
        logger.info(f"  {icon} {report['page_id']}: "
                     f"{report['translated']}/{report['bubbles']} trans "
                     f"({report['trans_rate_pct']:.0f}%)"
                     + (f" | {'; '.join(report['issues'])}" if report['issues'] else ""))

    total = len(reports)
    ok_count = sum(1 for r in reports if r["severity"] == "ok")
    warn_count = sum(1 for r in reports if r["severity"] == "warn")
    bad_count = sum(1 for r in reports if r["severity"] == "bad")
    avg_ocr = np.mean([r["ocr_rate_pct"] for r in reports]) if reports else 0
    avg_trans = np.mean([r["trans_rate_pct"] for r in reports]) if reports else 0

    logger.info(f"\n{'='*50}")
    logger.info(f"QA SUMMARY — {total} page(s)")
    logger.info(f"  ✅ OK:  {ok_count}")
    logger.info(f"  ⚠️ WARN: {warn_count}")
    logger.info(f"  ❌ BAD:  {bad_count}")
    logger.info(f"  Avg OCR: {avg_ocr:.0f}%  |  Avg Translation: {avg_trans:.0f}%")
    logger.info(f"{'='*50}")

    if report_path:
        agg = {
            "summary": {
                "total": total,
                "ok": ok_count,
                "warn": warn_count,
                "bad": bad_count,
                "avg_ocr_pct": round(avg_ocr, 0),
                "avg_trans_pct": round(avg_trans, 0),
            },
            "pages": reports,
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(agg, f, indent=2, ensure_ascii=False)
        logger.info(f"  QA report saved: {report_path}")

    return reports


def main():
    parser = argparse.ArgumentParser(description="Manga Pipeline v2 — QA Module")
    parser.add_argument("--dir", default=None, help="Output directory to scan")
    parser.add_argument("--report", default=None, help="Save QA report JSON path")
    args = parser.parse_args()

    if args.dir:
        scan_output_dir(args.dir, args.report)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
