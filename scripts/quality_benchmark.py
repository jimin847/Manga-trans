#!/usr/bin/env python3
"""Evaluate frozen manga pages without committing the source images."""
import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path


def _normalize(text: str | None) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(text or "")))


def evaluate_page(spec: dict, qa_report: dict) -> dict:
    regions = qa_report.get("bubbles", [])
    failures = []
    minimum_regions = int(spec.get("minimum_regions", 0))
    if len(regions) < minimum_regions:
        failures.append(f"regions: {len(regions)} < required {minimum_regions}")

    recognized = "\n".join(_normalize(region.get("ocr_text")) for region in regions)
    for required in spec.get("required_source_texts", []):
        if _normalize(required) not in recognized:
            failures.append(f"ocr: missing required text {required!r}")

    translated = "\n".join(_normalize(region.get("translation")) for region in regions)
    for required in spec.get("required_translation_terms", []):
        if _normalize(required) not in translated:
            failures.append(f"translation: missing required term {required!r}")

    for alternatives in spec.get("required_translation_any", []):
        if not any(_normalize(term) in translated for term in alternatives):
            failures.append(f"translation: missing any acceptable term {alternatives!r}")

    unresolved = sum(
        1 for region in regions
        if region.get("type", "dialogue") != "sfx" and not region.get("ocr_text")
    )
    allowed_unresolved = int(spec.get("allowed_unresolved_regions", 0))
    if unresolved > allowed_unresolved:
        failures.append(f"ocr: {unresolved} unresolved > allowed {allowed_unresolved}")

    review_pending = sum(
        1 for region in regions
        if region.get("ocr_status") == "needs_review"
        or region.get("translation_status") == "needs_review"
        or region.get("render_status") == "needs_review"
    )
    if review_pending:
        failures.append(f"review: {review_pending} region(s) pending")

    coverage = float(qa_report.get("dialogue_coverage", 0.0))
    if coverage < 100.0:
        failures.append(f"coverage: {coverage:.2f}% < 100%")
    if qa_report.get("issues"):
        failures.append(f"qa: {len(qa_report['issues'])} unresolved issue(s)")

    return {
        "page_id": spec["page_id"],
        "passed": not failures,
        "region_count": len(regions),
        "dialogue_coverage": coverage,
        "failures": failures,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_benchmark(gold_path: Path, results_dir: Path, source_dir: Path | None = None) -> dict:
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    pages = []
    for spec in gold["pages"]:
        page_id = spec["page_id"]
        qa_path = results_dir / f"{page_id}_dialogue_qa.json"
        if not qa_path.exists():
            pages.append({
                "page_id": page_id,
                "passed": False,
                "region_count": 0,
                "dialogue_coverage": 0.0,
                "failures": [f"missing QA report: {qa_path}"],
            })
            continue
        result = evaluate_page(spec, json.loads(qa_path.read_text(encoding="utf-8")))
        if source_dir:
            source_path = source_dir / spec["filename"]
            if not source_path.exists():
                result["failures"].append(f"missing source: {source_path}")
            elif _sha256(source_path) != spec["sha256"]:
                result["failures"].append("source SHA-256 mismatch")
            result["passed"] = not result["failures"]
        pages.append(result)
    return {
        "passed": all(page["passed"] for page in pages),
        "passed_pages": sum(page["passed"] for page in pages),
        "total_pages": len(pages),
        "pages": pages,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the frozen manga quality benchmark")
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    report = run_benchmark(args.gold, args.results_dir, args.source_dir)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
