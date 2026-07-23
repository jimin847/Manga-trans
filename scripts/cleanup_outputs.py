#!/usr/bin/env python3
"""Clean stale local improvement outputs from Manga-trans.

Keeps:
- out_put_test/ (current improvement result bucket)
- output/ (production/default output directory)
- artifacts/ (intentional saved smoke artifacts)

Removes root-level output_* test directories that are no longer the active
out_put_test bucket.
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


def _version_number(path: Path) -> int:
    match = re.match(r".+_v(\d+)$", path.name)
    return int(match.group(1)) if match else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    keep_names = {"out_put_test", "output", "artifacts"}
    candidates = sorted(
        [p for p in root.iterdir() if p.is_dir() and p.name.startswith("output_")],
        key=lambda p: p.stat().st_mtime,
    )

    removed = []
    for path in candidates:
        if path.name in keep_names:
            continue
        removed.append(path)
        if not args.dry_run:
            shutil.rmtree(path)

    for path in removed:
        print(("DRY-RUN " if args.dry_run else "REMOVED ") + str(path))

    bucket = root / "out_put_test"
    if bucket.is_dir():
        versioned = {}
        for path in bucket.iterdir():
            match = re.match(r"(.+)_v(\d+)$", path.name)
            if match:
                versioned.setdefault(match.group(1), []).append(path)
        for paths in versioned.values():
            latest = max(paths, key=_version_number)
            for path in sorted(paths, key=lambda p: p.stat().st_mtime):
                if path is latest:
                    continue
                removed.append(path)
                if not args.dry_run:
                    shutil.rmtree(path)

    for path in removed:
        print(("DRY-RUN " if args.dry_run else "REMOVED ") + str(path))

    print(f"removed={len(removed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
