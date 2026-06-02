#!/usr/bin/env python3
"""Validate completed K annotation CSVs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mci.annotations import LABELLED_ANNOTATION_DIR, format_validation_report, validate_annotations


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate completed annotation labels.")
    parser.add_argument(
        "--labelled-dir",
        default=LABELLED_ANNOTATION_DIR,
        type=Path,
        help="Directory containing completed labelled CSVs.",
    )
    parser.add_argument("--file", action="append", type=Path, dest="files", help="Specific labelled CSV to include.")
    parser.add_argument("--allow-empty", action="store_true", help="Allow validation to run with no labelled CSVs.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = args.files if args.files else None
    try:
        report = validate_annotations(args.labelled_dir, paths=paths, allow_empty=args.allow_empty)
    except (OSError, ValueError) as exc:
        print(f"Annotation validation did not complete: {exc}", file=sys.stderr)
        return 1

    print(format_validation_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
