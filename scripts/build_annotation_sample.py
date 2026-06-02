#!/usr/bin/env python3
"""Build a deterministic mixed annotation sample for K."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mci.annotations import AnnotationSampleSpec, build_annotation_sample


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a mixed headline annotation sample for K.")
    parser.add_argument("--candidate-csv", required=True, type=Path, help="Cleaned candidate-criticism CSV.")
    parser.add_argument("--all-market-csv", required=True, type=Path, help="Cleaned all-market headline CSV.")
    parser.add_argument("--output-path", type=Path, help="Optional explicit annotation sample CSV path.")
    parser.add_argument("--output-dir", type=Path, help="Directory for the default annotation sample filename.")
    parser.add_argument("--likely-criticism-count", default=200, type=int)
    parser.add_argument("--general-market-count", default=200, type=int)
    parser.add_argument("--ambiguous-count", default=100, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--run-date", type=date.fromisoformat, help="YYYY-MM-DD used in the default filename.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing generated sample.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    kwargs = {}
    if args.output_dir is not None:
        kwargs["output_dir"] = args.output_dir

    spec = AnnotationSampleSpec(
        candidate_csv=args.candidate_csv,
        all_market_csv=args.all_market_csv,
        output_path=args.output_path,
        likely_criticism_count=args.likely_criticism_count,
        general_market_count=args.general_market_count,
        ambiguous_count=args.ambiguous_count,
        seed=args.seed,
        run_date=args.run_date,
        overwrite=args.overwrite,
        **kwargs,
    )

    try:
        result = build_annotation_sample(spec)
    except (FileExistsError, OSError, ValueError) as exc:
        print(f"Annotation sample export did not complete: {exc}", file=sys.stderr)
        return 1

    print(f"Saved annotation sample: {result.output_path}")
    print(f"Rows: {result.total_rows}")
    print(f"Counts: {result.counts}")
    print(f"Shortfalls: {result.shortfalls}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
