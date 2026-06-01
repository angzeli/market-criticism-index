#!/usr/bin/env python3
"""Run GDELT headline collection from a source checkout."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mci.gdelt import GdeltClientError, GdeltQueryType, GdeltRequestSpec, collect_gdelt_headlines


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect headline metadata from GDELT DOC 2.0.")
    parser.add_argument(
        "--query-type",
        required=True,
        choices=[query_type.value for query_type in GdeltQueryType],
        help="Headline query to run.",
    )
    parser.add_argument("--start-date", required=True, type=date.fromisoformat)
    parser.add_argument("--end-date", required=True, type=date.fromisoformat)
    parser.add_argument("--max-records", default=250, type=int, help="Maximum records per day.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing deterministic interim output.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    spec = GdeltRequestSpec(
        query_type=GdeltQueryType(args.query_type),
        start_date=args.start_date,
        end_date=args.end_date,
        max_records=args.max_records,
        overwrite=args.overwrite,
    )

    try:
        result = collect_gdelt_headlines(spec)
    except (GdeltClientError, FileExistsError, ValueError) as exc:
        print(f"GDELT collection did not complete: {exc}", file=sys.stderr)
        return 1

    print(f"Saved raw responses: {result.raw_path}")
    print(f"Saved cleaned metadata: {result.interim_path}")
    print(f"Articles: {result.article_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
