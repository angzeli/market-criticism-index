#!/usr/bin/env python3
"""Build the daily Market Criticism Index CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mci.config import PROCESSED_DATA_DIR
from mci.index import IndexConstructionSpec, build_daily_mci


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build daily Market Criticism Index variables.")
    parser.add_argument("--market-csv", required=True, type=Path, help="Cleaned all-market headline CSV.")
    parser.add_argument("--criticism-csv", required=True, type=Path, help="Cleaned candidate-criticism headline CSV.")
    parser.add_argument("--labels", type=Path, help="Optional completed-label CSV or directory.")
    parser.add_argument("--output", default=PROCESSED_DATA_DIR / "mci_daily.csv", type=Path)
    parser.add_argument("--rolling-window", default=60, type=int)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing MCI output CSV.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    spec = IndexConstructionSpec(
        market_headlines_path=args.market_csv,
        criticism_headlines_path=args.criticism_csv,
        output_path=args.output,
        labels_path=args.labels,
        rolling_window=args.rolling_window,
        overwrite=args.overwrite,
    )

    try:
        output_path = build_daily_mci(spec)
    except (FileExistsError, OSError, ValueError) as exc:
        print(f"MCI construction did not complete: {exc}", file=sys.stderr)
        return 1

    print(f"Saved daily MCI: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

