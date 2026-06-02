#!/usr/bin/env python3
"""Build the daily MCI and market-feature panel CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mci.config import PROCESSED_DATA_DIR
from mci.market_data import DEFAULT_MARKET_HORIZONS, MARKET_SYMBOLS, MarketPanelSpec, build_market_panel


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build daily MCI and market-feature panel variables.")
    parser.add_argument("--prices-csv", required=True, type=Path, help="Normalized long market-price CSV.")
    parser.add_argument("--mci-csv", default=PROCESSED_DATA_DIR / "mci_daily.csv", type=Path)
    parser.add_argument("--output", default=PROCESSED_DATA_DIR / "panel_daily.csv", type=Path)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing panel output CSV.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    spec = MarketPanelSpec(
        prices_path=args.prices_csv,
        mci_path=args.mci_csv,
        output_path=args.output,
        overwrite=args.overwrite,
    )

    try:
        output_path = build_market_panel(spec)
    except (FileExistsError, OSError, ValueError) as exc:
        print(f"Panel construction did not complete: {exc}", file=sys.stderr)
        return 1

    panel = pd.read_csv(output_path)
    date_range = "empty"
    if not panel.empty and "date" in panel.columns:
        date_range = f"{panel['date'].min()} to {panel['date'].max()}"

    print(f"Saved daily panel: {output_path}")
    print(f"Date range: {date_range}")
    print(f"Rows: {len(panel)}")
    print(f"Symbols: {', '.join(MARKET_SYMBOLS)}")
    print(f"Feature horizons: {', '.join(str(horizon) for horizon in DEFAULT_MARKET_HORIZONS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
