#!/usr/bin/env python3
"""Optionally fetch raw benchmark market-price data."""

from __future__ import annotations

import argparse
from dataclasses import fields
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mci.config import RAW_DATA_DIR
from mci.market_data import (
    MARKET_SYMBOLS,
    MarketDataSpec,
    collect_market_data,
    market_data_output_path,
    preflight_market_data_provider,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optionally fetch raw benchmark market-price CSV data.")
    parser.add_argument("--start-date", required=True, type=date.fromisoformat)
    parser.add_argument("--end-date", required=True, type=date.fromisoformat)
    parser.add_argument("--symbols", nargs="+", default=list(MARKET_SYMBOLS), help="Symbols, space or comma separated.")
    parser.add_argument("--raw-output-dir", default=RAW_DATA_DIR / "market", type=Path)
    parser.add_argument("--max-retries", default=_market_data_default("max_retries"), type=int)
    parser.add_argument("--request-pause-seconds", default=_market_data_default("request_pause_seconds"), type=float)
    parser.add_argument("--preflight", action="store_true", help="Run a fast provider probe before collection.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    symbols = _parse_symbols(args.symbols)
    spec = MarketDataSpec(
        start_date=args.start_date,
        end_date=args.end_date,
        symbols=symbols,
        raw_output_dir=args.raw_output_dir,
        max_retries=args.max_retries,
        request_pause_seconds=args.request_pause_seconds,
    )
    output_path = market_data_output_path(spec.raw_output_dir, spec.symbols, spec.start_date, spec.end_date)
    print(f"Expected raw market CSV: {output_path}")

    if args.preflight:
        try:
            preflight_market_data_provider(spec)
        except (RuntimeError, ValueError, OSError) as exc:
            print(f"Market-data provider preflight failed: {exc}", file=sys.stderr)
            print(_local_csv_guidance(output_path), file=sys.stderr)
            return 1

    try:
        saved_path = collect_market_data(spec)
    except (FileExistsError, RuntimeError, ValueError, OSError) as exc:
        print(f"Market-data collection did not complete: {exc}", file=sys.stderr)
        print(_local_csv_guidance(output_path), file=sys.stderr)
        return 1

    print(f"Saved raw market CSV: {saved_path}")
    return 0


def _parse_symbols(values: list[str]) -> tuple[str, ...]:
    symbols: list[str] = []
    for value in values:
        symbols.extend(part.strip() for part in value.split(",") if part.strip())
    return tuple(symbols)


def _market_data_default(field_name: str) -> object:
    for field in fields(MarketDataSpec):
        if field.name == field_name:
            return field.default
    raise ValueError(f"Unknown MarketDataSpec field: {field_name}")


def _local_csv_guidance(output_path: Path) -> str:
    return (
        "HTTP 429 means the provider is rate-limiting requests. Retry later, or place a normalized "
        f"local CSV at {output_path} with columns date,symbol,close and optional "
        "open,high,low,adj_close,volume. Then use scripts/build_panel_daily.py --prices-csv "
        f"{output_path}."
    )


if __name__ == "__main__":
    raise SystemExit(main())
