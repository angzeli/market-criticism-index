"""Interfaces for market data collection and trading-day alignment."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Sequence

from mci.config import INTERIM_DATA_DIR, MARKET_SYMBOLS, RAW_DATA_DIR
from mci.text_processing import (
    DERIVED_FIELDS,
    clean_headline_records,
    validate_generated_output_path,
    validate_trading_calendar_coverage,
)


@dataclass(frozen=True)
class MarketDataSpec:
    """Parameters for collecting benchmark market data."""

    start_date: date
    end_date: date
    symbols: Sequence[str] = MARKET_SYMBOLS
    raw_output_dir: Path = RAW_DATA_DIR / "market"


def collect_market_data(spec: MarketDataSpec) -> Path:
    """Collect price, volatility, and volume data for benchmark symbols."""

    raise NotImplementedError("Market data collection is not implemented in the scaffold.")


def align_to_trading_days(
    headline_path: Path,
    market_path: Path,
    *,
    output_path: Path | None = None,
) -> Path:
    """Align headline records to trading days inferred from a market-data CSV.

    This writes a generated CSV and does not modify input files in place.
    """

    resolved_output_path = output_path or INTERIM_DATA_DIR / f"{headline_path.stem}_aligned.csv"
    validate_generated_output_path(resolved_output_path)

    headline_records, headline_fieldnames = _read_csv_records(headline_path)
    trading_days = _read_market_dates(market_path)
    validate_trading_calendar_coverage(headline_records, trading_days)
    aligned = clean_headline_records(headline_records, trading_days=trading_days)

    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv_records(resolved_output_path, aligned, fieldnames=headline_fieldnames)
    return resolved_output_path


def _read_market_dates(path: Path) -> list[date]:
    records, _ = _read_csv_records(path)
    dates: list[date] = []
    for record in records:
        value = record.get("date") or record.get("Date")
        if value:
            dates.append(date.fromisoformat(value))
    if not dates:
        raise ValueError("market_path must contain a date or Date column.")
    return sorted(set(dates))


def _read_csv_records(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        return list(reader), list(reader.fieldnames or [])


def _write_csv_records(
    path: Path,
    records: Sequence[dict[str, object]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> None:
    output_fieldnames: list[str] = list(fieldnames or [])
    for record in records:
        for field in record:
            if field not in output_fieldnames:
                output_fieldnames.append(str(field))
    for field in DERIVED_FIELDS:
        if field not in output_fieldnames:
            output_fieldnames.append(field)

    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=output_fieldnames)
        writer.writeheader()
        writer.writerows(records)
