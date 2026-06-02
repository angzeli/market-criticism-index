"""Tests for market-data calendar alignment helpers."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from mci.config import RAW_DATA_DIR
from mci.market_data import align_to_trading_days


def test_align_to_trading_days_writes_new_file_without_mutating_input(tmp_path: Path) -> None:
    headline_path = tmp_path / "headlines.csv"
    market_path = tmp_path / "market.csv"
    headline_path.write_text(
        "title,domain,seendate\n"
        "Wall Street warning,example.com,20240105213000\n",
        encoding="utf-8",
    )
    original_headline_csv = headline_path.read_text(encoding="utf-8")
    market_path.write_text("date\n2024-01-05\n2024-01-09\n", encoding="utf-8")
    output_path = tmp_path / "interim" / "headlines_aligned.csv"

    result_path = align_to_trading_days(headline_path, market_path, output_path=output_path)

    assert result_path == output_path
    assert headline_path.read_text(encoding="utf-8") == original_headline_csv

    with result_path.open(encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert rows[0]["published_date_ny"] == "2024-01-05"
    assert rows[0]["trading_day"] == "2024-01-09"


def test_align_to_trading_days_refuses_raw_output_path(tmp_path: Path) -> None:
    headline_path = tmp_path / "headlines.csv"
    market_path = tmp_path / "market.csv"
    headline_path.write_text("title,domain,seendate\n", encoding="utf-8")
    market_path.write_text("date\n2024-01-05\n", encoding="utf-8")

    with pytest.raises(ValueError, match="raw data directory"):
        align_to_trading_days(headline_path, market_path, output_path=RAW_DATA_DIR / "bad_aligned.csv")


def test_align_to_trading_days_raises_when_calendar_coverage_is_insufficient(tmp_path: Path) -> None:
    headline_path = tmp_path / "headlines.csv"
    market_path = tmp_path / "market.csv"
    output_path = tmp_path / "aligned.csv"
    headline_path.write_text(
        "title,domain,seendate\n"
        "Wall Street warning,example.com,20240110213000\n",
        encoding="utf-8",
    )
    market_path.write_text("date\n2024-01-05\n2024-01-09\n", encoding="utf-8")

    with pytest.raises(ValueError, match="2024-01-11.*2024-01-05 to 2024-01-09"):
        align_to_trading_days(headline_path, market_path, output_path=output_path)

    assert not output_path.exists()


def test_align_to_trading_days_preserves_headers_for_empty_headline_csv(tmp_path: Path) -> None:
    headline_path = tmp_path / "empty_headlines.csv"
    market_path = tmp_path / "market.csv"
    output_path = tmp_path / "aligned" / "empty_headlines_aligned.csv"
    headline_path.write_text("title,domain,seendate\n", encoding="utf-8")
    market_path.write_text("date\n2024-01-05\n2024-01-09\n", encoding="utf-8")

    align_to_trading_days(headline_path, market_path, output_path=output_path)

    with output_path.open(encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)

    assert rows == []
    assert reader.fieldnames == [
        "title",
        "domain",
        "seendate",
        "normalised_title",
        "seendate_ny",
        "published_date_ny",
        "trading_day",
    ]
