"""Tests for headline text cleaning and trading-day assignment."""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import pytest

from mci.config import RAW_DATA_DIR
from mci.text_processing import (
    assign_trading_day,
    clean_headline_file,
    clean_headline_record,
    deduplicate_headlines,
    fuzzy_deduplicate_headlines,
    normalise_title,
    parse_seendate_to_new_york,
    TextProcessingSpec,
)


def test_normalise_title_lowercases_removes_punctuation_and_collapses_spaces() -> None:
    assert normalise_title("  S&P 500: Bubble, or Bargain?!  ") == "sp 500 bubble or bargain"


def test_normalise_title_splits_separator_punctuation() -> None:
    assert normalise_title("sell-off warning") == normalise_title("sell off warning")
    assert normalise_title("AI/tech sell-off") == "ai tech sell off"
    assert normalise_title("S&P 500") == "sp 500"


def test_deduplicate_headlines_by_date_normalised_title_and_domain() -> None:
    records = [
        {
            "title": "S&P 500 Bubble Warning!",
            "domain": "Example.com",
            "seendate": "20240101140000",
        },
        {
            "title": "S&P 500: bubble warning.",
            "domain": "example.com",
            "seendate": "20240101150000",
        },
        {
            "title": "S&P 500 Bubble Warning!",
            "domain": "other.example",
            "seendate": "20240101150000",
        },
    ]
    original_first = dict(records[0])

    deduplicated = deduplicate_headlines(records)

    assert len(deduplicated) == 2
    assert deduplicated[0]["normalised_title"] == "sp 500 bubble warning"
    assert records[0] == original_first


def test_post_close_timestamp_is_reassigned_to_next_trading_day() -> None:
    record = {
        "title": "Wall Street correction warning",
        "domain": "example.com",
        "seendate": "20240105213000",
    }

    cleaned = clean_headline_record(record)

    assert cleaned["published_date_ny"] == "2024-01-05"
    assert cleaned["trading_day"] == "2024-01-08"
    assert record == {
        "title": "Wall Street correction warning",
        "domain": "example.com",
        "seendate": "20240105213000",
    }


def test_pre_close_timestamp_stays_on_same_trading_day() -> None:
    assert assign_trading_day("20240105195900") == date(2024, 1, 5)


def test_trading_day_uses_supplied_market_calendar() -> None:
    trading_days = [date(2024, 1, 5), date(2024, 1, 9)]

    assert assign_trading_day("20240105213000", trading_days=trading_days) == date(2024, 1, 9)


def test_parse_iso_timestamp_with_timezone_to_new_york() -> None:
    parsed = parse_seendate_to_new_york("2024-01-05T21:30:00+00:00")

    assert parsed is not None
    assert parsed.isoformat() == "2024-01-05T16:30:00-05:00"


def test_date_only_seendate_values_leave_derived_dates_blank() -> None:
    gdelt_date_only = clean_headline_record(
        {"title": "Wall Street warning", "domain": "example.com", "seendate": "20240105"}
    )
    iso_date_only = clean_headline_record(
        {"title": "Wall Street warning", "domain": "example.com", "seendate": "2024-01-05"}
    )

    assert gdelt_date_only["seendate_ny"] == ""
    assert gdelt_date_only["published_date_ny"] == ""
    assert gdelt_date_only["trading_day"] == ""
    assert iso_date_only["seendate_ny"] == ""
    assert iso_date_only["published_date_ny"] == ""
    assert iso_date_only["trading_day"] == ""


def test_fuzzy_deduplication_is_separate_from_exact_deduplication() -> None:
    records = [
        {"title": "Wall Street bubble warning", "domain": "example.com", "seendate": "20240101140000"},
        {"title": "Wall Street bubble warnings", "domain": "example.com", "seendate": "20240101141000"},
    ]

    assert len(deduplicate_headlines(records)) == 2
    assert len(fuzzy_deduplicate_headlines(records, threshold=0.95)) == 1


def test_clean_headline_file_refuses_raw_output_path(tmp_path: Path) -> None:
    input_path = tmp_path / "headlines.csv"
    input_path.write_text("title,domain,seendate\n", encoding="utf-8")
    spec = TextProcessingSpec(input_path=input_path, output_path=RAW_DATA_DIR / "bad_cleaned.csv")

    with pytest.raises(ValueError, match="raw data directory"):
        clean_headline_file(spec)


def test_clean_headline_file_preserves_headers_for_zero_row_input(tmp_path: Path) -> None:
    input_path = tmp_path / "empty_headlines.csv"
    output_path = tmp_path / "cleaned" / "empty_headlines_cleaned.csv"
    input_path.write_text("title,domain,seendate\n", encoding="utf-8")

    clean_headline_file(TextProcessingSpec(input_path=input_path, output_path=output_path))

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
