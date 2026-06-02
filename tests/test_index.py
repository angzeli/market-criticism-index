"""Tests for daily Market Criticism Index construction."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

from mci.config import CRITICISM_CATEGORIES, RAW_DATA_DIR
from mci.index import IndexConstructionSpec, _ratio_series, build_daily_mci


def test_build_daily_mci_counts_ratios_ordering_and_rolling_zscore(tmp_path: Path) -> None:
    market_path = tmp_path / "market.csv"
    criticism_path = tmp_path / "criticism.csv"
    output_path = tmp_path / "mci.csv"
    _write_csv(
        market_path,
        [
            {"title": "Market A", "domain": "a.com", "trading_day": "2024-01-03"},
            {"title": "Market B", "domain": "b.com", "trading_day": "2024-01-01"},
            {"title": "Market C", "domain": "c.com", "trading_day": "2024-01-02"},
            {"title": "Market D", "domain": "d.com", "trading_day": "2024-01-03"},
        ],
        ("title", "domain", "trading_day"),
    )
    _write_csv(
        criticism_path,
        [
            {"title": "Candidate A", "domain": "a.com", "trading_day": "2024-01-02"},
            {"title": "Candidate B", "domain": "b.com", "trading_day": "2024-01-03"},
            {"title": "Candidate C", "domain": "c.com", "trading_day": "2024-01-03"},
        ],
        ("title", "domain", "trading_day"),
    )

    build_daily_mci(
        IndexConstructionSpec(
            market_headlines_path=market_path,
            criticism_headlines_path=criticism_path,
            output_path=output_path,
            rolling_window=3,
        )
    )

    output = pd.read_csv(output_path)

    assert list(output.columns) == [
        "date",
        "raw_criticism_count",
        "total_market_article_count",
        "MCI",
        "mci_rolling_60d_zscore",
    ]
    assert output["date"].tolist() == ["2024-01-01", "2024-01-02", "2024-01-03"]
    assert output["total_market_article_count"].tolist() == [1, 1, 2]
    assert output["raw_criticism_count"].tolist() == [0, 1, 2]
    assert output["MCI"].tolist() == [0.0, 1.0, 1.0]
    assert pd.isna(output.loc[0, "mci_rolling_60d_zscore"])
    assert pd.isna(output.loc[1, "mci_rolling_60d_zscore"])
    assert output.loc[2, "mci_rolling_60d_zscore"] == pytest.approx(0.577350269, rel=1e-6)


def test_ratio_series_writes_nan_for_zero_denominator() -> None:
    ratios = _ratio_series(pd.Series([1, 0]), pd.Series([0, 2]))

    assert pd.isna(ratios.iloc[0])
    assert ratios.iloc[1] == 0


def test_rolling_zscore_is_blank_for_zero_variance(tmp_path: Path) -> None:
    market_path = tmp_path / "market.csv"
    criticism_path = tmp_path / "criticism.csv"
    output_path = tmp_path / "mci.csv"
    rows = [
        {"title": "Market A", "domain": "a.com", "trading_day": "2024-01-01"},
        {"title": "Market B", "domain": "b.com", "trading_day": "2024-01-02"},
        {"title": "Market C", "domain": "c.com", "trading_day": "2024-01-03"},
    ]
    _write_csv(market_path, rows, ("title", "domain", "trading_day"))
    _write_csv(
        criticism_path,
        [
            {"title": "Candidate A", "domain": "a.com", "trading_day": "2024-01-01"},
            {"title": "Candidate B", "domain": "b.com", "trading_day": "2024-01-02"},
            {"title": "Candidate C", "domain": "c.com", "trading_day": "2024-01-03"},
        ],
        ("title", "domain", "trading_day"),
    )

    build_daily_mci(
        IndexConstructionSpec(
            market_headlines_path=market_path,
            criticism_headlines_path=criticism_path,
            output_path=output_path,
            rolling_window=2,
        )
    )

    output = pd.read_csv(output_path)

    assert output["MCI"].tolist() == [1.0, 1.0, 1.0]
    assert output["mci_rolling_60d_zscore"].isna().all()


def test_labels_filter_negatives_keep_unmatched_and_add_category_columns(tmp_path: Path) -> None:
    market_path = tmp_path / "market.csv"
    criticism_path = tmp_path / "criticism.csv"
    labels_path = tmp_path / "labels.csv"
    output_path = tmp_path / "mci.csv"
    _write_csv(
        market_path,
        [
            {"title": "Market A", "domain": "a.com", "trading_day": "2024-01-01"},
            {"title": "Market B", "domain": "b.com", "trading_day": "2024-01-01"},
            {"title": "Market C", "domain": "c.com", "trading_day": "2024-01-01"},
            {"title": "Market D", "domain": "d.com", "trading_day": "2024-01-02"},
        ],
        ("title", "domain", "trading_day"),
    )
    _write_csv(
        criticism_path,
        [
            {
                "title": "Valuation warning",
                "normalised_title": "valuation warning",
                "domain": "a.com",
                "url": "https://a.com/1",
                "trading_day": "2024-01-01",
            },
            {
                "title": "Not criticism",
                "normalised_title": "not criticism",
                "domain": "b.com",
                "url": "https://b.com/1",
                "trading_day": "2024-01-01",
            },
            {
                "title": "Unmatched automated candidate",
                "normalised_title": "unmatched automated candidate",
                "domain": "c.com",
                "url": "https://c.com/1",
                "trading_day": "2024-01-01",
            },
            {
                "title": "Concentration risk grows",
                "normalised_title": "concentration risk grows",
                "domain": "d.com",
                "url": "",
                "trading_day": "2024-01-02",
            },
        ],
        ("title", "normalised_title", "domain", "url", "trading_day"),
    )
    _write_csv(
        labels_path,
        [
            _label_row("Valuation warning", "a.com", "https://a.com/1", "2024-01-01", "1", "valuation"),
            _label_row("Not criticism", "b.com", "https://b.com/1", "2024-01-01", "0", ""),
            _label_row("Concentration risk grows", "d.com", "", "2024-01-02", "1", "concentration"),
            _label_row("Sampled all-market positive", "z.com", "https://z.com/1", "2024-01-01", "1", "bubble_speculation"),
            _label_row("Unknown category", "x.com", "https://x.com/1", "2024-01-01", "1", "unknown"),
        ],
        ("title", "domain", "url", "date", "criticism_label", "category_label"),
    )

    build_daily_mci(
        IndexConstructionSpec(
            market_headlines_path=market_path,
            criticism_headlines_path=criticism_path,
            labels_path=labels_path,
            output_path=output_path,
            rolling_window=2,
        )
    )

    output = pd.read_csv(output_path)

    assert output["raw_criticism_count"].tolist() == [2, 1]
    assert output["MCI"].tolist() == pytest.approx([2 / 3, 1.0])
    for category in CRITICISM_CATEGORIES:
        assert f"raw_criticism_count_{category}" in output.columns
        assert f"MCI_{category}" in output.columns
    assert output["raw_criticism_count_valuation"].tolist() == [1, 0]
    assert output["MCI_valuation"].tolist() == pytest.approx([1 / 3, 0.0])
    assert output["raw_criticism_count_concentration"].tolist() == [0, 1]
    assert output["MCI_concentration"].tolist() == pytest.approx([0.0, 1.0])
    assert output["raw_criticism_count_bubble_speculation"].tolist() == [0, 0]


def test_date_source_priority_uses_trading_day_before_other_dates(tmp_path: Path) -> None:
    market_path = tmp_path / "market.csv"
    criticism_path = tmp_path / "criticism.csv"
    output_path = tmp_path / "mci.csv"
    _write_csv(
        market_path,
        [{"title": "Market", "domain": "a.com", "trading_day": "2024-01-02", "date": "2024-01-01"}],
        ("title", "domain", "trading_day", "date"),
    )
    _write_csv(
        criticism_path,
        [{"title": "Candidate", "domain": "a.com", "trading_day": "2024-01-02", "date": "2024-01-01"}],
        ("title", "domain", "trading_day", "date"),
    )

    build_daily_mci(
        IndexConstructionSpec(
            market_headlines_path=market_path,
            criticism_headlines_path=criticism_path,
            output_path=output_path,
        )
    )

    output = pd.read_csv(output_path)

    assert output["date"].tolist() == ["2024-01-02"]
    assert output["MCI"].tolist() == [1.0]


def test_build_daily_mci_refuses_raw_output_path(tmp_path: Path) -> None:
    market_path, criticism_path = _write_minimal_inputs(tmp_path)

    with pytest.raises(ValueError, match="raw data directory"):
        build_daily_mci(
            IndexConstructionSpec(
                market_headlines_path=market_path,
                criticism_headlines_path=criticism_path,
                output_path=RAW_DATA_DIR / "mci_daily.csv",
            )
        )


def test_build_daily_mci_refuses_existing_output_without_overwrite(tmp_path: Path) -> None:
    market_path, criticism_path = _write_minimal_inputs(tmp_path)
    output_path = tmp_path / "mci.csv"
    output_path.write_text("existing\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        build_daily_mci(
            IndexConstructionSpec(
                market_headlines_path=market_path,
                criticism_headlines_path=criticism_path,
                output_path=output_path,
            )
        )

    build_daily_mci(
        IndexConstructionSpec(
            market_headlines_path=market_path,
            criticism_headlines_path=criticism_path,
            output_path=output_path,
            overwrite=True,
        )
    )

    assert pd.read_csv(output_path)["MCI"].tolist() == [1.0]


def _write_minimal_inputs(tmp_path: Path) -> tuple[Path, Path]:
    market_path = tmp_path / "market.csv"
    criticism_path = tmp_path / "criticism.csv"
    _write_csv(
        market_path,
        [{"title": "Market", "domain": "a.com", "trading_day": "2024-01-01"}],
        ("title", "domain", "trading_day"),
    )
    _write_csv(
        criticism_path,
        [{"title": "Candidate", "domain": "a.com", "trading_day": "2024-01-01"}],
        ("title", "domain", "trading_day"),
    )
    return market_path, criticism_path


def _label_row(
    title: str,
    domain: str,
    url: str,
    date: str,
    criticism_label: str,
    category_label: str,
) -> dict[str, str]:
    return {
        "title": title,
        "domain": domain,
        "url": url,
        "date": date,
        "criticism_label": criticism_label,
        "category_label": category_label,
    }


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

