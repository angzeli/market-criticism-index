"""Tests for market-feature construction and daily panel merge."""

from __future__ import annotations

import csv
import math
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from mci.config import RAW_DATA_DIR
from mci.market_data import (
    MarketDataSpec,
    MarketPanelSpec,
    build_market_panel,
    collect_market_data,
    market_data_output_path,
)


def test_build_market_panel_computes_market_features_and_preserves_mci_rows(tmp_path: Path) -> None:
    mci_path = tmp_path / "mci_daily.csv"
    prices_path = tmp_path / "market_prices.csv"
    output_path = tmp_path / "panel_daily.csv"
    _write_mci(mci_path, ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"])
    _write_price_fixture(prices_path)

    build_market_panel(
        MarketPanelSpec(
            prices_path=prices_path,
            mci_path=mci_path,
            output_path=output_path,
            horizons=(1, 2),
            realized_vol_window=2,
        )
    )

    output = pd.read_csv(output_path)

    assert output.columns[:3].tolist() == ["date", "MCI", "raw_criticism_count"]
    assert output["date"].tolist() == ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]

    assert output.loc[0, "spy_fwd_log_return_1d"] == pytest.approx(math.log(110 / 100))
    assert output.loc[0, "spy_fwd_log_return_2d"] == pytest.approx(math.log(105 / 100))
    next_two_returns = pd.Series([math.log(110 / 100), math.log(105 / 110)])
    assert output.loc[0, "spy_fwd_realized_vol_1d"] == pytest.approx(abs(math.log(110 / 100)) * math.sqrt(252))
    assert output.loc[0, "spy_fwd_realized_vol_2d"] == pytest.approx(
        math.sqrt((next_two_returns**2).mean()) * math.sqrt(252)
    )
    assert output.loc[1, "spy_lag_log_return_1d"] == pytest.approx(math.log(110 / 100))
    assert output.loc[2, "spy_lag_log_return_2d"] == pytest.approx(math.log(105 / 100))

    first_two_returns = pd.Series([math.log(110 / 100), math.log(105 / 110)])
    expected_realized_vol = first_two_returns.std() * math.sqrt(252)
    assert pd.isna(output.loc[0, "spy_realized_vol_2d"])
    assert pd.isna(output.loc[1, "spy_realized_vol_2d"])
    assert output.loc[2, "spy_realized_vol_2d"] == pytest.approx(expected_realized_vol)
    assert output.loc[3, "spy_lag_realized_vol_2d"] == pytest.approx(expected_realized_vol)

    assert output.loc[0, "spy_fwd_max_drawdown_2d"] == 0
    assert output.loc[1, "spy_fwd_max_drawdown_2d"] == pytest.approx(105 / 110 - 1)
    assert pd.isna(output.loc[3, "spy_fwd_max_drawdown_2d"])

    assert output.loc[0, "vix_level"] == 20
    assert output.loc[0, "vix_fwd_change_1d"] == 2
    assert output.loc[0, "vix_fwd_change_2d"] == 1


def test_build_market_panel_fails_when_mci_date_has_no_current_market_price(tmp_path: Path) -> None:
    mci_path = tmp_path / "mci_daily.csv"
    prices_path = tmp_path / "market_prices.csv"
    output_path = tmp_path / "panel_daily.csv"
    _write_mci(mci_path, ["2024-01-10"])
    _write_price_fixture(prices_path)

    with pytest.raises(ValueError, match="no current-day market price row"):
        build_market_panel(MarketPanelSpec(prices_path=prices_path, mci_path=mci_path, output_path=output_path))

    assert not output_path.exists()


def test_build_market_panel_rejects_duplicate_mci_dates(tmp_path: Path) -> None:
    mci_path = tmp_path / "mci_daily.csv"
    prices_path = tmp_path / "market_prices.csv"
    output_path = tmp_path / "panel_daily.csv"
    _write_mci(mci_path, ["2024-01-01", "2024-01-01"])
    _write_price_fixture(prices_path)

    with pytest.raises(ValueError, match="duplicate date values.*2024-01-01"):
        build_market_panel(MarketPanelSpec(prices_path=prices_path, mci_path=mci_path, output_path=output_path))

    assert not output_path.exists()


def test_build_market_panel_rejects_missing_intervening_price_coverage(tmp_path: Path) -> None:
    mci_path = tmp_path / "mci_daily.csv"
    prices_path = tmp_path / "market_prices.csv"
    output_path = tmp_path / "panel_daily.csv"
    _write_mci(mci_path, ["2024-01-01"])
    rows = [
        row
        for row in _price_fixture_rows()
        if not (row["date"] == "2024-01-02" and row["symbol"] == "SPY")
    ]
    _write_csv(prices_path, rows, ("date", "symbol", "close", "adj_close"))

    with pytest.raises(ValueError, match="missing required price coverage for SPY on 2024-01-02"):
        build_market_panel(
            MarketPanelSpec(
                prices_path=prices_path,
                mci_path=mci_path,
                output_path=output_path,
                horizons=(2,),
            )
        )

    assert not output_path.exists()


def test_build_market_panel_rejects_conflicting_duplicate_prices(tmp_path: Path) -> None:
    mci_path = tmp_path / "mci_daily.csv"
    prices_path = tmp_path / "market_prices.csv"
    output_path = tmp_path / "panel_daily.csv"
    _write_mci(mci_path, ["2024-01-01"])
    rows = _price_fixture_rows()
    rows.append({"date": "2024-01-01", "symbol": "SPY", "close": "1000", "adj_close": "101"})
    _write_csv(prices_path, rows, ("date", "symbol", "close", "adj_close"))

    with pytest.raises(ValueError, match="Conflicting duplicate market price rows"):
        build_market_panel(MarketPanelSpec(prices_path=prices_path, mci_path=mci_path, output_path=output_path))


@pytest.mark.parametrize("bad_price", ["0", "-1"])
def test_build_market_panel_rejects_non_positive_selected_prices(tmp_path: Path, bad_price: str) -> None:
    mci_path = tmp_path / "mci_daily.csv"
    prices_path = tmp_path / "market_prices.csv"
    output_path = tmp_path / "panel_daily.csv"
    _write_mci(mci_path, ["2024-01-01"])
    rows = _price_fixture_rows()
    rows[0]["adj_close"] = bad_price
    _write_csv(prices_path, rows, ("date", "symbol", "close", "adj_close"))

    with pytest.raises(ValueError, match=f"non-positive.*first bad CSV row 2 value '{bad_price}'"):
        build_market_panel(MarketPanelSpec(prices_path=prices_path, mci_path=mci_path, output_path=output_path))

    assert not output_path.exists()


def test_build_market_panel_allows_vix_duplicates_with_same_close_and_different_adj_close(tmp_path: Path) -> None:
    mci_path = tmp_path / "mci_daily.csv"
    prices_path = tmp_path / "market_prices.csv"
    output_path = tmp_path / "panel_daily.csv"
    _write_mci(mci_path, ["2024-01-01"])
    rows = _price_fixture_rows()
    rows.append({"date": "2024-01-01", "symbol": "^VIX", "close": "20", "adj_close": ""})
    _write_csv(prices_path, rows, ("date", "symbol", "close", "adj_close"))

    build_market_panel(
        MarketPanelSpec(
            prices_path=prices_path,
            mci_path=mci_path,
            output_path=output_path,
            horizons=(1,),
        )
    )

    output = pd.read_csv(output_path)
    assert output.loc[0, "vix_fwd_change_1d"] == 2


def test_build_market_panel_rejects_one_day_realized_volatility_window(tmp_path: Path) -> None:
    mci_path = tmp_path / "mci_daily.csv"
    prices_path = tmp_path / "market_prices.csv"
    output_path = tmp_path / "panel_daily.csv"
    _write_mci(mci_path, ["2024-01-01"])
    _write_price_fixture(prices_path)

    with pytest.raises(ValueError, match="realized_vol_window must be at least 2"):
        build_market_panel(
            MarketPanelSpec(
                prices_path=prices_path,
                mci_path=mci_path,
                output_path=output_path,
                realized_vol_window=1,
            )
        )

    assert not output_path.exists()


def test_build_market_panel_rejects_missing_required_symbols(tmp_path: Path) -> None:
    mci_path = tmp_path / "mci_daily.csv"
    prices_path = tmp_path / "market_prices.csv"
    output_path = tmp_path / "panel_daily.csv"
    _write_mci(mci_path, ["2024-01-01"])
    rows = [row for row in _price_fixture_rows() if row["symbol"] != "RSP"]
    _write_csv(prices_path, rows, ("date", "symbol", "close", "adj_close"))

    with pytest.raises(ValueError, match="missing required symbols: RSP"):
        build_market_panel(MarketPanelSpec(prices_path=prices_path, mci_path=mci_path, output_path=output_path))


def test_build_market_panel_rejects_raw_output_and_protects_existing_output(tmp_path: Path) -> None:
    mci_path = tmp_path / "mci_daily.csv"
    prices_path = tmp_path / "market_prices.csv"
    output_path = tmp_path / "panel_daily.csv"
    _write_mci(mci_path, ["2024-01-01"])
    _write_price_fixture(prices_path)

    with pytest.raises(ValueError, match="raw data directory"):
        build_market_panel(
            MarketPanelSpec(
                prices_path=prices_path,
                mci_path=mci_path,
                output_path=RAW_DATA_DIR / "panel_daily.csv",
            )
        )

    output_path.write_text("existing\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        build_market_panel(MarketPanelSpec(prices_path=prices_path, mci_path=mci_path, output_path=output_path))

    build_market_panel(
        MarketPanelSpec(
            prices_path=prices_path,
            mci_path=mci_path,
            output_path=output_path,
            overwrite=True,
        )
    )

    assert pd.read_csv(output_path)["date"].tolist() == ["2024-01-01"]


def test_market_data_output_path_is_deterministic_and_refuses_existing_raw_cache(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw_market"
    spec = MarketDataSpec(
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 31),
        raw_output_dir=raw_dir,
    )
    expected = market_data_output_path(raw_dir, spec.symbols, spec.start_date, spec.end_date)
    raw_dir.mkdir()
    expected.write_text("existing\n", encoding="utf-8")

    assert market_data_output_path(raw_dir, spec.symbols, spec.start_date, spec.end_date) == expected

    with pytest.raises(FileExistsError, match="Raw market data is never overwritten"):
        collect_market_data(spec)


def _write_mci(path: Path, dates: list[str]) -> None:
    rows = [
        {"date": day, "MCI": str(index / 10), "raw_criticism_count": str(index)}
        for index, day in enumerate(dates, start=1)
    ]
    _write_csv(path, rows, ("date", "MCI", "raw_criticism_count"))


def _write_price_fixture(path: Path) -> None:
    _write_csv(path, _price_fixture_rows(), ("date", "symbol", "close", "adj_close"))


def _price_fixture_rows() -> list[dict[str, str]]:
    dates = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    spy_adj = [100, 110, 105, 126, 126]
    qqq_close = [50, 55, 60, 66, 72]
    rsp_close = [200, 198, 202, 210, 205]
    vix_close = [20, 22, 21, 25, 23]

    rows: list[dict[str, str]] = []
    for index, day in enumerate(dates):
        rows.extend(
            [
                {"date": day, "symbol": "SPY", "close": "1000", "adj_close": str(spy_adj[index])},
                {"date": day, "symbol": "QQQ", "close": str(qqq_close[index]), "adj_close": ""},
                {"date": day, "symbol": "RSP", "close": str(rsp_close[index]), "adj_close": ""},
                {"date": day, "symbol": "VIX", "close": str(vix_close[index]), "adj_close": "900"},
            ]
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
