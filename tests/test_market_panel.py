"""Tests for market-feature construction and daily panel merge."""

from __future__ import annotations

import csv
import math
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import requests

import mci.market_data as market_data_module
from mci.config import RAW_DATA_DIR
from mci.market_data import (
    MarketDataSpec,
    MarketPanelSpec,
    _fetch_yahoo_daily_prices,
    build_market_panel,
    collect_market_data,
    market_data_output_path,
    preflight_market_data_provider,
    validate_market_price_csv,
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


def test_provider_preflight_fails_fast_on_http_429() -> None:
    session = _FakeMarketSession([_FakeMarketResponse(429, "")])
    spec = MarketDataSpec(start_date=date(2024, 1, 2), end_date=date(2024, 1, 2))

    with pytest.raises(RuntimeError, match="HTTP 429.*rate-limiting.*local market CSV"):
        preflight_market_data_provider(spec, session=session)

    assert len(session.requests) == 1


def test_collect_market_data_writes_no_raw_file_when_fetch_fails(tmp_path: Path, monkeypatch) -> None:
    session = _FakeMarketSession([_FakeMarketResponse(429, "")])
    monkeypatch.setattr(market_data_module.requests, "Session", lambda: session)
    spec = MarketDataSpec(
        start_date=date(2024, 1, 2),
        end_date=date(2024, 1, 2),
        symbols=("SPY",),
        raw_output_dir=tmp_path,
        max_retries=0,
    )
    expected = market_data_output_path(tmp_path, spec.symbols, spec.start_date, spec.end_date)

    with pytest.raises(RuntimeError, match="HTTP 429"):
        collect_market_data(spec)

    assert not expected.exists()


def test_validate_market_price_csv_accepts_valid_normalized_csv(tmp_path: Path) -> None:
    prices_path = tmp_path / "market_prices.csv"
    _write_price_fixture(prices_path)

    assert validate_market_price_csv(prices_path) == prices_path


def test_validate_market_price_csv_rejects_missing_columns(tmp_path: Path) -> None:
    prices_path = tmp_path / "market_prices.csv"
    _write_csv(prices_path, [{"date": "2024-01-01", "symbol": "SPY"}], ("date", "symbol"))

    with pytest.raises(ValueError, match="missing required columns: close"):
        validate_market_price_csv(prices_path)


def test_validate_market_price_csv_rejects_missing_symbols(tmp_path: Path) -> None:
    prices_path = tmp_path / "market_prices.csv"
    rows = [row for row in _price_fixture_rows() if row["symbol"] != "RSP"]
    _write_csv(prices_path, rows, ("date", "symbol", "close", "adj_close"))

    with pytest.raises(ValueError, match="missing required symbols: RSP"):
        validate_market_price_csv(prices_path)


def test_validate_market_price_csv_rejects_conflicting_duplicates(tmp_path: Path) -> None:
    prices_path = tmp_path / "market_prices.csv"
    rows = _price_fixture_rows()
    rows.append({"date": "2024-01-01", "symbol": "SPY", "close": "1000", "adj_close": "101"})
    _write_csv(prices_path, rows, ("date", "symbol", "close", "adj_close"))

    with pytest.raises(ValueError, match="Conflicting duplicate market price rows"):
        validate_market_price_csv(prices_path)


def test_validate_market_price_csv_rejects_nonpositive_selected_prices(tmp_path: Path) -> None:
    prices_path = tmp_path / "market_prices.csv"
    rows = _price_fixture_rows()
    rows[0]["adj_close"] = "0"
    _write_csv(prices_path, rows, ("date", "symbol", "close", "adj_close"))

    with pytest.raises(ValueError, match="non-positive"):
        validate_market_price_csv(prices_path)


def test_yahoo_fetch_retries_http_429_with_retry_after_header() -> None:
    session = _FakeMarketSession(
        [
            _FakeMarketResponse(429, "", headers={"Retry-After": "7"}),
            _FakeMarketResponse(200, _yahoo_price_csv()),
        ]
    )
    sleeps: list[float] = []

    rows = _fetch_yahoo_daily_prices(
        session,
        "SPY",
        date(2024, 1, 1),
        date(2024, 1, 2),
        timeout=30,
        max_retries=2,
        backoff_seconds=60,
        max_backoff_seconds=120,
        request_pause_seconds=0,
        sleep=sleeps.append,
    )

    assert len(session.requests) == 2
    assert sleeps == [7.0]
    assert rows == [
        {
            "date": "2024-01-02",
            "symbol": "SPY",
            "open": "100",
            "high": "102",
            "low": "99",
            "close": "101",
            "adj_close": "101",
            "volume": "1000",
        }
    ]


def test_yahoo_fetch_retries_timeout_then_success() -> None:
    session = _FakeMarketSession(
        [
            requests.Timeout("timed out"),
            _FakeMarketResponse(200, _yahoo_price_csv()),
        ]
    )
    sleeps: list[float] = []

    rows = _fetch_yahoo_daily_prices(
        session,
        "SPY",
        date(2024, 1, 1),
        date(2024, 1, 2),
        timeout=30,
        max_retries=1,
        backoff_seconds=3,
        max_backoff_seconds=30,
        request_pause_seconds=0,
        sleep=sleeps.append,
    )

    assert len(session.requests) == 2
    assert sleeps == [3]
    assert rows[0]["symbol"] == "SPY"


def test_yahoo_fetch_repeated_connection_errors_fail_after_retry_budget() -> None:
    session = _FakeMarketSession(
        [
            requests.ConnectionError("network down"),
            requests.ConnectionError("still down"),
            requests.ConnectionError("still down"),
        ]
    )
    sleeps: list[float] = []

    with pytest.raises(RuntimeError, match="SPY failed after 3 attempts.*ConnectionError.*local market CSV"):
        _fetch_yahoo_daily_prices(
            session,
            "SPY",
            date(2024, 1, 1),
            date(2024, 1, 2),
            timeout=30,
            max_retries=2,
            backoff_seconds=5,
            max_backoff_seconds=8,
            request_pause_seconds=0,
            sleep=sleeps.append,
        )

    assert len(session.requests) == 3
    assert sleeps == [5, 8]


def test_yahoo_fetch_caps_exponential_backoff_and_fails_after_retries() -> None:
    session = _FakeMarketSession(
        [
            _FakeMarketResponse(429, ""),
            _FakeMarketResponse(429, ""),
            _FakeMarketResponse(429, ""),
        ]
    )
    sleeps: list[float] = []

    with pytest.raises(RuntimeError, match="HTTP 429 after 3 attempts"):
        _fetch_yahoo_daily_prices(
            session,
            "SPY",
            date(2024, 1, 1),
            date(2024, 1, 2),
            timeout=30,
            max_retries=2,
            backoff_seconds=30,
            max_backoff_seconds=45,
            request_pause_seconds=0,
            sleep=sleeps.append,
        )

    assert len(session.requests) == 3
    assert sleeps == [30, 45]


def test_yahoo_fetch_no_rows_message_mentions_wider_trading_range() -> None:
    session = _FakeMarketSession(
        [
            _FakeMarketResponse(200, "Date,Open,High,Low,Close,Adj Close,Volume\n"),
        ]
    )

    with pytest.raises(RuntimeError, match="no usable daily rows.*known trading day.*wider trading-date range"):
        _fetch_yahoo_daily_prices(
            session,
            "SPY",
            date(2024, 1, 6),
            date(2024, 1, 6),
            timeout=30,
            max_retries=0,
            request_pause_seconds=0,
        )


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


class _FakeMarketResponse:
    def __init__(self, status_code: int, text: str, headers: dict[str, str] | None = None) -> None:
        self.status_code = status_code
        self.text = text
        self.headers = headers or {}


class _FakeMarketSession:
    def __init__(self, responses: list[_FakeMarketResponse | requests.RequestException]) -> None:
        self.responses = responses
        self.requests: list[dict[str, object]] = []

    def get(self, url: str, params: dict[str, str], timeout: int) -> _FakeMarketResponse:
        self.requests.append({"url": url, "params": params, "timeout": timeout})
        response = self.responses.pop(0)
        if isinstance(response, requests.RequestException):
            raise response
        return response


def _yahoo_price_csv() -> str:
    return (
        "Date,Open,High,Low,Close,Adj Close,Volume\n"
        "2024-01-02,100,102,99,101,101,1000\n"
    )
