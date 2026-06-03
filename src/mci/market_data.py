"""Interfaces for market data collection and trading-day alignment."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

from mci.config import INTERIM_DATA_DIR, MARKET_SYMBOLS, PROCESSED_DATA_DIR, RAW_DATA_DIR
from mci.text_processing import (
    DERIVED_FIELDS,
    clean_headline_records,
    validate_generated_output_path,
    validate_trading_calendar_coverage,
)

ETF_SYMBOLS = ("SPY", "QQQ", "RSP")
VIX_SYMBOL = "^VIX"
DEFAULT_MARKET_HORIZONS = (1, 5, 21)
MARKET_PRICE_COLUMNS = ("date", "symbol", "open", "high", "low", "close", "adj_close", "volume")
YAHOO_DOWNLOAD_ENDPOINT = "https://query1.finance.yahoo.com/v7/finance/download/{symbol}"


@dataclass(frozen=True)
class MarketDataSpec:
    """Parameters for collecting benchmark market data."""

    start_date: date
    end_date: date
    symbols: Sequence[str] = MARKET_SYMBOLS
    raw_output_dir: Path = RAW_DATA_DIR / "market"
    timeout: int = 30


@dataclass(frozen=True)
class MarketPanelSpec:
    """Input and output paths for daily MCI and market-feature panel construction."""

    prices_path: Path
    mci_path: Path = PROCESSED_DATA_DIR / "mci_daily.csv"
    output_path: Path = PROCESSED_DATA_DIR / "panel_daily.csv"
    symbols: Sequence[str] = MARKET_SYMBOLS
    horizons: Sequence[int] = DEFAULT_MARKET_HORIZONS
    realized_vol_window: int = 21
    overwrite: bool = False


def collect_market_data(spec: MarketDataSpec) -> Path:
    """Collect price, volatility, and volume data for benchmark symbols."""

    if spec.start_date > spec.end_date:
        raise ValueError("start_date must be on or before end_date.")

    symbols = _canonical_symbols(spec.symbols)
    output_path = market_data_output_path(spec.raw_output_dir, symbols, spec.start_date, spec.end_date)
    if output_path.exists():
        raise FileExistsError(f"{output_path} already exists. Raw market data is never overwritten.")

    session = requests.Session()
    rows: list[dict[str, str]] = []
    for symbol in symbols:
        rows.extend(_fetch_yahoo_daily_prices(session, symbol, spec.start_date, spec.end_date, spec.timeout))

    spec.raw_output_dir.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=MARKET_PRICE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def build_market_panel(spec: MarketPanelSpec) -> Path:
    """Merge daily MCI variables with market features and save a panel CSV."""

    if not spec.horizons:
        raise ValueError("At least one horizon is required.")
    if any(horizon <= 0 for horizon in spec.horizons):
        raise ValueError("All horizons must be positive.")
    if spec.realized_vol_window < 2:
        raise ValueError("realized_vol_window must be at least 2.")

    validate_generated_output_path(spec.output_path)
    if spec.output_path.exists() and not spec.overwrite:
        raise FileExistsError(f"{spec.output_path} already exists. Pass overwrite=True to replace it.")

    symbols = _canonical_symbols(spec.symbols)
    mci = _read_mci_csv(spec.mci_path)
    prices = _read_market_price_csv(spec.prices_path, symbols)
    features, price_wide = _market_feature_frame(
        prices,
        symbols=symbols,
        horizons=tuple(spec.horizons),
        realized_vol_window=spec.realized_vol_window,
    )
    _validate_market_coverage(
        mci,
        price_wide,
        symbols=symbols,
        horizons=tuple(spec.horizons),
        realized_vol_window=spec.realized_vol_window,
    )

    panel = mci.merge(features, on="date", how="left", sort=False)
    spec.output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(spec.output_path, index=False, na_rep="")
    return spec.output_path


def market_data_output_path(raw_output_dir: Path, symbols: Sequence[str], start_date: date, end_date: date) -> Path:
    """Return the deterministic raw market-data cache path."""

    symbol_slug = "_".join(_symbol_prefix(symbol) for symbol in _canonical_symbols(symbols))
    return raw_output_dir / f"market_prices_{symbol_slug}_{start_date:%Y%m%d}_{end_date:%Y%m%d}.csv"


def _fetch_yahoo_daily_prices(
    session: requests.Session,
    symbol: str,
    start_date: date,
    end_date: date,
    timeout: int,
) -> list[dict[str, str]]:
    period1 = _unix_timestamp(start_date)
    period2 = _unix_timestamp(end_date + timedelta(days=1))
    url = YAHOO_DOWNLOAD_ENDPOINT.format(symbol=quote(symbol, safe=""))
    params = {
        "period1": str(period1),
        "period2": str(period2),
        "interval": "1d",
        "events": "history",
        "includeAdjustedClose": "true",
    }

    response = session.get(url, params=params, timeout=timeout)
    if response.status_code >= 400:
        raise RuntimeError(f"Market data request for {symbol} returned HTTP {response.status_code}.")

    reader = csv.DictReader(io.StringIO(response.text))
    rows: list[dict[str, str]] = []
    for row in reader:
        if not row.get("Date") or row.get("Close") in {"", "null", None}:
            continue
        rows.append(
            {
                "date": _csv_value(row.get("Date")),
                "symbol": symbol,
                "open": _csv_value(row.get("Open")),
                "high": _csv_value(row.get("High")),
                "low": _csv_value(row.get("Low")),
                "close": _csv_value(row.get("Close")),
                "adj_close": _csv_value(row.get("Adj Close")),
                "volume": _csv_value(row.get("Volume")),
            }
        )

    if not rows:
        raise RuntimeError(f"Market data request for {symbol} returned no usable daily rows.")
    return rows


def _unix_timestamp(day: date) -> int:
    return int(datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc).timestamp())


def _csv_value(value: object) -> str:
    return "" if value is None else str(value).strip()


def _read_mci_csv(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "date" not in data.columns:
        raise ValueError("mci_path must contain a date column.")
    parsed_dates = _parse_required_dates(data["date"], "mci_path")
    data = data.copy()
    data["date"] = parsed_dates
    duplicated_dates = data["date"].duplicated(keep=False)
    if duplicated_dates.any():
        first_duplicate = data.loc[duplicated_dates, "date"].iloc[0]
        raise ValueError(f"mci_path has duplicate date values; first duplicate date {first_duplicate}.")
    return data


def _read_market_price_csv(path: Path, required_symbols: Sequence[str]) -> pd.DataFrame:
    data = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing_columns = [column for column in ("date", "symbol", "close") if column not in data.columns]
    if missing_columns:
        raise ValueError(f"prices_path is missing required columns: {', '.join(missing_columns)}.")

    data = data.copy()
    data["_date"] = _parse_required_dates(data["date"], "prices_path")
    data["_symbol"] = data["symbol"].map(_canonical_symbol)
    _validate_required_symbols(data, required_symbols)
    data["_price"] = _selected_price_series(data)
    _validate_duplicate_price_rows(data)
    data = data.drop_duplicates(subset=["_date", "_symbol"], keep="first").copy()
    return data[["_date", "_symbol", "_price"]]


def _parse_required_dates(values: pd.Series, path_name: str) -> pd.Series:
    text_values = values.astype(str).str.strip()
    parsed = pd.to_datetime(text_values.replace("", pd.NA), errors="coerce")
    invalid = parsed.isna()
    if invalid.any():
        first_index = invalid[invalid].index[0]
        raise ValueError(
            f"{path_name} has {int(invalid.sum())} row(s) without a parseable date; "
            f"first bad CSV row {int(first_index) + 2} value {text_values.loc[first_index]!r}."
        )
    return parsed.dt.strftime("%Y-%m-%d")


def _validate_required_symbols(data: pd.DataFrame, required_symbols: Sequence[str]) -> None:
    present = set(data["_symbol"])
    missing = [symbol for symbol in required_symbols if symbol not in present]
    if missing:
        raise ValueError(f"prices_path is missing required symbols: {', '.join(missing)}.")


def _validate_duplicate_price_rows(data: pd.DataFrame) -> None:
    duplicate_rows = data[data.duplicated(subset=["_date", "_symbol"], keep=False)]
    if duplicate_rows.empty:
        return

    for (day, symbol), group in duplicate_rows.groupby(["_date", "_symbol"], sort=True):
        if group["_price"].nunique(dropna=False) > 1:
            raise ValueError(
                f"Conflicting duplicate market price rows for {symbol} on {day}; selected price differs."
            )


def _selected_price_series(data: pd.DataFrame) -> pd.Series:
    close_values = data["close"].astype(str).str.strip()
    selected_values = close_values.copy()
    if "adj_close" in data.columns:
        adj_close_values = data["adj_close"].astype(str).str.strip()
        use_adjusted = data["_symbol"].isin(ETF_SYMBOLS) & (adj_close_values != "")
        selected_values.loc[use_adjusted] = adj_close_values.loc[use_adjusted]

    prices = pd.to_numeric(selected_values.replace("", pd.NA), errors="coerce")
    invalid = prices.isna()
    if invalid.any():
        first_index = invalid[invalid].index[0]
        raise ValueError(
            "prices_path has nonnumeric or missing selected price values; "
            f"first bad CSV row {int(first_index) + 2} value {selected_values.loc[first_index]!r}."
        )

    prices = prices.astype(float)
    invalid_price = (~np.isfinite(prices)) | (prices <= 0)
    if invalid_price.any():
        first_index = invalid_price[invalid_price].index[0]
        raise ValueError(
            "prices_path has non-positive or non-finite selected price values; "
            f"first bad CSV row {int(first_index) + 2} value {selected_values.loc[first_index]!r}."
        )
    return prices


def _market_feature_frame(
    prices: pd.DataFrame,
    *,
    symbols: Sequence[str],
    horizons: Sequence[int],
    realized_vol_window: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    price_wide = prices.pivot(index="_date", columns="_symbol", values="_price").sort_index()
    features = pd.DataFrame({"date": price_wide.index.astype(str)})

    for symbol in symbols:
        if symbol not in ETF_SYMBOLS:
            continue
        price = price_wide[symbol]
        prefix = _symbol_prefix(symbol)
        one_day_returns = np.log(price / price.shift(1))

        for horizon in horizons:
            features[f"{prefix}_fwd_log_return_{horizon}d"] = np.log(price.shift(-horizon) / price).to_numpy()
        for horizon in horizons:
            features[f"{prefix}_fwd_realized_vol_{horizon}d"] = _forward_realized_vol(
                one_day_returns,
                horizon,
            ).to_numpy()
        for horizon in horizons:
            features[f"{prefix}_lag_log_return_{horizon}d"] = np.log(price / price.shift(horizon)).to_numpy()

        realized_vol_column = f"{prefix}_realized_vol_{realized_vol_window}d"
        realized_vol = one_day_returns.rolling(
            window=realized_vol_window,
            min_periods=realized_vol_window,
        ).std() * np.sqrt(252)
        features[realized_vol_column] = realized_vol.to_numpy()
        features[f"{prefix}_lag_realized_vol_{realized_vol_window}d"] = realized_vol.shift(1).to_numpy()

        for horizon in horizons:
            features[f"{prefix}_fwd_max_drawdown_{horizon}d"] = _forward_max_drawdown(price, horizon).to_numpy()

    if VIX_SYMBOL in symbols:
        vix = price_wide[VIX_SYMBOL]
        features["vix_level"] = vix.to_numpy()
        for horizon in horizons:
            features[f"vix_fwd_change_{horizon}d"] = (vix.shift(-horizon) - vix).to_numpy()

    return features, price_wide


def _forward_realized_vol(one_day_returns: pd.Series, horizon: int) -> pd.Series:
    returns = one_day_returns.to_numpy(dtype=float)
    realized_vol = np.full(len(returns), np.nan, dtype=float)
    for index in range(len(returns)):
        end = index + horizon
        if end >= len(returns):
            continue
        future_returns = returns[index + 1 : end + 1]
        if np.isnan(future_returns).any():
            continue
        realized_vol[index] = float(np.sqrt(np.mean(future_returns**2)) * np.sqrt(252))
    return pd.Series(realized_vol, index=one_day_returns.index, dtype="float64")


def _forward_max_drawdown(price: pd.Series, horizon: int) -> pd.Series:
    values = price.to_numpy(dtype=float)
    drawdowns = np.full(len(values), np.nan, dtype=float)
    for index, base_price in enumerate(values):
        end = index + horizon
        if end >= len(values) or np.isnan(base_price):
            continue
        future_prices = values[index + 1 : end + 1]
        if np.isnan(future_prices).any():
            continue
        drawdowns[index] = min(0.0, float(np.min(future_prices / base_price - 1.0)))
    return pd.Series(drawdowns, index=price.index, dtype="float64")


def _validate_market_coverage(
    mci: pd.DataFrame,
    price_wide: pd.DataFrame,
    *,
    symbols: Sequence[str],
    horizons: Sequence[int],
    realized_vol_window: int,
) -> None:
    if mci.empty:
        return

    mci_dates = mci["date"].astype(str)
    missing_dates = [day for day in mci_dates if day not in price_wide.index]
    if missing_dates:
        raise ValueError(f"MCI date has no current-day market price row: {missing_dates[0]}.")

    current_prices = price_wide.reindex(mci_dates)
    missing_prices = current_prices[list(symbols)].isna()
    if missing_prices.any().any():
        first_row, first_column = np.argwhere(missing_prices.to_numpy())[0]
        first_date = missing_prices.index[first_row]
        first_symbol = missing_prices.columns[first_column]
        raise ValueError(f"MCI date {first_date} is missing a current-day price for {first_symbol}.")

    _validate_required_price_windows(
        mci_dates,
        price_wide,
        symbols=symbols,
        horizons=horizons,
        realized_vol_window=realized_vol_window,
    )


def _validate_required_price_windows(
    mci_dates: pd.Series,
    price_wide: pd.DataFrame,
    *,
    symbols: Sequence[str],
    horizons: Sequence[int],
    realized_vol_window: int,
) -> None:
    market_dates = list(price_wide.index.astype(str))

    for mci_date in mci_dates:
        position = market_dates.index(mci_date)
        for symbol in symbols:
            if symbol in ETF_SYMBOLS:
                ranges = _required_etf_ranges(position, len(market_dates), horizons, realized_vol_window)
                for start, end in ranges:
                    _validate_symbol_window(price_wide, symbol, start, end, mci_date)
            elif symbol == VIX_SYMBOL:
                for horizon in horizons:
                    future_position = position + horizon
                    if future_position < len(market_dates):
                        _validate_symbol_window(price_wide, symbol, future_position, future_position, mci_date)


def _required_etf_ranges(
    position: int,
    market_date_count: int,
    horizons: Sequence[int],
    realized_vol_window: int,
) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []

    for horizon in horizons:
        if position + horizon < market_date_count:
            ranges.append((position, position + horizon))
        if position - horizon >= 0:
            ranges.append((position - horizon, position - horizon))
    if position >= realized_vol_window:
        ranges.append((position - realized_vol_window, position))
    if position > realized_vol_window:
        ranges.append((position - realized_vol_window - 1, position - 1))

    return ranges


def _validate_symbol_window(
    price_wide: pd.DataFrame,
    symbol: str,
    start: int,
    end: int,
    mci_date: str,
) -> None:
    window = price_wide[symbol].iloc[start : end + 1]
    if not window.isna().any():
        return

    first_missing_date = window[window.isna()].index[0]
    raise ValueError(
        f"prices_path is missing required price coverage for {symbol} on {first_missing_date}; "
        f"needed by MCI date {mci_date}."
    )


def _canonical_symbols(symbols: Sequence[str]) -> tuple[str, ...]:
    canonical: list[str] = []
    for symbol in symbols:
        normalized = _canonical_symbol(symbol)
        if normalized not in canonical:
            canonical.append(normalized)
    return tuple(canonical)


def _canonical_symbol(symbol: object) -> str:
    text = "" if symbol is None else str(symbol).strip().upper()
    if text in {"VIX", "^VIX"}:
        return VIX_SYMBOL
    return text


def _symbol_prefix(symbol: str) -> str:
    if _canonical_symbol(symbol) == VIX_SYMBOL:
        return "vix"
    return _canonical_symbol(symbol).lower()


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
