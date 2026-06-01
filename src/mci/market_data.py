"""Interfaces for market data collection and trading-day alignment."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Sequence

from mci.config import MARKET_SYMBOLS, RAW_DATA_DIR


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


def align_to_trading_days(headline_path: Path, market_path: Path) -> Path:
    """Align daily headline measures to the relevant trading-day calendar."""

    raise NotImplementedError("Trading-day alignment is not implemented in the scaffold.")

