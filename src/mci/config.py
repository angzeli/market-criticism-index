"""Shared configuration for the Market Criticism Index scaffold."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"

DEFAULT_START_DATE = date(2022, 1, 1)
DEFAULT_END_DATE = date(2026, 5, 31)

MARKET_QUERY_TERMS = (
    "US stock market",
    "Wall Street",
    "S&P 500",
    "Nasdaq",
    "US equities",
    "American stocks",
)

CRITICISM_QUERY_TERMS = (
    "overvalued",
    "expensive",
    "bubble",
    "mania",
    "crash",
    "correction",
    "selloff",
    "fragile",
    "frothy",
    "speculative",
    "concentration risk",
    "AI bubble",
    "Magnificent Seven",
)

CRITICISM_CATEGORIES = (
    "valuation",
    "bubble_speculation",
    "crash_correction_warning",
    "ai_tech_hype",
    "concentration",
)

MARKET_SYMBOLS = ("SPY", "QQQ", "RSP", "^VIX")


@dataclass(frozen=True)
class MvpConfig:
    """Date range and output conventions for MVP runs."""

    start_date: date = DEFAULT_START_DATE
    end_date: date = DEFAULT_END_DATE
    interim_dir: Path = INTERIM_DATA_DIR
    processed_dir: Path = PROCESSED_DATA_DIR
    figures_dir: Path = FIGURES_DIR
    tables_dir: Path = TABLES_DIR

