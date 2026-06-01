"""Interfaces for headline data collection.

The MVP will start with structured headline sources such as GDELT. This module
only defines the scaffold; it does not download or write data yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Sequence

from mci.config import MARKET_QUERY_TERMS, RAW_DATA_DIR


@dataclass(frozen=True)
class HeadlineCollectionSpec:
    """Parameters for collecting market-related headline metadata."""

    start_date: date
    end_date: date
    query_terms: Sequence[str] = MARKET_QUERY_TERMS
    raw_output_dir: Path = RAW_DATA_DIR / "gdelt"


def collect_market_headlines(spec: HeadlineCollectionSpec) -> Path:
    """Collect market-related headlines and return the raw-output path."""

    raise NotImplementedError("Headline collection is not implemented in the scaffold.")


def collect_candidate_criticism_headlines(spec: HeadlineCollectionSpec) -> Path:
    """Collect or filter candidate criticism headlines and return an interim path."""

    raise NotImplementedError("Candidate criticism collection is not implemented in the scaffold.")

