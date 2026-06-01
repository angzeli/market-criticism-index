"""Interfaces for Market Criticism Index construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class IndexConstructionSpec:
    """Input and output paths for daily criticism-index construction."""

    market_headlines_path: Path
    criticism_headlines_path: Path
    output_path: Path
    rolling_window: int = 60


def build_daily_mci(spec: IndexConstructionSpec) -> Path:
    """Build raw, normalised, and rolling-standardised daily MCI variables."""

    raise NotImplementedError("MCI construction is not implemented in the scaffold.")


def build_category_indices(spec: IndexConstructionSpec) -> Path:
    """Build category-specific daily criticism-index variables."""

    raise NotImplementedError("Category index construction is not implemented in the scaffold.")

