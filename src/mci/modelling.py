"""Interfaces for baseline event studies and regressions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class ModelSpec:
    """Regression and event-study settings for the MVP baseline."""

    panel_path: Path
    output_dir: Path
    horizons: Sequence[int] = (1, 5, 21)
    mci_column: str = "mci_z_60"


def run_event_studies(spec: ModelSpec) -> Path:
    """Run baseline event studies around high-MCI days."""

    raise NotImplementedError("Event studies are not implemented in the scaffold.")


def run_baseline_regressions(spec: ModelSpec) -> Path:
    """Run baseline return, volatility, and drawdown regressions."""

    raise NotImplementedError("Baseline regressions are not implemented in the scaffold.")

