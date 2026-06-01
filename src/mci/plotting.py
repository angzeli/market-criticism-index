"""Interfaces for MVP figures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PlotSpec:
    """Input and output paths for deterministic figure generation."""

    input_path: Path
    output_dir: Path


def plot_mci_timeseries(spec: PlotSpec) -> Path:
    """Save a daily Market Criticism Index time-series figure."""

    raise NotImplementedError("MCI plotting is not implemented in the scaffold.")


def plot_event_study_paths(spec: PlotSpec) -> Path:
    """Save event-study path figures."""

    raise NotImplementedError("Event-study plotting is not implemented in the scaffold.")

