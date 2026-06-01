"""Placeholder tests for deterministic scaffold configuration."""

from __future__ import annotations

from mci import config


def test_default_mvp_date_range() -> None:
    assert config.DEFAULT_START_DATE.isoformat() == "2022-01-01"
    assert config.DEFAULT_END_DATE.isoformat() == "2026-05-31"


def test_output_paths_are_deterministic() -> None:
    assert config.INTERIM_DATA_DIR.name == "interim"
    assert config.PROCESSED_DATA_DIR.name == "processed"
    assert config.FIGURES_DIR.name == "figures"
    assert config.TABLES_DIR.name == "tables"

