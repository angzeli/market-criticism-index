"""Tests for MVP empirical analysis routines."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mci.modelling import (
    CORRELATIONAL_WARNING,
    EVENT_FIGURE_FILENAME,
    EVENT_TABLE_FILENAME,
    REGRESSION_CSV_FILENAME,
    REGRESSION_MARKDOWN_FILENAME,
    ModelSpec,
    _newey_west_maxlags,
    _top_decile_event_positions,
    run_baseline_regressions,
    run_event_studies,
    run_mvp_analysis,
)


def test_top_decile_event_selection_uses_nonmissing_mci_values() -> None:
    values = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, np.nan])

    assert _top_decile_event_positions(values) == [9]


def test_event_study_aligns_windows_drops_incomplete_events_and_writes_outputs(tmp_path: Path) -> None:
    panel_path = tmp_path / "panel_daily.csv"
    figures_dir = tmp_path / "figures"
    tables_dir = tmp_path / "tables"
    _write_event_panel(panel_path)

    result = run_event_studies(
        ModelSpec(
            panel_path=panel_path,
            figures_dir=figures_dir,
            tables_dir=tables_dir,
            overwrite=False,
        )
    )

    assert result.figure_path == figures_dir / EVENT_FIGURE_FILENAME
    assert result.table_path == tables_dir / EVENT_TABLE_FILENAME
    assert result.figure_path.exists()
    assert result.event_count == 1
    assert result.dropped_incomplete_events == 1

    event_path = pd.read_csv(result.table_path)
    assert event_path.loc[event_path["relative_day"] == -10, "average_cumulative_spy_log_return"].iloc[0] == pytest.approx(
        -0.10
    )
    assert event_path.loc[event_path["relative_day"] == 0, "average_cumulative_spy_log_return"].iloc[0] == 0
    assert event_path.loc[event_path["relative_day"] == 21, "average_cumulative_spy_log_return"].iloc[0] == pytest.approx(
        0.21
    )


def test_baseline_regressions_write_expected_rows_and_warning(tmp_path: Path) -> None:
    panel_path = tmp_path / "panel_daily.csv"
    tables_dir = tmp_path / "tables"
    _write_regression_panel(panel_path)

    result = run_baseline_regressions(
        ModelSpec(
            panel_path=panel_path,
            tables_dir=tables_dir,
            horizons=(1, 5, 21),
        )
    )

    assert result.csv_path == tables_dir / REGRESSION_CSV_FILENAME
    assert result.markdown_path == tables_dir / REGRESSION_MARKDOWN_FILENAME
    table = pd.read_csv(result.csv_path)

    assert set(table["model"]) == {"return", "future_volatility", "future_drawdown"}
    assert set(table["horizon"]) == {1, 5, 21}
    assert table.loc[table["horizon"] == 1, "maxlags"].unique().tolist() == [0]
    assert table.loc[table["horizon"] == 5, "maxlags"].unique().tolist() == [4]
    assert table.loc[table["horizon"] == 21, "maxlags"].unique().tolist() == [20]
    assert "mci_rolling_60d_zscore" in set(table["term"])
    assert CORRELATIONAL_WARNING in result.markdown_path.read_text(encoding="utf-8")
    assert result.row_count == len(table)


def test_newey_west_lag_rule_is_horizon_minus_one() -> None:
    assert _newey_west_maxlags(1) == 0
    assert _newey_west_maxlags(5) == 4
    assert _newey_west_maxlags(21) == 20


def test_baseline_regressions_fail_with_missing_panel_columns(tmp_path: Path) -> None:
    panel_path = tmp_path / "panel_daily.csv"
    _write_csv(
        panel_path,
        [{"date": "2024-01-01", "mci_rolling_60d_zscore": "1.0"}],
        ("date", "mci_rolling_60d_zscore"),
    )

    with pytest.raises(ValueError, match="spy_fwd_log_return_1d"):
        run_baseline_regressions(ModelSpec(panel_path=panel_path, tables_dir=tmp_path / "tables"))


def test_baseline_regressions_reject_underidentified_design(tmp_path: Path) -> None:
    panel_path = tmp_path / "panel_daily.csv"
    tables_dir = tmp_path / "tables"
    _write_small_regression_panel(panel_path, rows=3, rank_deficient=False)

    with pytest.raises(ValueError, match="underidentified.*3 complete rows for 5 parameters"):
        run_baseline_regressions(
            ModelSpec(
                panel_path=panel_path,
                tables_dir=tables_dir,
                horizons=(1,),
            )
        )

    assert not (tables_dir / REGRESSION_CSV_FILENAME).exists()
    assert not (tables_dir / REGRESSION_MARKDOWN_FILENAME).exists()


def test_baseline_regressions_reject_rank_deficient_design(tmp_path: Path) -> None:
    panel_path = tmp_path / "panel_daily.csv"
    tables_dir = tmp_path / "tables"
    _write_small_regression_panel(panel_path, rows=10, rank_deficient=True)

    with pytest.raises(ValueError, match="rank-deficient design matrix"):
        run_baseline_regressions(
            ModelSpec(
                panel_path=panel_path,
                tables_dir=tables_dir,
                horizons=(1,),
            )
        )

    assert not (tables_dir / REGRESSION_CSV_FILENAME).exists()
    assert not (tables_dir / REGRESSION_MARKDOWN_FILENAME).exists()


def test_run_mvp_analysis_does_not_leave_event_outputs_when_regression_preflight_fails(tmp_path: Path) -> None:
    panel_path = tmp_path / "panel_daily.csv"
    figures_dir = tmp_path / "figures"
    tables_dir = tmp_path / "tables"
    _write_event_panel(panel_path)

    with pytest.raises(ValueError, match="spy_lag_log_return_1d"):
        run_mvp_analysis(
            ModelSpec(
                panel_path=panel_path,
                figures_dir=figures_dir,
                tables_dir=tables_dir,
            )
        )

    assert not (figures_dir / EVENT_FIGURE_FILENAME).exists()
    assert not (tables_dir / EVENT_TABLE_FILENAME).exists()
    assert not (tables_dir / REGRESSION_CSV_FILENAME).exists()
    assert not (tables_dir / REGRESSION_MARKDOWN_FILENAME).exists()


def test_run_mvp_analysis_writes_all_outputs_and_protects_overwrite(tmp_path: Path) -> None:
    panel_path = tmp_path / "panel_daily.csv"
    figures_dir = tmp_path / "figures"
    tables_dir = tmp_path / "tables"
    _write_regression_panel(panel_path)

    result = run_mvp_analysis(
        ModelSpec(
            panel_path=panel_path,
            figures_dir=figures_dir,
            tables_dir=tables_dir,
        )
    )

    assert result.event_study.figure_path.exists()
    assert result.event_study.table_path.exists()
    assert result.regressions.csv_path.exists()
    assert result.regressions.markdown_path.exists()

    with pytest.raises(FileExistsError):
        run_mvp_analysis(
            ModelSpec(
                panel_path=panel_path,
                figures_dir=figures_dir,
                tables_dir=tables_dir,
            )
        )

    run_mvp_analysis(
        ModelSpec(
            panel_path=panel_path,
            figures_dir=figures_dir,
            tables_dir=tables_dir,
            overwrite=True,
        )
    )


def _write_event_panel(path: Path) -> None:
    rows: list[dict[str, str]] = []
    mci_values = [""] * 40
    for offset, value in enumerate(range(1, 19), start=20):
        mci_values[offset] = str(value)
    mci_values[5] = "19"
    mci_values[15] = "20"

    dates = pd.date_range("2024-01-01", periods=40, freq="D").strftime("%Y-%m-%d")
    for index in range(40):
        rows.append(
            {
                "date": dates[index],
                "mci_rolling_60d_zscore": mci_values[index],
                "spy_fwd_log_return_1d": "0.01",
            }
        )
    _write_csv(path, rows, ("date", "mci_rolling_60d_zscore", "spy_fwd_log_return_1d"))


def _write_regression_panel(path: Path) -> None:
    rows: list[dict[str, str]] = []
    dates = pd.date_range("2024-03-01", periods=90, freq="D").strftime("%Y-%m-%d")
    for index in range(90):
        mci = math.sin(index / 6)
        vix = 18 + (index % 13) * 0.35 + index * 0.01
        lag_vol = 0.12 + math.sin(index / 7) * 0.015 + index * 0.0002
        row: dict[str, str] = {
            "date": dates[index],
            "mci_rolling_60d_zscore": str(mci),
            "vix_level": str(vix),
            "spy_lag_realized_vol_21d": str(lag_vol),
        }
        for horizon in (1, 5, 21):
            lag_return = math.cos(index / (horizon + 2)) * 0.01
            row[f"spy_lag_log_return_{horizon}d"] = str(lag_return)
            row[f"spy_fwd_log_return_{horizon}d"] = str(
                0.001 + 0.015 * mci + 0.4 * lag_return - 0.0001 * vix + 0.02 * lag_vol + math.sin(index / 5) * 0.001
            )
            row[f"spy_fwd_realized_vol_{horizon}d"] = str(
                0.15 + 0.01 * abs(mci) + 0.001 * horizon + 0.0005 * vix + math.cos(index / 4) * 0.001
            )
            row[f"spy_fwd_max_drawdown_{horizon}d"] = str(
                -abs(0.005 + 0.004 * mci + 0.2 * lag_return - 0.00005 * vix + math.sin(index / 8) * 0.001)
            )
        rows.append(row)

    _write_csv(path, rows, tuple(rows[0].keys()))


def _write_small_regression_panel(path: Path, *, rows: int, rank_deficient: bool) -> None:
    output_rows: list[dict[str, str]] = []
    dates = pd.date_range("2024-06-01", periods=rows, freq="D").strftime("%Y-%m-%d")
    for index in range(rows):
        if rank_deficient:
            mci = float(index + 1)
            lag_return = 2 * mci
            vix = 3 * mci
            lag_vol = 4 * mci
        else:
            mci = float(index + 1)
            lag_return = float(index + 2)
            vix = float(index + 3)
            lag_vol = float(index + 4)

        output_rows.append(
            {
                "date": dates[index],
                "mci_rolling_60d_zscore": str(mci),
                "vix_level": str(vix),
                "spy_lag_realized_vol_21d": str(lag_vol),
                "spy_lag_log_return_1d": str(lag_return),
                "spy_fwd_log_return_1d": str(0.01 * (index + 1)),
                "spy_fwd_realized_vol_1d": str(0.2 + index * 0.01),
                "spy_fwd_max_drawdown_1d": str(-0.01 * (index + 1)),
            }
        )

    _write_csv(path, output_rows, tuple(output_rows[0].keys()))


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
