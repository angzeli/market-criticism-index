"""Baseline event studies and regressions for the MVP empirical analysis."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mci_matplotlib"))

from matplotlib import pyplot as plt

from mci.config import FIGURES_DIR, PROCESSED_DATA_DIR, TABLES_DIR
from mci.text_processing import validate_generated_output_path

CORRELATIONAL_WARNING = "These estimates are correlational and should not be interpreted as causal effects."
EVENT_FIGURE_FILENAME = "event_study_top_decile_mci_spy.png"
EVENT_TABLE_FILENAME = "event_study_top_decile_mci_spy.csv"
REGRESSION_CSV_FILENAME = "baseline_regressions.csv"
REGRESSION_MARKDOWN_FILENAME = "baseline_regressions.md"


@dataclass(frozen=True)
class ModelSpec:
    """Regression and event-study settings for the MVP baseline."""

    panel_path: Path = PROCESSED_DATA_DIR / "panel_daily.csv"
    figures_dir: Path = FIGURES_DIR
    tables_dir: Path = TABLES_DIR
    horizons: Sequence[int] = (1, 5, 21)
    mci_column: str = "mci_rolling_60d_zscore"
    event_window: tuple[int, int] = (-10, 21)
    overwrite: bool = False


@dataclass(frozen=True)
class EventStudyResult:
    """Output paths and counts from the top-decile MCI event study."""

    figure_path: Path
    table_path: Path
    event_count: int
    dropped_incomplete_events: int


@dataclass(frozen=True)
class RegressionResult:
    """Output paths and row count from baseline regressions."""

    csv_path: Path
    markdown_path: Path
    row_count: int


@dataclass(frozen=True)
class MvpAnalysisResult:
    """Output metadata from the full MVP empirical analysis."""

    event_study: EventStudyResult
    regressions: RegressionResult


def run_mvp_analysis(spec: ModelSpec) -> MvpAnalysisResult:
    """Run the MVP event study and baseline regressions."""

    _preflight_mvp_analysis(spec)
    return MvpAnalysisResult(
        event_study=run_event_studies(spec),
        regressions=run_baseline_regressions(spec),
    )


def run_event_studies(spec: ModelSpec) -> EventStudyResult:
    """Run baseline event studies around high-MCI days."""

    window_start, window_end = spec.event_window
    if window_start >= 0 or window_end <= 0:
        raise ValueError("event_window must include days before and after event day zero.")

    figure_path = spec.figures_dir / EVENT_FIGURE_FILENAME
    table_path = spec.tables_dir / EVENT_TABLE_FILENAME
    _prepare_outputs((figure_path, table_path), overwrite=spec.overwrite)

    summary, event_count, dropped_incomplete = _event_study_summary(_read_panel(spec.panel_path), spec)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(table_path, index=False)
    _write_event_plot(summary, figure_path)

    return EventStudyResult(
        figure_path=figure_path,
        table_path=table_path,
        event_count=event_count,
        dropped_incomplete_events=dropped_incomplete,
    )


def run_baseline_regressions(spec: ModelSpec) -> RegressionResult:
    """Run baseline return, volatility, and drawdown regressions."""

    csv_path = spec.tables_dir / REGRESSION_CSV_FILENAME
    markdown_path = spec.tables_dir / REGRESSION_MARKDOWN_FILENAME
    _prepare_outputs((csv_path, markdown_path), overwrite=spec.overwrite)

    panel = _read_panel(spec.panel_path)
    rows: list[dict[str, object]] = []
    for horizon in spec.horizons:
        for model_name, dependent, regressors in _regression_specs(horizon, spec.mci_column):
            rows.extend(_fit_regression_rows(panel, model_name, horizon, dependent, regressors))

    table = pd.DataFrame(rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False)
    _write_regression_markdown(markdown_path, table)
    return RegressionResult(csv_path=csv_path, markdown_path=markdown_path, row_count=len(table))


def _read_panel(path: Path) -> pd.DataFrame:
    panel = pd.read_csv(path)
    if "date" not in panel.columns:
        raise ValueError("panel_path must contain a date column.")

    parsed_dates = pd.to_datetime(panel["date"], errors="coerce")
    invalid_dates = parsed_dates.isna()
    if invalid_dates.any():
        first_index = invalid_dates[invalid_dates].index[0]
        raise ValueError(
            "panel_path has unparseable date values; "
            f"first bad CSV row {int(first_index) + 2} value {panel.loc[first_index, 'date']!r}."
        )

    panel = panel.copy()
    panel["date"] = parsed_dates.dt.strftime("%Y-%m-%d")
    duplicated_dates = panel["date"].duplicated(keep=False)
    if duplicated_dates.any():
        first_duplicate = panel.loc[duplicated_dates, "date"].iloc[0]
        raise ValueError(f"panel_path has duplicate date values; first duplicate date {first_duplicate}.")
    return panel.sort_values("date").reset_index(drop=True)


def _top_decile_event_positions(mci_values: pd.Series) -> list[int]:
    valid_values = mci_values.dropna()
    if valid_values.empty:
        raise ValueError("Event study requires at least one nonmissing MCI z-score value.")

    threshold = valid_values.quantile(0.9)
    return list(valid_values[valid_values >= threshold].index)


def _event_path_for_position(
    spy_returns: pd.Series,
    event_position: int,
    event_window: tuple[int, int],
) -> pd.DataFrame | None:
    window_start, window_end = event_window
    if event_position + window_start < 0 or event_position + window_end >= len(spy_returns):
        return None

    rows: list[dict[str, float | int]] = []
    for relative_day in range(window_start, window_end + 1):
        if relative_day == 0:
            cumulative_return = 0.0
        elif relative_day > 0:
            returns = spy_returns.iloc[event_position : event_position + relative_day]
            if returns.isna().any():
                return None
            cumulative_return = float(returns.sum())
        else:
            returns = spy_returns.iloc[event_position + relative_day : event_position]
            if returns.isna().any():
                return None
            cumulative_return = float(-returns.sum())
        rows.append({"relative_day": relative_day, "cumulative_spy_log_return": cumulative_return})
    return pd.DataFrame(rows)


def _event_study_summary(panel: pd.DataFrame, spec: ModelSpec) -> tuple[pd.DataFrame, int, int]:
    _require_columns(panel, ("date", spec.mci_column, "spy_fwd_log_return_1d"), "event study")
    mci_values = pd.to_numeric(panel[spec.mci_column], errors="coerce")
    spy_returns = pd.to_numeric(panel["spy_fwd_log_return_1d"], errors="coerce")
    event_positions = _top_decile_event_positions(mci_values)

    event_paths: list[pd.DataFrame] = []
    dropped_incomplete = 0
    for event_position in event_positions:
        path = _event_path_for_position(spy_returns, event_position, spec.event_window)
        if path is None:
            dropped_incomplete += 1
            continue
        path["event_date"] = panel.loc[event_position, "date"]
        event_paths.append(path)

    if not event_paths:
        raise ValueError("No complete top-decile MCI event windows were available.")

    combined = pd.concat(event_paths, ignore_index=True)
    summary = (
        combined.groupby("relative_day", sort=True)["cumulative_spy_log_return"]
        .mean()
        .reset_index(name="average_cumulative_spy_log_return")
    )
    summary["event_count"] = len(event_paths)
    return summary, len(event_paths), dropped_incomplete


def _write_event_plot(summary: pd.DataFrame, figure_path: Path) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    axis.plot(summary["relative_day"], summary["average_cumulative_spy_log_return"], color="#1f77b4", linewidth=2)
    axis.axhline(0, color="#666666", linewidth=0.8)
    axis.axvline(0, color="#333333", linewidth=0.8, linestyle="--")
    axis.set_title("Top-Decile MCI Event Study: SPY Cumulative Log Return")
    axis.set_xlabel("Relative trading day")
    axis.set_ylabel("Average cumulative SPY log return")
    axis.grid(alpha=0.25)
    figure.text(0.5, 0.02, CORRELATIONAL_WARNING, ha="center", fontsize=8)
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)


def _regression_specs(horizon: int, mci_column: str) -> list[tuple[str, str, list[str]]]:
    lag_return = f"spy_lag_log_return_{horizon}d"
    return [
        (
            "return",
            f"spy_fwd_log_return_{horizon}d",
            [mci_column, lag_return, "vix_level", "spy_lag_realized_vol_21d"],
        ),
        (
            "future_volatility",
            f"spy_fwd_realized_vol_{horizon}d",
            [mci_column, "spy_lag_realized_vol_21d", "vix_level"],
        ),
        (
            "future_drawdown",
            f"spy_fwd_max_drawdown_{horizon}d",
            [mci_column, lag_return, "vix_level", "spy_lag_realized_vol_21d"],
        ),
    ]


def _fit_regression_rows(
    panel: pd.DataFrame,
    model_name: str,
    horizon: int,
    dependent: str,
    regressors: Sequence[str],
) -> list[dict[str, object]]:
    model_data, y, x = _regression_design(panel, model_name, horizon, dependent, regressors)
    maxlags = _newey_west_maxlags(horizon)
    fit = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    formula = f"{dependent} ~ {' + '.join(regressors)}"

    rows: list[dict[str, object]] = []
    for term in fit.params.index:
        rows.append(
            {
                "model": model_name,
                "horizon": horizon,
                "dependent": dependent,
                "term": term,
                "coefficient": fit.params[term],
                "std_error": fit.bse[term],
                "t_value": fit.tvalues[term],
                "p_value": fit.pvalues[term],
                "n": int(fit.nobs),
                "maxlags": maxlags,
                "formula": formula,
            }
        )
    return rows


def _regression_design(
    panel: pd.DataFrame,
    model_name: str,
    horizon: int,
    dependent: str,
    regressors: Sequence[str],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    columns = [dependent, *regressors]
    _require_columns(panel, columns, f"{model_name} regression horizon {horizon}")
    model_data = panel[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if model_data.empty:
        raise ValueError(f"No complete rows available for {model_name} regression horizon {horizon}.")

    y = model_data[dependent]
    x = sm.add_constant(model_data[list(regressors)], has_constant="add")
    _validate_regression_design(x, model_name, horizon)
    return model_data, y, x


def _validate_regression_design(x: pd.DataFrame, model_name: str, horizon: int) -> None:
    observations, parameters = x.shape
    if observations <= parameters:
        raise ValueError(
            f"{model_name} regression horizon {horizon} is underidentified: "
            f"{observations} complete rows for {parameters} parameters. "
            "Add more observations or reduce regressors."
        )

    rank = np.linalg.matrix_rank(x.to_numpy(dtype=float))
    if rank < parameters:
        raise ValueError(
            f"{model_name} regression horizon {horizon} has a rank-deficient design matrix: "
            f"rank {rank} for {parameters} parameters."
        )


def _newey_west_maxlags(horizon: int) -> int:
    return horizon - 1


def _write_regression_markdown(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["model", "horizon", "term", "coefficient", "std_error", "p_value", "n", "maxlags"]
    lines = [
        "# Baseline Newey-West Regressions",
        "",
        CORRELATIONAL_WARNING,
        "",
        _markdown_table(table[columns]),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(table: pd.DataFrame) -> str:
    columns = list(table.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(_format_markdown_value(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def _format_markdown_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _require_columns(panel: pd.DataFrame, columns: Sequence[str], context: str) -> None:
    missing = [column for column in columns if column not in panel.columns]
    if missing:
        raise ValueError(f"{context} requires missing panel columns: {', '.join(missing)}.")


def _prepare_outputs(paths: Sequence[Path], *, overwrite: bool) -> None:
    for path in paths:
        validate_generated_output_path(path)
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} already exists. Pass overwrite=True to replace it.")


def _preflight_mvp_analysis(spec: ModelSpec) -> None:
    _prepare_outputs(
        (
            spec.figures_dir / EVENT_FIGURE_FILENAME,
            spec.tables_dir / EVENT_TABLE_FILENAME,
            spec.tables_dir / REGRESSION_CSV_FILENAME,
            spec.tables_dir / REGRESSION_MARKDOWN_FILENAME,
        ),
        overwrite=spec.overwrite,
    )
    panel = _read_panel(spec.panel_path)
    _event_study_summary(panel, spec)
    for horizon in spec.horizons:
        for model_name, dependent, regressors in _regression_specs(horizon, spec.mci_column):
            _regression_design(panel, model_name, horizon, dependent, regressors)
