"""Daily Market Criticism Index construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mci.config import CRITICISM_CATEGORIES, PROCESSED_DATA_DIR
from mci.text_processing import normalise_title, validate_generated_output_path

DATE_SOURCE_COLUMNS = ("trading_day", "date", "published_date_ny")
BASE_OUTPUT_COLUMNS = (
    "date",
    "raw_criticism_count",
    "total_market_article_count",
    "MCI",
)


@dataclass(frozen=True)
class IndexConstructionSpec:
    """Input and output paths for daily criticism-index construction."""

    market_headlines_path: Path
    criticism_headlines_path: Path
    output_path: Path = PROCESSED_DATA_DIR / "mci_daily.csv"
    labels_path: Path | None = None
    rolling_window: int = 60
    overwrite: bool = False


def build_daily_mci(spec: IndexConstructionSpec) -> Path:
    """Build daily Market Criticism Index variables and save them as CSV."""

    if spec.rolling_window <= 0:
        raise ValueError("rolling_window must be positive.")

    validate_generated_output_path(spec.output_path)
    if spec.output_path.exists() and not spec.overwrite:
        raise FileExistsError(f"{spec.output_path} already exists. Pass overwrite=True to replace it.")

    market = _read_headline_csv(spec.market_headlines_path, "market_headlines_path")
    criticism = _read_headline_csv(spec.criticism_headlines_path, "criticism_headlines_path")
    labels = _read_labels(spec.labels_path) if spec.labels_path is not None else None

    market_counts = _daily_counts(market)
    criticism = _apply_labels_to_candidates(criticism, labels)
    criticism_counts = _daily_counts(criticism[criticism["_counts_as_criticism"]])
    zscore_column = _zscore_column(spec.rolling_window)

    output = pd.DataFrame(
        {
            "date": market_counts.index.astype(str),
            "total_market_article_count": market_counts.to_numpy(dtype=int),
        }
    )
    output["raw_criticism_count"] = (
        output["date"].map(criticism_counts).fillna(0).astype(int)
        if not output.empty
        else pd.Series(dtype="int64")
    )
    _validate_numerator_not_over_denominator(output)
    output["MCI"] = _ratio_series(output["raw_criticism_count"], output["total_market_article_count"])
    output[zscore_column] = _rolling_zscore(output["MCI"], spec.rolling_window)

    category_columns = _category_counts(criticism, output)
    if category_columns:
        for column_name, values in category_columns.items():
            output[column_name] = values

    output = _ordered_output_columns(output, zscore_column)
    spec.output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(spec.output_path, index=False, na_rep="")
    return spec.output_path


def build_category_indices(spec: IndexConstructionSpec) -> Path:
    """Build daily MCI variables, including category columns when labels permit."""

    return build_daily_mci(spec)


def _read_headline_csv(path: Path, path_name: str) -> pd.DataFrame:
    data = pd.read_csv(path, dtype=str, keep_default_na=False)
    data["_mci_date"] = _derive_date_series(data, path_name)
    _validate_parseable_dates(data, path_name)
    return data.copy()


def _derive_date_series(data: pd.DataFrame, path_name: str) -> pd.Series:
    if not any(column in data.columns for column in DATE_SOURCE_COLUMNS):
        raise ValueError(
            f"{path_name} must contain at least one date column: {', '.join(DATE_SOURCE_COLUMNS)}."
        )

    values = pd.Series("", index=data.index, dtype="object")
    for column in DATE_SOURCE_COLUMNS:
        if column not in data.columns:
            continue
        column_values = data[column].astype(str).str.strip()
        mask = (values == "") & (column_values != "")
        values.loc[mask] = column_values.loc[mask]

    parsed = pd.to_datetime(values.replace("", pd.NA), errors="coerce")
    return parsed.dt.strftime("%Y-%m-%d").fillna("")


def _chosen_date_source_values(data: pd.DataFrame) -> pd.Series:
    values = pd.Series("", index=data.index, dtype="object")
    for column in DATE_SOURCE_COLUMNS:
        if column not in data.columns:
            continue
        column_values = data[column].astype(str).str.strip()
        mask = (values == "") & (column_values != "")
        values.loc[mask] = column_values.loc[mask]
    return values


def _validate_parseable_dates(data: pd.DataFrame, path_name: str) -> None:
    invalid = data["_mci_date"] == ""
    if not invalid.any():
        return

    source_values = _chosen_date_source_values(data)
    first_index = invalid[invalid].index[0]
    first_row_number = int(first_index) + 2
    first_value = source_values.loc[first_index]
    raise ValueError(
        f"{path_name} has {int(invalid.sum())} row(s) without a parseable date; "
        f"first bad CSV row {first_row_number} value {first_value!r}."
    )


def _daily_counts(data: pd.DataFrame) -> pd.Series:
    if data.empty:
        return pd.Series(dtype="int64")
    return data.groupby("_mci_date", sort=True).size().astype(int)


def _apply_labels_to_candidates(candidates: pd.DataFrame, labels: pd.DataFrame | None) -> pd.DataFrame:
    labelled = candidates.copy()
    labelled["_counts_as_criticism"] = True
    labelled["_matched_category"] = ""
    if labels is None or candidates.empty:
        return labelled

    label_by_url, label_by_key = _label_lookup_tables(labels)
    counts: list[bool] = []
    categories: list[str] = []
    for _, row in labelled.iterrows():
        label = _match_label(row, label_by_url, label_by_key)
        if label is None:
            counts.append(True)
            categories.append("")
            continue

        label_value, category = label
        if label_value not in {0, 1}:
            raise ValueError(
                "Matched labels must have criticism_label 0 or 1 before MCI construction; "
                f"found invalid value for candidate {_candidate_identifier(row)!r}."
            )

        counts.append(label_value == 1)
        categories.append(category if label_value == 1 and category in CRITICISM_CATEGORIES else "")

    labelled["_counts_as_criticism"] = counts
    labelled["_matched_category"] = categories
    return labelled


def _read_labels(labels_path: Path) -> pd.DataFrame:
    paths = _label_paths(labels_path)
    frames = [pd.read_csv(path, dtype=str, keep_default_na=False) for path in paths]
    if not frames:
        return pd.DataFrame()
    labels = pd.concat(frames, ignore_index=True)
    labels["_mci_date"] = _derive_date_series(labels, "labels_path")
    _validate_parseable_dates(labels, "labels_path")
    return labels


def _label_paths(labels_path: Path) -> list[Path]:
    if labels_path.is_dir():
        paths = sorted(labels_path.glob("*.csv"))
        if not paths:
            raise ValueError(f"No labelled annotation CSVs found in {labels_path}.")
        return paths
    return [labels_path]


def _label_lookup_tables(
    labels: pd.DataFrame,
) -> tuple[dict[str, tuple[int | None, str]], dict[tuple[str, str, str], tuple[int | None, str]]]:
    label_by_url: dict[str, tuple[int | None, str]] = {}
    label_by_key: dict[tuple[str, str, str], tuple[int | None, str]] = {}

    for _, row in labels.iterrows():
        label_value = _parse_binary_label(row.get("criticism_label", ""))
        category = str(row.get("category_label", "")).strip()
        value = (label_value, category)

        url = _normalise_url(row.get("url", ""))
        if url:
            _store_label(label_by_url, url, value, "url")

        key = _match_key(row)
        _store_label(label_by_key, key, value, "date/title/domain key")

    return label_by_url, label_by_key


def _store_label(
    lookup: dict[str, tuple[int | None, str]] | dict[tuple[str, str, str], tuple[int | None, str]],
    key: str | tuple[str, str, str],
    value: tuple[int | None, str],
    key_type: str,
) -> None:
    existing = lookup.get(key)
    if existing is not None and existing != value:
        raise ValueError(
            f"Conflicting labels for {key_type} {key!r}: {existing} vs {value}. "
            "Use a consensus-labelled CSV for MCI construction."
        )
    lookup[key] = value


def _match_label(
    row: pd.Series,
    label_by_url: dict[str, tuple[int | None, str]],
    label_by_key: dict[tuple[str, str, str], tuple[int | None, str]],
) -> tuple[int | None, str] | None:
    url = _normalise_url(row.get("url", ""))
    if url and url in label_by_url:
        return label_by_url[url]
    return label_by_key.get(_match_key(row))


def _match_key(row: pd.Series) -> tuple[str, str, str]:
    title = row.get("normalised_title", "") or row.get("title", "")
    return (
        str(row.get("_mci_date", "")).strip(),
        normalise_title(str(title)),
        str(row.get("domain", "")).strip().lower(),
    )


def _candidate_identifier(row: pd.Series) -> str:
    url = _normalise_url(row.get("url", ""))
    if url:
        return url
    return "|".join(_match_key(row))


def _parse_binary_label(value: object) -> int | None:
    text = "" if value is None else str(value).strip()
    if text in {"0", "1"}:
        return int(text)
    return None


def _normalise_url(value: object) -> str:
    return "" if value is None else str(value).strip()


def _ratio_series(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator_array = numerator.astype(float).to_numpy()
    denominator_array = denominator.astype(float).to_numpy()
    result = np.divide(
        numerator_array,
        denominator_array,
        out=np.full_like(numerator_array, np.nan, dtype=float),
        where=denominator_array != 0,
    )
    return pd.Series(result, index=numerator.index, dtype="float64")


def _validate_numerator_not_over_denominator(output: pd.DataFrame) -> None:
    if output.empty:
        return

    invalid = output["raw_criticism_count"] > output["total_market_article_count"]
    if not invalid.any():
        return

    first = output.loc[invalid].iloc[0]
    raise ValueError(
        "raw_criticism_count exceeds total_market_article_count for "
        f"{int(invalid.sum())} date(s); first date {first['date']}: "
        f"raw_criticism_count={int(first['raw_criticism_count'])}, "
        f"total_market_article_count={int(first['total_market_article_count'])}. "
        "Candidate rows must be a subset of all-market rows."
    )


def _rolling_zscore(values: pd.Series, rolling_window: int) -> pd.Series:
    rolling = values.rolling(window=rolling_window, min_periods=rolling_window)
    rolling_mean = rolling.mean()
    rolling_std = rolling.std()
    zscore = (values - rolling_mean) / rolling_std
    return zscore.mask((rolling_std == 0) | rolling_std.isna())


def _category_counts(criticism: pd.DataFrame, output: pd.DataFrame) -> dict[str, pd.Series]:
    if criticism.empty or "_matched_category" not in criticism.columns:
        return {}

    categorised = criticism[
        (criticism["_counts_as_criticism"])
        & (criticism["_matched_category"].isin(CRITICISM_CATEGORIES))
        & (criticism["_matched_category"] != "")
    ]
    if categorised.empty:
        return {}

    grouped = categorised.groupby(["_mci_date", "_matched_category"], sort=True).size().unstack(fill_value=0)
    columns: dict[str, pd.Series] = {}
    for category in CRITICISM_CATEGORIES:
        raw_column = f"raw_criticism_count_{category}"
        mci_column = f"MCI_{category}"
        counts = output["date"].map(grouped[category] if category in grouped.columns else {}).fillna(0).astype(int)
        columns[raw_column] = counts
        columns[mci_column] = _ratio_series(counts, output["total_market_article_count"])
    return columns


def _zscore_column(rolling_window: int) -> str:
    return f"mci_rolling_{rolling_window}d_zscore"


def _ordered_output_columns(output: pd.DataFrame, zscore_column: str) -> pd.DataFrame:
    always_output_columns = list(BASE_OUTPUT_COLUMNS) + [zscore_column]
    category_columns = [column for column in output.columns if column not in always_output_columns]
    return output[always_output_columns + category_columns]
