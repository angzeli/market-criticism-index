"""Annotation sample export and validation utilities."""

from __future__ import annotations

import csv
import itertools
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from mci.config import CRITICISM_CATEGORIES, CRITICISM_QUERY_TERMS, INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from mci.text_processing import normalise_title, parse_seendate_to_new_york, validate_generated_output_path

ANNOTATION_COLUMNS = (
    "id",
    "date",
    "title",
    "source",
    "domain",
    "url",
    "query_type",
    "sample_stratum",
    "market_relevant_label",
    "criticism_label",
    "category_label",
    "intensity_score",
    "annotator_notes",
)

MANUAL_LABEL_COLUMNS = (
    "market_relevant_label",
    "criticism_label",
    "category_label",
    "intensity_score",
    "annotator_notes",
)

LIKELY_CRITICISM = "likely_criticism"
GENERAL_MARKET = "general_market"
AMBIGUOUS = "ambiguous"

ANNOTATION_SAMPLE_DIR = INTERIM_DATA_DIR / "annotations" / "samples"
LABELLED_ANNOTATION_DIR = PROCESSED_DATA_DIR / "annotations" / "labelled"
REQUIRED_SOURCE_COLUMNS = frozenset(("title", "domain", "query_type"))
DATE_SOURCE_COLUMNS = frozenset(("published_date_ny", "date", "seendate"))


@dataclass(frozen=True)
class AnnotationSampleSpec:
    """Inputs and deterministic sampling settings for K annotation exports."""

    candidate_csv: Path
    all_market_csv: Path
    output_path: Path | None = None
    output_dir: Path = ANNOTATION_SAMPLE_DIR
    likely_criticism_count: int = 200
    general_market_count: int = 200
    ambiguous_count: int = 100
    seed: int = 1
    run_date: date | None = None
    overwrite: bool = False


@dataclass(frozen=True)
class AnnotationSampleResult:
    """Metadata from an annotation sample export."""

    output_path: Path
    counts: dict[str, int]
    shortfalls: dict[str, int]
    total_rows: int


@dataclass(frozen=True)
class AnnotationValidationReport:
    """Summary metrics for completed annotation CSVs."""

    total_rows: int
    predicted_positive: int
    actual_positive: int
    true_positive: int
    false_positive: int
    false_negative: int
    precision: float | None
    sample_recall: float | None
    category_counts: dict[str, int]
    intensity_distribution: dict[str, int]
    missing_labels: dict[str, int]
    invalid_labels: dict[str, int]
    agreement: dict[str, float | int] | None


def build_annotation_sample(spec: AnnotationSampleSpec) -> AnnotationSampleResult:
    """Export a deterministic mixed annotation sample for K."""

    output_path = _annotation_output_path(spec)
    validate_generated_output_path(output_path)
    if output_path.exists() and not spec.overwrite:
        raise FileExistsError(f"{output_path} already exists. Pass overwrite=True to replace it.")

    candidate_rows = _deduplicate_for_sampling(_read_source_csv_records(spec.candidate_csv))
    all_market_rows = _deduplicate_for_sampling(_read_source_csv_records(spec.all_market_csv))

    candidate_keys = {_dedupe_key(row) for row in candidate_rows}
    criticism_terms = tuple(normalise_title(term) for term in CRITICISM_QUERY_TERMS)

    likely_pool = candidate_rows
    ambiguous_pool = [
        row
        for row in all_market_rows
        if _dedupe_key(row) not in candidate_keys and _contains_criticism_term(row, criticism_terms)
    ]
    ambiguous_keys = {_dedupe_key(row) for row in ambiguous_pool}
    general_pool = [
        row
        for row in all_market_rows
        if _dedupe_key(row) not in candidate_keys and _dedupe_key(row) not in ambiguous_keys
    ]

    selected_likely, likely_shortfall = _sample_stratum(likely_pool, spec.likely_criticism_count, spec.seed, 11)
    selected_ambiguous, ambiguous_shortfall = _sample_stratum(ambiguous_pool, spec.ambiguous_count, spec.seed, 29)
    selected_general, general_shortfall = _sample_stratum(general_pool, spec.general_market_count, spec.seed, 47)

    selected = (
        [(LIKELY_CRITICISM, row) for row in selected_likely]
        + [(GENERAL_MARKET, row) for row in selected_general]
        + [(AMBIGUOUS, row) for row in selected_ambiguous]
    )
    output_rows = [_annotation_row(index, stratum, row) for index, (stratum, row) in enumerate(selected, start=1)]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv_records(output_path, output_rows, ANNOTATION_COLUMNS)

    counts = Counter(row["sample_stratum"] for row in output_rows)
    return AnnotationSampleResult(
        output_path=output_path,
        counts={stratum: counts.get(stratum, 0) for stratum in (LIKELY_CRITICISM, GENERAL_MARKET, AMBIGUOUS)},
        shortfalls={
            LIKELY_CRITICISM: likely_shortfall,
            GENERAL_MARKET: general_shortfall,
            AMBIGUOUS: ambiguous_shortfall,
        },
        total_rows=len(output_rows),
    )


def validate_annotations(
    labelled_dir: Path = LABELLED_ANNOTATION_DIR,
    *,
    paths: Sequence[Path] | None = None,
    allow_empty: bool = False,
) -> AnnotationValidationReport:
    """Validate one or more completed K annotation CSVs."""

    csv_paths = list(paths) if paths is not None else sorted(labelled_dir.glob("*.csv"))
    if not csv_paths and not allow_empty:
        raise ValueError(f"No labelled annotation CSVs found in {labelled_dir}.")
    rows = _load_labelled_rows(csv_paths)

    predicted_positive_rows = [row for row in rows if row.get("query_type", "") == "candidate_criticism"]
    actual_positive_rows = [row for row in rows if _binary_label(row.get("criticism_label")) == 1]

    true_positive = sum(
        1
        for row in rows
        if row.get("query_type", "") == "candidate_criticism" and _binary_label(row.get("criticism_label")) == 1
    )
    false_positive = len(predicted_positive_rows) - true_positive
    false_negative = len(actual_positive_rows) - true_positive
    precision = _safe_ratio(true_positive, len(predicted_positive_rows))
    sample_recall = _safe_ratio(true_positive, len(actual_positive_rows))

    category_counts = Counter(
        row.get("category_label", "").strip()
        for row in actual_positive_rows
        if row.get("category_label", "").strip()
    )
    intensity_distribution = Counter(
        row.get("intensity_score", "").strip()
        for row in actual_positive_rows
        if row.get("intensity_score", "").strip()
    )

    return AnnotationValidationReport(
        total_rows=len(rows),
        predicted_positive=len(predicted_positive_rows),
        actual_positive=len(actual_positive_rows),
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        precision=precision,
        sample_recall=sample_recall,
        category_counts=dict(sorted(category_counts.items())),
        intensity_distribution=dict(sorted(intensity_distribution.items())),
        missing_labels=_missing_label_counts(rows),
        invalid_labels=_invalid_label_counts(rows),
        agreement=_inter_annotator_agreement(rows),
    )


def format_validation_report(report: AnnotationValidationReport) -> str:
    """Format annotation validation metrics for CLI output."""

    lines = [
        "Annotation validation summary",
        f"Rows: {report.total_rows}",
        f"Predicted positives: {report.predicted_positive}",
        f"Actual positives: {report.actual_positive}",
        f"Precision: {_format_metric(report.precision)}",
        f"Sample recall: {_format_metric(report.sample_recall)}",
        f"True positives: {report.true_positive}",
        f"False positives: {report.false_positive}",
        f"False negatives: {report.false_negative}",
        f"Category counts: {_format_dict(report.category_counts)}",
        f"Intensity distribution: {_format_dict(report.intensity_distribution)}",
        f"Missing labels: {_format_dict(report.missing_labels)}",
        f"Invalid labels: {_format_dict(report.invalid_labels)}",
    ]
    if report.agreement is not None:
        lines.append(f"Inter-annotator agreement: {_format_dict(report.agreement)}")
    return "\n".join(lines)


def _annotation_output_path(spec: AnnotationSampleSpec) -> Path:
    if spec.output_path is not None:
        return spec.output_path
    run_date = spec.run_date or date.today()
    filename = f"annotation_sample_{run_date:%Y%m%d}_seed{spec.seed}.csv"
    return spec.output_dir / filename


def _deduplicate_for_sampling(rows: Iterable[Mapping[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str, str]] = set()
    deduplicated: list[dict[str, str]] = []
    for row in rows:
        copied = dict(row)
        key = _dedupe_key(copied)
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(copied)
    return sorted(deduplicated, key=_dedupe_key)


def _dedupe_key(row: Mapping[str, str]) -> tuple[str, str, str]:
    return (
        normalise_title(row.get("normalised_title") or row.get("title", "")),
        _derive_export_date(row),
        row.get("domain", "").strip().lower(),
    )


def _contains_criticism_term(row: Mapping[str, str], terms: Sequence[str]) -> bool:
    title = normalise_title(row.get("normalised_title") or row.get("title", ""))
    return any(term and term in title for term in terms)


def _sample_stratum(
    rows: Sequence[dict[str, str]],
    count: int,
    seed: int,
    salt: int,
) -> tuple[list[dict[str, str]], int]:
    if count < 0:
        raise ValueError("Stratum counts must be non-negative.")
    if len(rows) <= count:
        return list(rows), count - len(rows)

    rng = random.Random(seed + salt)
    selected = rng.sample(list(rows), count)
    return sorted(selected, key=_dedupe_key), 0


def _annotation_row(index: int, stratum: str, row: Mapping[str, str]) -> dict[str, str]:
    output = {
        "id": f"ann_{index:06d}",
        "date": _derive_export_date(row),
        "title": row.get("title", ""),
        "source": row.get("source", ""),
        "domain": row.get("domain", ""),
        "url": row.get("url", ""),
        "query_type": row.get("query_type", ""),
        "sample_stratum": stratum,
    }
    for column in MANUAL_LABEL_COLUMNS:
        output[column] = ""
    return output


def _derive_export_date(row: Mapping[str, str]) -> str:
    for column in ("published_date_ny", "date"):
        value = row.get(column, "").strip()
        if value:
            return value

    seen_at = parse_seendate_to_new_york(row.get("seendate", ""))
    if seen_at is not None:
        return seen_at.date().isoformat()
    return ""


def _load_labelled_rows(paths: Sequence[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        for row in _read_csv_records(path):
            copied = dict(row)
            copied["_annotation_file"] = str(path)
            rows.append(copied)
    return rows


def _read_source_csv_records(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = set(reader.fieldnames or [])
        _validate_source_fieldnames(path, fieldnames)
        rows = list(reader)

    rows_without_date = [index for index, row in enumerate(rows, start=2) if not _derive_export_date(row)]
    if rows_without_date:
        first_row = rows_without_date[0]
        raise ValueError(
            f"{path} has rows without a usable date source; first missing or unparsable date is on CSV row {first_row}."
        )
    return rows


def _validate_source_fieldnames(path: Path, fieldnames: set[str]) -> None:
    missing = sorted(REQUIRED_SOURCE_COLUMNS - fieldnames)
    has_date_source = bool(DATE_SOURCE_COLUMNS & fieldnames)
    problems: list[str] = []
    if missing:
        problems.append(f"missing required columns: {', '.join(missing)}")
    if not has_date_source:
        problems.append("missing one date source column: published_date_ny, date, or parseable seendate")
    if problems:
        raise ValueError(f"{path} is not a valid annotation source CSV: {'; '.join(problems)}.")


def _missing_label_counts(rows: Sequence[Mapping[str, str]]) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        for column in ("market_relevant_label", "criticism_label"):
            if not row.get(column, "").strip():
                counts[column] += 1
        if _binary_label(row.get("criticism_label")) == 1:
            for column in ("category_label", "intensity_score"):
                if not row.get(column, "").strip():
                    counts[column] += 1
    return {column: counts.get(column, 0) for column in ("market_relevant_label", "criticism_label", "category_label", "intensity_score")}


def _invalid_label_counts(rows: Sequence[Mapping[str, str]]) -> dict[str, int]:
    counts = Counter()
    valid_categories = set(CRITICISM_CATEGORIES)
    for duplicate_id in _duplicate_ids_within_files(rows).values():
        counts["duplicate_id_within_file"] += duplicate_id

    for row in rows:
        for column in ("market_relevant_label", "criticism_label"):
            value = row.get(column, "").strip()
            if value and value not in {"0", "1"}:
                counts[column] += 1

        criticism_label = _binary_label(row.get("criticism_label"))
        category = row.get("category_label", "").strip()
        if category and category not in valid_categories:
            counts["category_label"] += 1
        if category and criticism_label != 1:
            counts["category_label_consistency"] += 1

        intensity = row.get("intensity_score", "").strip()
        if intensity and not _valid_intensity(intensity):
            counts["intensity_score"] += 1
        if intensity and criticism_label != 1:
            counts["intensity_score_consistency"] += 1

    return {
        column: counts.get(column, 0)
        for column in (
            "market_relevant_label",
            "criticism_label",
            "category_label",
            "intensity_score",
            "category_label_consistency",
            "intensity_score_consistency",
            "duplicate_id_within_file",
        )
    }


def _inter_annotator_agreement(rows: Sequence[Mapping[str, str]]) -> dict[str, float | int] | None:
    labels_by_id: dict[str, dict[str, int]] = defaultdict(dict)
    for row in rows:
        label = _binary_label(row.get("criticism_label"))
        item_id = row.get("id", "").strip()
        annotation_file = row.get("_annotation_file", "").strip()
        if item_id and label in (0, 1):
            labels_by_id[item_id].setdefault(annotation_file, label)

    pairs: list[tuple[int, int]] = []
    for labels_by_file in labels_by_id.values():
        labels = list(labels_by_file.values())
        if len(labels) >= 2:
            pairs.extend(itertools.combinations(labels, 2))

    if not pairs:
        return None

    observed = sum(1 for left, right in pairs if left == right) / len(pairs)
    left_yes = sum(left for left, _ in pairs) / len(pairs)
    right_yes = sum(right for _, right in pairs) / len(pairs)
    expected = left_yes * right_yes + (1 - left_yes) * (1 - right_yes)
    kappa = 1.0 if expected == 1.0 else (observed - expected) / (1 - expected)
    return {
        "paired_items": len(pairs),
        "criticism_label_percent_agreement": observed,
        "criticism_label_cohens_kappa": kappa,
    }


def _duplicate_ids_within_files(rows: Sequence[Mapping[str, str]]) -> dict[tuple[str, str], int]:
    counts = Counter(
        (row.get("_annotation_file", "").strip(), row.get("id", "").strip())
        for row in rows
        if row.get("id", "").strip()
    )
    return {key: count - 1 for key, count in counts.items() if count > 1}


def _binary_label(value: object) -> int | None:
    text = "" if value is None else str(value).strip()
    if text in {"0", "1"}:
        return int(text)
    return None


def _valid_intensity(value: str) -> bool:
    try:
        parsed = float(value)
    except ValueError:
        return False
    return 0 <= parsed <= 3


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _format_dict(values: Mapping[str, object]) -> str:
    if not values:
        return "{}"
    return ", ".join(f"{key}={value}" for key, value in values.items())


def _read_csv_records(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def _write_csv_records(path: Path, rows: Sequence[Mapping[str, str]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
