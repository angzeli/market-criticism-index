"""Tests for K annotation sample export and validation."""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import pytest

from mci.annotations import (
    AMBIGUOUS,
    ANNOTATION_COLUMNS,
    GENERAL_MARKET,
    LIKELY_CRITICISM,
    AnnotationSampleSpec,
    build_annotation_sample,
    format_validation_report,
    validate_annotations,
)
from mci.config import RAW_DATA_DIR


def test_build_annotation_sample_exports_deterministic_columns_strata_and_blank_labels(tmp_path: Path) -> None:
    candidate_path, all_market_path = _write_sample_source_csvs(tmp_path)
    output_path = tmp_path / "samples" / "annotation_sample.csv"
    spec = AnnotationSampleSpec(
        candidate_csv=candidate_path,
        all_market_csv=all_market_path,
        output_path=output_path,
        likely_criticism_count=5,
        general_market_count=5,
        ambiguous_count=2,
        seed=7,
    )

    result = build_annotation_sample(spec)
    rows = _read_csv(output_path)

    assert result.output_path == output_path
    assert result.total_rows == 5
    assert result.counts == {
        LIKELY_CRITICISM: 2,
        GENERAL_MARKET: 2,
        AMBIGUOUS: 1,
    }
    assert result.shortfalls == {
        LIKELY_CRITICISM: 3,
        GENERAL_MARKET: 3,
        AMBIGUOUS: 1,
    }
    assert list(rows[0].keys()) == list(ANNOTATION_COLUMNS)
    assert [row["id"] for row in rows] == [f"ann_{index:06d}" for index in range(1, 6)]
    assert {row["sample_stratum"] for row in rows} == {LIKELY_CRITICISM, GENERAL_MARKET, AMBIGUOUS}
    assert all(row["market_relevant_label"] == "" for row in rows)
    assert all(row["criticism_label"] == "" for row in rows)
    assert all(row["category_label"] == "" for row in rows)
    assert all(row["intensity_score"] == "" for row in rows)
    assert all(row["annotator_notes"] == "" for row in rows)


def test_build_annotation_sample_default_filename_uses_run_date_and_seed(tmp_path: Path) -> None:
    candidate_path, all_market_path = _write_sample_source_csvs(tmp_path)
    spec = AnnotationSampleSpec(
        candidate_csv=candidate_path,
        all_market_csv=all_market_path,
        output_dir=tmp_path / "samples",
        likely_criticism_count=1,
        general_market_count=1,
        ambiguous_count=1,
        seed=42,
        run_date=date(2026, 6, 2),
    )

    result = build_annotation_sample(spec)

    assert result.output_path.name == "annotation_sample_20260602_seed42.csv"


def test_build_annotation_sample_refuses_raw_output_path(tmp_path: Path) -> None:
    candidate_path, all_market_path = _write_sample_source_csvs(tmp_path)
    spec = AnnotationSampleSpec(
        candidate_csv=candidate_path,
        all_market_csv=all_market_path,
        output_path=RAW_DATA_DIR / "bad_annotation_sample.csv",
    )

    with pytest.raises(ValueError, match="raw data directory"):
        build_annotation_sample(spec)


def test_build_annotation_sample_requires_source_provenance_columns(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidate.csv"
    all_market_path = tmp_path / "all_market.csv"
    _write_csv(candidate_path, [{"title": "Wall Street warning", "domain": "example.com"}], ("title", "domain"))
    _write_csv(all_market_path, [{"title": "Wall Street rally", "domain": "example.com"}], ("title", "domain"))

    spec = AnnotationSampleSpec(
        candidate_csv=candidate_path,
        all_market_csv=all_market_path,
        output_path=tmp_path / "sample.csv",
    )

    with pytest.raises(ValueError, match="missing required columns: query_type"):
        build_annotation_sample(spec)


def test_build_annotation_sample_requires_usable_date_source(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidate.csv"
    all_market_path = tmp_path / "all_market.csv"
    fieldnames = ("title", "domain", "query_type", "seendate")
    _write_csv(
        candidate_path,
        [{"title": "Wall Street warning", "domain": "example.com", "query_type": "candidate_criticism", "seendate": "20240105"}],
        fieldnames,
    )
    _write_csv(
        all_market_path,
        [{"title": "Wall Street rally", "domain": "example.com", "query_type": "all_us_market", "seendate": "20240105"}],
        fieldnames,
    )

    spec = AnnotationSampleSpec(
        candidate_csv=candidate_path,
        all_market_csv=all_market_path,
        output_path=tmp_path / "sample.csv",
    )

    with pytest.raises(ValueError, match="without a usable date source"):
        build_annotation_sample(spec)


def test_validate_annotations_reports_candidate_metrics_and_label_quality(tmp_path: Path) -> None:
    labelled_path = tmp_path / "labelled.csv"
    _write_csv(
        labelled_path,
        [
            _labelled_row("ann_000001", "candidate_criticism", "1", "1", "valuation", "2"),
            _labelled_row("ann_000002", "candidate_criticism", "1", "0", "", ""),
            _labelled_row("ann_000003", "all_us_market", "1", "1", "bubble_speculation", "3"),
            _labelled_row("ann_000004", "all_us_market", "", "", "", ""),
        ],
        ANNOTATION_COLUMNS,
    )

    report = validate_annotations(paths=[labelled_path])

    assert report.total_rows == 4
    assert report.predicted_positive == 2
    assert report.actual_positive == 2
    assert report.true_positive == 1
    assert report.false_positive == 1
    assert report.false_negative == 1
    assert report.precision == 0.5
    assert report.sample_recall == 0.5
    assert report.category_counts == {"bubble_speculation": 1, "valuation": 1}
    assert report.intensity_distribution == {"2": 1, "3": 1}
    assert report.missing_labels == {
        "market_relevant_label": 1,
        "criticism_label": 1,
        "category_label": 0,
        "intensity_score": 0,
    }
    assert report.invalid_labels == {
        "market_relevant_label": 0,
        "criticism_label": 0,
        "category_label": 0,
        "intensity_score": 0,
        "category_label_consistency": 0,
        "intensity_score_consistency": 0,
        "duplicate_id_within_file": 0,
    }


def test_validate_annotations_flags_label_consistency_errors(tmp_path: Path) -> None:
    labelled_path = tmp_path / "labelled.csv"
    _write_csv(
        labelled_path,
        [
            _labelled_row("ann_000001", "all_us_market", "1", "0", "valuation", "2"),
            _labelled_row("ann_000002", "all_us_market", "1", "", "concentration", "1"),
        ],
        ANNOTATION_COLUMNS,
    )

    report = validate_annotations(paths=[labelled_path])

    assert report.invalid_labels["category_label_consistency"] == 2
    assert report.invalid_labels["intensity_score_consistency"] == 2


def test_validate_annotations_flags_duplicate_ids_within_one_file_and_excludes_them_from_agreement(tmp_path: Path) -> None:
    labelled_path = tmp_path / "labelled.csv"
    _write_csv(
        labelled_path,
        [
            _labelled_row("ann_000001", "candidate_criticism", "1", "1", "valuation", "2"),
            _labelled_row("ann_000001", "candidate_criticism", "1", "0", "", ""),
        ],
        ANNOTATION_COLUMNS,
    )

    report = validate_annotations(paths=[labelled_path])

    assert report.invalid_labels["duplicate_id_within_file"] == 1
    assert report.agreement is None


def test_validate_annotations_reports_optional_inter_annotator_agreement(tmp_path: Path) -> None:
    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"
    _write_csv(
        first_path,
        [
            _labelled_row("ann_000001", "candidate_criticism", "1", "1", "valuation", "2"),
            _labelled_row("ann_000002", "all_us_market", "1", "0", "", ""),
        ],
        ANNOTATION_COLUMNS,
    )
    _write_csv(
        second_path,
        [
            _labelled_row("ann_000001", "candidate_criticism", "1", "1", "valuation", "2"),
            _labelled_row("ann_000002", "all_us_market", "1", "1", "concentration", "1"),
        ],
        ANNOTATION_COLUMNS,
    )

    report = validate_annotations(paths=[first_path, second_path])

    assert report.agreement is not None
    assert report.agreement["paired_items"] == 2
    assert report.agreement["criticism_label_percent_agreement"] == 0.5
    assert report.agreement["criticism_label_cohens_kappa"] == 0.0


def test_validate_annotations_raises_on_empty_inputs_unless_allowed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No labelled annotation CSVs found"):
        validate_annotations(labelled_dir=tmp_path)

    report = validate_annotations(labelled_dir=tmp_path, allow_empty=True)

    assert report.total_rows == 0
    assert "Rows: 0" in format_validation_report(report)


def _write_sample_source_csvs(tmp_path: Path) -> tuple[Path, Path]:
    candidate_path = tmp_path / "candidate.csv"
    all_market_path = tmp_path / "all_market.csv"
    source_columns = (
        "title",
        "source",
        "domain",
        "url",
        "query_type",
        "published_date_ny",
        "normalised_title",
        "seendate",
    )
    _write_csv(
        candidate_path,
        [
            _source_row("Wall Street bubble warning", "A", "a.com", "candidate_criticism", "2024-01-01"),
            _source_row("AI bubble concerns", "B", "b.com", "candidate_criticism", "2024-01-02"),
            _source_row("Wall Street bubble warning!", "A", "a.com", "candidate_criticism", "2024-01-01"),
        ],
        source_columns,
    )
    _write_csv(
        all_market_path,
        [
            _source_row("Wall Street bubble warning", "A", "a.com", "all_us_market", "2024-01-01"),
            _source_row("S&P 500 closes higher", "C", "c.com", "all_us_market", "2024-01-01"),
            _source_row("Nasdaq correction warning grows", "D", "d.com", "all_us_market", "2024-01-02"),
            _source_row("Wall Street rally broadens", "E", "e.com", "all_us_market", "2024-01-03"),
            _source_row("Wall Street rally broadens.", "E", "e.com", "all_us_market", "2024-01-03"),
        ],
        source_columns,
    )
    return candidate_path, all_market_path


def _source_row(title: str, source: str, domain: str, query_type: str, published_date: str) -> dict[str, str]:
    return {
        "title": title,
        "source": source,
        "domain": domain,
        "url": f"https://{domain}/story",
        "query_type": query_type,
        "published_date_ny": published_date,
        "normalised_title": "",
        "seendate": "",
    }


def _labelled_row(
    item_id: str,
    query_type: str,
    market_label: str,
    criticism_label: str,
    category: str,
    intensity: str,
) -> dict[str, str]:
    return {
        "id": item_id,
        "date": "2024-01-01",
        "title": "Example headline",
        "source": "Example",
        "domain": "example.com",
        "url": "https://example.com/story",
        "query_type": query_type,
        "sample_stratum": LIKELY_CRITICISM if query_type == "candidate_criticism" else GENERAL_MARKET,
        "market_relevant_label": market_label,
        "criticism_label": criticism_label,
        "category_label": category,
        "intensity_score": intensity,
        "annotator_notes": "",
    }


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))
