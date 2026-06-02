"""Headline text normalisation, deduplication, and trading-day assignment."""

from __future__ import annotations

import csv
import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

from mci.config import RAW_DATA_DIR

NEW_YORK_TZ = ZoneInfo("America/New_York")
MARKET_CLOSE_NY = time(16, 0)

NORMALISED_TITLE_FIELD = "normalised_title"
SEENDATE_NY_FIELD = "seendate_ny"
PUBLISHED_DATE_NY_FIELD = "published_date_ny"
TRADING_DAY_FIELD = "trading_day"
DERIVED_FIELDS = (
    NORMALISED_TITLE_FIELD,
    SEENDATE_NY_FIELD,
    PUBLISHED_DATE_NY_FIELD,
    TRADING_DAY_FIELD,
)
SEPARATOR_PUNCTUATION = frozenset("-‐‑‒–—―/\\_")


@dataclass(frozen=True)
class TextProcessingSpec:
    """Input and output paths for deterministic text-processing steps."""

    input_path: Path
    output_path: Path
    text_column: str = "title"


def normalise_text(text: str) -> str:
    """Return a normalised version of text for matching and deduplication."""

    lowered = unicodedata.normalize("NFKC", str(text)).lower()
    normalised_characters: list[str] = []
    for character in lowered:
        if character in SEPARATOR_PUNCTUATION:
            normalised_characters.append(" ")
        elif unicodedata.category(character).startswith("P"):
            continue
        else:
            normalised_characters.append(character)
    return re.sub(r"\s+", " ", "".join(normalised_characters)).strip()


def normalise_title(title: str) -> str:
    """Return a normalised headline title."""

    return normalise_text(title)


def parse_seendate_to_new_york(value: object) -> datetime | None:
    """Parse a timestamp and convert it to New York time when possible.

    Naive timestamps are treated as UTC because GDELT `seendate` values are
    UTC-like publication-observation timestamps.
    """

    parsed = _parse_datetime(value)
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(NEW_YORK_TZ)


def assign_trading_day(
    value: object,
    *,
    trading_days: Sequence[date] | None = None,
    market_close: time = MARKET_CLOSE_NY,
) -> date | None:
    """Assign a timestamp to a trading day using the 16:00 New York cutoff."""

    target = headline_target_date(value, market_close=market_close)
    if target is None:
        return None

    return _first_trading_day_on_or_after(target, trading_days)


def headline_target_date(
    value: object,
    *,
    market_close: time = MARKET_CLOSE_NY,
) -> date | None:
    """Return the pre-calendar target date after applying the NY close cutoff."""

    seen_at = parse_seendate_to_new_york(value)
    if seen_at is None:
        return None

    target = seen_at.date()
    if seen_at.time() > market_close:
        target += timedelta(days=1)
    return target


def clean_headline_record(
    record: Mapping[str, object],
    *,
    text_column: str = "title",
    timestamp_column: str = "seendate",
    trading_days: Sequence[date] | None = None,
) -> dict[str, object]:
    """Return a cleaned copy of one headline record without mutating input."""

    cleaned = dict(record)
    title = cleaned.get(text_column, cleaned.get("headline", ""))
    seen_at = parse_seendate_to_new_york(cleaned.get(timestamp_column))
    trading_day = assign_trading_day(cleaned.get(timestamp_column), trading_days=trading_days)

    cleaned[NORMALISED_TITLE_FIELD] = normalise_title(str(title))
    cleaned[SEENDATE_NY_FIELD] = seen_at.isoformat() if seen_at is not None else ""
    cleaned[PUBLISHED_DATE_NY_FIELD] = seen_at.date().isoformat() if seen_at is not None else ""
    cleaned[TRADING_DAY_FIELD] = trading_day.isoformat() if trading_day is not None else ""
    return cleaned


def clean_headline_records(
    records: Iterable[Mapping[str, object]],
    *,
    text_column: str = "title",
    timestamp_column: str = "seendate",
    trading_days: Sequence[date] | None = None,
) -> list[dict[str, object]]:
    """Return cleaned copies of headline records without mutating input."""

    return [
        clean_headline_record(
            record,
            text_column=text_column,
            timestamp_column=timestamp_column,
            trading_days=trading_days,
        )
        for record in records
    ]


def validate_trading_calendar_coverage(
    records: Iterable[Mapping[str, object]],
    trading_days: Sequence[date],
    *,
    timestamp_column: str = "seendate",
) -> None:
    """Validate that parseable headline target dates are covered by a calendar."""

    ordered_days = sorted(set(trading_days))
    if not ordered_days:
        raise ValueError("Trading calendar must contain at least one date.")

    min_date = ordered_days[0]
    max_date = ordered_days[-1]
    for record in records:
        target = headline_target_date(record.get(timestamp_column))
        if target is None:
            continue
        if target < min_date or target > max_date:
            raise _calendar_coverage_error(target, min_date, max_date)


def deduplicate_headlines(
    records: Iterable[Mapping[str, object]],
    *,
    text_column: str = "title",
    timestamp_column: str = "seendate",
    domain_column: str = "domain",
    trading_days: Sequence[date] | None = None,
) -> list[dict[str, object]]:
    """Deduplicate by NY publication date, normalised title, and domain.

    The returned records are cleaned copies; input mappings are not modified.
    """

    seen: set[tuple[str, str, str]] = set()
    deduplicated: list[dict[str, object]] = []

    for record in records:
        cleaned = clean_headline_record(
            record,
            text_column=text_column,
            timestamp_column=timestamp_column,
            trading_days=trading_days,
        )
        key = (
            _dedupe_date(cleaned),
            str(cleaned.get(NORMALISED_TITLE_FIELD, "")),
            str(cleaned.get(domain_column, "")).strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(cleaned)

    return deduplicated


def fuzzy_deduplicate_headlines(
    records: Iterable[Mapping[str, object]],
    *,
    threshold: float = 0.96,
    text_column: str = "title",
    timestamp_column: str = "seendate",
    domain_column: str = "domain",
    trading_days: Sequence[date] | None = None,
) -> list[dict[str, object]]:
    """Optionally remove near-duplicate titles within each date/domain group."""

    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")

    exact_records = deduplicate_headlines(
        records,
        text_column=text_column,
        timestamp_column=timestamp_column,
        domain_column=domain_column,
        trading_days=trading_days,
    )
    kept: list[dict[str, object]] = []

    for record in exact_records:
        group_key = (_dedupe_date(record), str(record.get(domain_column, "")).strip().lower())
        title = str(record.get(NORMALISED_TITLE_FIELD, ""))
        has_near_duplicate = any(
            group_key == (_dedupe_date(existing), str(existing.get(domain_column, "")).strip().lower())
            and SequenceMatcher(None, title, str(existing.get(NORMALISED_TITLE_FIELD, ""))).ratio() >= threshold
            for existing in kept
        )
        if not has_near_duplicate:
            kept.append(record)

    return kept


def clean_headline_file(spec: TextProcessingSpec, *, fuzzy: bool = False) -> Path:
    """Clean and deduplicate a CSV file, writing a new CSV at `spec.output_path`."""

    validate_generated_output_path(spec.output_path)
    records, input_fieldnames = _read_csv_records_with_fieldnames(spec.input_path)
    if fuzzy:
        cleaned = fuzzy_deduplicate_headlines(records, text_column=spec.text_column)
    else:
        cleaned = deduplicate_headlines(records, text_column=spec.text_column)

    spec.output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv_records(spec.output_path, cleaned, fieldnames=input_fieldnames)
    return spec.output_path


def validate_generated_output_path(output_path: Path) -> None:
    """Reject generated-output paths under the raw-data directory."""

    resolved_output = output_path.expanduser().resolve(strict=False)
    resolved_raw = RAW_DATA_DIR.expanduser().resolve(strict=False)
    if resolved_output == resolved_raw or resolved_output.is_relative_to(resolved_raw):
        raise ValueError(f"Generated outputs may not be written under raw data directory: {RAW_DATA_DIR}")


def build_annotation_sample(spec: TextProcessingSpec) -> Path:
    """Export a mixed candidate and non-candidate sample for K's annotation."""

    raise NotImplementedError("Annotation sample export is not implemented in the scaffold.")


def _parse_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value

    text = str(value).strip()
    if not text:
        return None

    if _is_date_only_text(text):
        return None

    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"

    for parser in (_parse_gdelt_datetime, _parse_iso_datetime):
        parsed = parser(text)
        if parsed is not None:
            return parsed

    return None


def _parse_gdelt_datetime(text: str) -> datetime | None:
    if not re.fullmatch(r"\d{14}", text):
        return None
    try:
        return datetime.strptime(text, "%Y%m%d%H%M%S")
    except ValueError:
        return None


def _is_date_only_text(text: str) -> bool:
    return bool(re.fullmatch(r"\d{8}", text) or re.fullmatch(r"\d{4}-\d{2}-\d{2}", text))


def _has_iso_time_component(text: str) -> bool:
    return "T" in text or " " in text


def _parse_iso_datetime(text: str) -> datetime | None:
    if not _has_iso_time_component(text):
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if isinstance(parsed, datetime):
        return parsed
    return None


def _first_trading_day_on_or_after(target: date, trading_days: Sequence[date] | None) -> date:
    if trading_days is not None:
        ordered_days = sorted(set(trading_days))
        if not ordered_days:
            raise ValueError("Trading calendar must contain at least one date.")
        min_date = ordered_days[0]
        max_date = ordered_days[-1]
        if target < min_date or target > max_date:
            raise _calendar_coverage_error(target, min_date, max_date)
        for trading_day in ordered_days:
            if trading_day >= target:
                return trading_day
        raise _calendar_coverage_error(target, min_date, max_date)

    current = target
    while current.weekday() >= 5:
        current += timedelta(days=1)
    return current


def _dedupe_date(record: Mapping[str, object]) -> str:
    for field in (PUBLISHED_DATE_NY_FIELD, "date", TRADING_DAY_FIELD):
        value = record.get(field)
        if value:
            return str(value)
    return ""


def _calendar_coverage_error(target: date, min_date: date, max_date: date) -> ValueError:
    return ValueError(
        "Trading calendar does not cover target date "
        f"{target.isoformat()}; calendar range is {min_date.isoformat()} to {max_date.isoformat()}."
    )


def _read_csv_records(path: Path) -> list[dict[str, str]]:
    records, _ = _read_csv_records_with_fieldnames(path)
    return records


def _read_csv_records_with_fieldnames(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        return list(reader), list(reader.fieldnames or [])


def _write_csv_records(
    path: Path,
    records: Sequence[Mapping[str, object]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> None:
    output_fieldnames = _fieldnames(records, base_fieldnames=fieldnames)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=output_fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _fieldnames(
    records: Sequence[Mapping[str, object]],
    *,
    base_fieldnames: Sequence[str] | None = None,
) -> list[str]:
    fieldnames: list[str] = list(base_fieldnames or [])
    for record in records:
        for field in record:
            if field not in fieldnames:
                fieldnames.append(str(field))
    for field in DERIVED_FIELDS:
        if field not in fieldnames:
            fieldnames.append(field)
    return fieldnames
