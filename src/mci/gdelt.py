"""GDELT DOC 2.0 headline collection utilities."""

from __future__ import annotations

import csv
from email.utils import parsedate_to_datetime
import io
import json
import random
import time
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import pandas as pd
import requests

from mci.config import CRITICISM_QUERY_TERMS, INTERIM_DATA_DIR, MARKET_QUERY_TERMS, RAW_DATA_DIR
from mci.text_processing import normalise_title

GDELT_DOC_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_V2_MASTERFILELIST = "https://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
GDELT_ARTICLE_FIELDS = ("title", "url", "domain", "source", "seendate", "language", "query_type")
DEFAULT_GDELT_SOURCE_LANGUAGE = "english"


class GdeltClientError(RuntimeError):
    """Raised when GDELT collection cannot complete cleanly."""

    def __init__(
        self,
        message: str,
        *,
        resume_guidance: str | None = None,
        completed_days: int | None = None,
        next_missing_day: date | None = None,
    ) -> None:
        super().__init__(message)
        self.resume_guidance = resume_guidance
        self.completed_days = completed_days
        self.next_missing_day = next_missing_day


class GdeltQueryType(str, Enum):
    """Supported headline query types for the MVP."""

    ALL_MARKET = "all_us_market"
    CANDIDATE_CRITICISM = "candidate_criticism"


@dataclass(frozen=True)
class GdeltRequestSpec:
    """Parameters for a deterministic GDELT collection run."""

    query_type: GdeltQueryType
    start_date: date
    end_date: date
    market_terms: Sequence[str] = MARKET_QUERY_TERMS
    criticism_terms: Sequence[str] = CRITICISM_QUERY_TERMS
    source_language: str | None = DEFAULT_GDELT_SOURCE_LANGUAGE
    raw_dir: Path = RAW_DATA_DIR / "gdelt"
    interim_dir: Path = INTERIM_DATA_DIR
    max_records: int = 250
    overwrite: bool = False


@dataclass(frozen=True)
class GdeltCollectionResult:
    """Output paths and record count from a GDELT collection run."""

    raw_path: Path
    interim_path: Path
    article_count: int


@dataclass(frozen=True)
class GdeltLongRunSpec:
    """Parameters for resumable multi-day GDELT DOC ArticleList collection."""

    start_date: date
    end_date: date
    query_types: Sequence[GdeltQueryType] = (
        GdeltQueryType.ALL_MARKET,
        GdeltQueryType.CANDIDATE_CRITICISM,
    )
    market_terms: Sequence[str] = MARKET_QUERY_TERMS
    criticism_terms: Sequence[str] = CRITICISM_QUERY_TERMS
    source_language: str | None = DEFAULT_GDELT_SOURCE_LANGUAGE
    raw_dir: Path = RAW_DATA_DIR / "gdelt"
    interim_dir: Path = INTERIM_DATA_DIR
    max_records: int = 250
    resume: bool = True
    chunking_mode: str = "daily"
    overwrite_interim: bool = False
    request_pause_seconds: float = 10.0


@dataclass(frozen=True)
class GdeltLongRunResult:
    """Output metadata from a resumable GDELT long-run collection."""

    raw_paths: dict[str, Path]
    interim_paths: dict[str, Path]
    article_counts: dict[str, int]
    completed_days: int
    skipped_existing_checkpoints: int


@dataclass(frozen=True)
class GkgBulkSpec:
    """Parameters for optional filtered GKG historical extraction."""

    start_date: date
    end_date: date
    filter_terms: Sequence[str]
    raw_dir: Path = RAW_DATA_DIR / "gdelt" / "gkg_filtered"
    masterfilelist_url: str = GDELT_V2_MASTERFILELIST
    overwrite: bool = False
    keep_archives: bool = False
    max_archives: int | None = None


@dataclass(frozen=True)
class GkgBulkResult:
    """Output metadata from a filtered GKG archive extraction."""

    extract_path: Path
    manifest_path: Path
    archive_count: int
    matched_row_count: int


@dataclass(frozen=True)
class _LongRunQueryResult:
    raw_path: Path
    interim_path: Path
    article_count: int
    skipped_existing_checkpoints: int


class GdeltClient:
    """Small GDELT DOC 2.0 client with retry handling."""

    def __init__(
        self,
        *,
        endpoint: str = GDELT_DOC_ENDPOINT,
        session: Any | None = None,
        timeout: int = 30,
        max_retries: int = 10,
        backoff_seconds: float = 30.0,
        max_backoff_seconds: float = 1800.0,
        jitter_seconds: float = 10.0,
        request_pause_seconds: float = 10.0,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[float, float], float] | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.session = session or requests.Session()
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.max_backoff_seconds = max_backoff_seconds
        self.jitter_seconds = jitter_seconds
        self.request_pause_seconds = request_pause_seconds
        self.sleep = sleep
        self.jitter = jitter or random.uniform

        headers = getattr(self.session, "headers", None)
        if headers is not None:
            headers.setdefault("User-Agent", "market-criticism-index/0.1.0 research")

    def fetch_day(
        self,
        query: str,
        day: date,
        max_records: int,
        *,
        progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Fetch one day of article-list results from GDELT."""

        params = {
            "query": query,
            "mode": "ArtList",
            "format": "json",
            "startdatetime": f"{day:%Y%m%d}000000",
            "enddatetime": f"{day:%Y%m%d}235959",
            "maxrecords": str(max_records),
            "sort": "datedesc",
        }
        return self._get_json(params, progress_callback=progress_callback, context=context)

    def _get_json(
        self,
        params: Mapping[str, str],
        *,
        progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        event_context = dict(context or {})

        for attempt in range(self.max_retries + 1):
            self._emit_progress(progress_callback, event_context, phase="request", attempt=attempt + 1)
            try:
                response = self.session.get(self.endpoint, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                last_error = exc
                if attempt < self.max_retries:
                    sleep_seconds = self._retry_delay(attempt)
                    self._emit_progress(
                        progress_callback,
                        event_context,
                        phase="retry",
                        attempt=attempt + 1,
                        error=str(exc),
                        sleep_seconds=sleep_seconds,
                    )
                    self._pause(sleep_seconds)
                    continue
                self._emit_progress(progress_callback, event_context, phase="fatal_failure", error=str(exc))
                raise GdeltClientError(
                    "GDELT request failed after retries. Please check the network connection "
                    "or try a smaller date range."
                ) from exc

            status_code = getattr(response, "status_code", 0)
            if status_code == 429 or status_code >= 500:
                if attempt < self.max_retries:
                    sleep_seconds = self._retry_delay(attempt, response=response)
                    self._emit_progress(
                        progress_callback,
                        event_context,
                        phase="retry",
                        attempt=attempt + 1,
                        status_code=status_code,
                        sleep_seconds=sleep_seconds,
                    )
                    self._pause(sleep_seconds)
                    continue
                self._emit_progress(
                    progress_callback,
                    event_context,
                    phase="fatal_failure",
                    status_code=status_code,
                )
                raise GdeltClientError(
                    f"GDELT returned HTTP status {status_code} after retries. "
                    "Please wait and try again later."
                )

            if status_code >= 400:
                message = _response_text(response)
                self._emit_progress(
                    progress_callback,
                    event_context,
                    phase="fatal_failure",
                    status_code=status_code,
                    error=message,
                )
                raise GdeltClientError(f"GDELT returned HTTP status {status_code}: {message}")

            try:
                data = response.json()
            except ValueError as exc:
                self._emit_progress(
                    progress_callback,
                    event_context,
                    phase="fatal_failure",
                    error="invalid json",
                )
                raise GdeltClientError("GDELT returned a response that was not valid JSON.") from exc

            if not isinstance(data, dict):
                self._emit_progress(
                    progress_callback,
                    event_context,
                    phase="fatal_failure",
                    error="unexpected json",
                )
                raise GdeltClientError("GDELT returned JSON in an unexpected format.")
            self._emit_progress(progress_callback, event_context, phase="success")
            self._pause(self.request_pause_seconds)
            return data

        self._emit_progress(progress_callback, event_context, phase="fatal_failure", error=str(last_error))
        raise GdeltClientError("GDELT request failed unexpectedly.") from last_error

    def _retry_delay(self, attempt: int, *, response: Any | None = None) -> float:
        retry_after = _retry_after_seconds(response) if response is not None else None
        if retry_after is not None:
            return min(retry_after, self.max_backoff_seconds)

        delay = min(self.backoff_seconds * (2**attempt), self.max_backoff_seconds)
        if self.jitter_seconds > 0:
            delay += self.jitter(0, self.jitter_seconds)
        return min(delay, self.max_backoff_seconds)

    def _pause(self, delay: float) -> None:
        if delay > 0:
            self.sleep(delay)

    @staticmethod
    def _emit_progress(
        progress_callback: Callable[[Mapping[str, Any]], None] | None,
        context: Mapping[str, Any],
        **event: Any,
    ) -> None:
        if progress_callback is None:
            return
        payload = dict(context)
        payload.update(event)
        progress_callback(payload)


def build_gdelt_query(
    query_type: GdeltQueryType,
    *,
    market_terms: Sequence[str] = MARKET_QUERY_TERMS,
    criticism_terms: Sequence[str] = CRITICISM_QUERY_TERMS,
    source_language: str | None = DEFAULT_GDELT_SOURCE_LANGUAGE,
) -> str:
    """Build the GDELT query string for a supported MVP query type."""

    market_query = f"({_join_terms(market_terms)})"
    if query_type is GdeltQueryType.ALL_MARKET:
        return _append_source_language(market_query, source_language)

    criticism_query = f"({_join_terms(criticism_terms)})"
    return _append_source_language(f"{market_query} {criticism_query}", source_language)


def collect_gdelt_headlines(
    spec: GdeltRequestSpec,
    *,
    client: GdeltClient | None = None,
) -> GdeltCollectionResult:
    """Collect GDELT headlines, save raw JSON, and write cleaned metadata CSV."""

    if spec.start_date > spec.end_date:
        raise ValueError("start_date must be on or before end_date.")
    if spec.max_records <= 0:
        raise ValueError("max_records must be positive.")

    raw_path = gdelt_output_path(spec.raw_dir, spec.query_type, spec.start_date, spec.end_date, "json")
    interim_path = gdelt_output_path(spec.interim_dir, spec.query_type, spec.start_date, spec.end_date, "csv")
    _ensure_writable(raw_path, False, allow_overwrite=False)
    _ensure_writable(interim_path, spec.overwrite)

    query = build_gdelt_query(
        spec.query_type,
        market_terms=spec.market_terms,
        criticism_terms=spec.criticism_terms,
        source_language=spec.source_language,
    )
    gdelt_client = client or GdeltClient()
    responses: list[dict[str, Any]] = []

    for day in _iter_dates(spec.start_date, spec.end_date):
        data = gdelt_client.fetch_day(query, day, spec.max_records)
        responses.append(
            {
                "date": day.isoformat(),
                "query": query,
                "response": data,
            }
        )

    raw_payload = {
        "query_type": spec.query_type.value,
        "start_date": spec.start_date.isoformat(),
        "end_date": spec.end_date.isoformat(),
        "source_language": spec.source_language or "",
        "responses": responses,
    }
    rows = clean_gdelt_articles(raw_payload, spec.query_type)

    spec.raw_dir.mkdir(parents=True, exist_ok=True)
    spec.interim_dir.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(raw_payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_article_csv(interim_path, rows)

    return GdeltCollectionResult(raw_path=raw_path, interim_path=interim_path, article_count=len(rows))


def collect_gdelt_longrun(
    spec: GdeltLongRunSpec,
    *,
    client: GdeltClient | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> GdeltLongRunResult:
    """Collect GDELT DOC ArticleList data with resumable daily checkpoints."""

    _validate_longrun_spec(spec)
    gdelt_client = client or GdeltClient(request_pause_seconds=spec.request_pause_seconds)

    raw_paths: dict[str, Path] = {}
    interim_paths: dict[str, Path] = {}
    article_counts: dict[str, int] = {}
    completed_days = 0
    skipped_existing = 0

    for query_type in spec.query_types:
        result = _collect_longrun_query(spec, query_type, gdelt_client, progress_callback)
        raw_paths[query_type.value] = result.raw_path
        interim_paths[query_type.value] = result.interim_path
        article_counts[query_type.value] = result.article_count
        completed_days += _day_count(spec.start_date, spec.end_date)
        skipped_existing += result.skipped_existing_checkpoints

    return GdeltLongRunResult(
        raw_paths=raw_paths,
        interim_paths=interim_paths,
        article_counts=article_counts,
        completed_days=completed_days,
        skipped_existing_checkpoints=skipped_existing,
    )


def gdelt_longrun_status(spec: GdeltLongRunSpec) -> pd.DataFrame:
    """Return daily checkpoint coverage for a resumable GDELT long run."""

    _validate_longrun_spec(spec)
    rows: list[dict[str, object]] = []
    for query_type in spec.query_types:
        for day in _iter_dates(spec.start_date, spec.end_date):
            checkpoint_path = _daily_checkpoint_path(spec.raw_dir, query_type, day)
            rows.append(
                {
                    "query_type": query_type.value,
                    "date": day.isoformat(),
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_exists": checkpoint_path.exists(),
                    "aggregate_raw_path": str(
                        gdelt_output_path(spec.raw_dir, query_type, spec.start_date, spec.end_date, "json")
                    ),
                    "interim_path": str(
                        gdelt_output_path(spec.interim_dir, query_type, spec.start_date, spec.end_date, "csv")
                    ),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "query_type",
            "date",
            "checkpoint_path",
            "checkpoint_exists",
            "aggregate_raw_path",
            "interim_path",
        ],
    )


def dry_run_gdelt_longrun(spec: GdeltLongRunSpec) -> pd.DataFrame:
    """Summarise request counts and checkpoint coverage before a GDELT long run."""

    status = gdelt_longrun_status(spec)
    rows: list[dict[str, object]] = []
    for query_type, group in status.groupby("query_type", sort=False):
        existing = int(group["checkpoint_exists"].sum())
        missing = int((~group["checkpoint_exists"]).sum())
        rows.append(
            {
                "query_type": query_type,
                "start_date": spec.start_date.isoformat(),
                "end_date": spec.end_date.isoformat(),
                "request_count": len(group),
                "existing_checkpoints": existing,
                "missing_checkpoints": missing,
                "estimated_minimum_duration_seconds": missing * spec.request_pause_seconds,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "query_type",
            "start_date",
            "end_date",
            "request_count",
            "existing_checkpoints",
            "missing_checkpoints",
            "estimated_minimum_duration_seconds",
        ],
    )


def collect_gkg_bulk_extract(
    spec: GkgBulkSpec,
    *,
    session: Any | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> GkgBulkResult:
    """Collect filtered GKG archive rows as a historical fallback extract.

    GKG rows are not equivalent to DOC ArticleList headline records. This helper stores
    filtered extracts plus a source manifest and does not retain full zip archives unless
    explicitly requested.
    """

    if spec.start_date > spec.end_date:
        raise ValueError("start_date must be on or before end_date.")
    if not spec.filter_terms:
        raise ValueError("At least one filter term is required.")

    extract_path = spec.raw_dir / f"gkg_filtered_{spec.start_date:%Y%m%d}_{spec.end_date:%Y%m%d}.csv"
    manifest_path = spec.raw_dir / f"gkg_filtered_manifest_{spec.start_date:%Y%m%d}_{spec.end_date:%Y%m%d}.csv"
    _ensure_writable(extract_path, spec.overwrite, allow_overwrite=False)
    _ensure_writable(manifest_path, spec.overwrite, allow_overwrite=False)

    http = session or requests.Session()
    master_response = http.get(spec.masterfilelist_url, timeout=60)
    if getattr(master_response, "status_code", 0) >= 400:
        raise GdeltClientError(f"GKG masterfile list returned HTTP status {master_response.status_code}.")

    archive_urls = _gkg_archive_urls(master_response.text, spec.start_date, spec.end_date)
    if spec.max_archives is not None:
        archive_urls = archive_urls[: spec.max_archives]

    spec.raw_dir.mkdir(parents=True, exist_ok=True)
    archive_dir = spec.raw_dir / "archives"
    if spec.keep_archives:
        archive_dir.mkdir(parents=True, exist_ok=True)

    terms = tuple(term.lower() for term in spec.filter_terms)
    matched_rows = 0
    with extract_path.open("x", encoding="utf-8", newline="") as extract_file, manifest_path.open(
        "x",
        encoding="utf-8",
        newline="",
    ) as manifest_file:
        extract_writer = csv.DictWriter(extract_file, fieldnames=("archive_url", "row_text"))
        manifest_writer = csv.DictWriter(
            manifest_file,
            fieldnames=("archive_url", "archive_date", "matched_rows"),
        )
        extract_writer.writeheader()
        manifest_writer.writeheader()

        for archive_url in archive_urls:
            _emit_longrun_progress(progress_callback, phase="gkg_request", archive_url=archive_url)
            response = http.get(archive_url, timeout=120)
            if getattr(response, "status_code", 0) >= 400:
                raise GdeltClientError(f"GKG archive returned HTTP status {response.status_code}: {archive_url}")

            content = response.content
            if spec.keep_archives:
                (archive_dir / Path(archive_url).name).write_bytes(content)

            archive_matches = _write_matching_gkg_rows(content, archive_url, terms, extract_writer)
            matched_rows += archive_matches
            manifest_writer.writerow(
                {
                    "archive_url": archive_url,
                    "archive_date": _gkg_archive_date(archive_url).isoformat(),
                    "matched_rows": archive_matches,
                }
            )
            _emit_longrun_progress(
                progress_callback,
                phase="gkg_archive_complete",
                archive_url=archive_url,
                matched_rows=archive_matches,
            )

    return GkgBulkResult(
        extract_path=extract_path,
        manifest_path=manifest_path,
        archive_count=len(archive_urls),
        matched_row_count=matched_rows,
    )


def _collect_longrun_query(
    spec: GdeltLongRunSpec,
    query_type: GdeltQueryType,
    client: GdeltClient,
    progress_callback: Callable[[Mapping[str, Any]], None] | None,
) -> _LongRunQueryResult:
    raw_path = gdelt_output_path(spec.raw_dir, query_type, spec.start_date, spec.end_date, "json")
    interim_path = gdelt_output_path(spec.interim_dir, query_type, spec.start_date, spec.end_date, "csv")
    query = build_gdelt_query(
        query_type,
        market_terms=spec.market_terms,
        criticism_terms=spec.criticism_terms,
        source_language=spec.source_language,
    )

    if raw_path.exists() and interim_path.exists() and not spec.overwrite_interim:
        _validate_saved_longrun_payload(_read_json(raw_path), query, spec.source_language, raw_path)
        rows = _read_article_count(interim_path)
        _emit_longrun_progress(
            progress_callback,
            phase="skip",
            query_type=query_type.value,
            message="aggregate outputs already exist",
            raw_path=str(raw_path),
            interim_path=str(interim_path),
        )
        return _LongRunQueryResult(raw_path, interim_path, rows, _day_count(spec.start_date, spec.end_date))

    _ensure_writable(interim_path, spec.overwrite_interim)
    if raw_path.exists():
        raw_payload = _read_json(raw_path)
        _validate_saved_longrun_payload(raw_payload, query, spec.source_language, raw_path)
        rows = clean_gdelt_articles(raw_payload, query_type)
        spec.interim_dir.mkdir(parents=True, exist_ok=True)
        _write_article_csv(interim_path, rows)
        _emit_longrun_progress(
            progress_callback,
            phase="aggregate",
            query_type=query_type.value,
            raw_path=str(raw_path),
            interim_path=str(interim_path),
            article_count=len(rows),
            message="rebuilt interim CSV from existing raw aggregate",
        )
        return _LongRunQueryResult(raw_path, interim_path, len(rows), 0)

    responses: list[dict[str, Any]] = []
    skipped_existing = 0

    for day in _iter_dates(spec.start_date, spec.end_date):
        checkpoint_path = _daily_checkpoint_path(spec.raw_dir, query_type, day)
        if checkpoint_path.exists():
            if not spec.resume:
                raise FileExistsError(
                    f"{checkpoint_path} already exists. Pass resume=True to reuse daily raw checkpoints."
                )
            checkpoint_payload = _read_json(checkpoint_path)
            _validate_saved_response_block(checkpoint_payload, query, checkpoint_path)
            responses.append(checkpoint_payload)
            skipped_existing += 1
            _emit_longrun_progress(
                progress_callback,
                phase="skip",
                query_type=query_type.value,
                date=day.isoformat(),
                checkpoint_path=str(checkpoint_path),
                message="skipped existing checkpoint",
            )
            continue

        context = {"query_type": query_type.value, "date": day.isoformat()}
        try:
            data = client.fetch_day(query, day, spec.max_records, progress_callback=progress_callback, context=context)
        except GdeltClientError as exc:
            _raise_longrun_error(exc, spec, query_type, completed_days=len(responses))

        checkpoint_payload = {
            "date": day.isoformat(),
            "query": query,
            "response": data,
        }
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with checkpoint_path.open("x", encoding="utf-8") as checkpoint_file:
                json.dump(checkpoint_payload, checkpoint_file, indent=2, sort_keys=True)
        except FileExistsError:
            checkpoint_payload = _read_json(checkpoint_path)
            _validate_saved_response_block(checkpoint_payload, query, checkpoint_path)
            skipped_existing += 1
        responses.append(checkpoint_payload)
        _emit_longrun_progress(
            progress_callback,
            phase="checkpoint",
            query_type=query_type.value,
            date=day.isoformat(),
            checkpoint_path=str(checkpoint_path),
        )

    raw_payload = {
        "query_type": query_type.value,
        "start_date": spec.start_date.isoformat(),
        "end_date": spec.end_date.isoformat(),
        "source_language": spec.source_language or "",
        "checkpoint_dir": str(_daily_checkpoint_root(spec.raw_dir, query_type)),
        "responses": responses,
    }
    rows = clean_gdelt_articles(raw_payload, query_type)

    spec.raw_dir.mkdir(parents=True, exist_ok=True)
    spec.interim_dir.mkdir(parents=True, exist_ok=True)
    with raw_path.open("x", encoding="utf-8") as raw_file:
        json.dump(raw_payload, raw_file, indent=2, sort_keys=True)
    _write_article_csv(interim_path, rows)
    _emit_longrun_progress(
        progress_callback,
        phase="aggregate",
        query_type=query_type.value,
        raw_path=str(raw_path),
        interim_path=str(interim_path),
        article_count=len(rows),
    )
    return _LongRunQueryResult(raw_path, interim_path, len(rows), skipped_existing)


def clean_gdelt_articles(
    raw_payload: Mapping[str, Any],
    query_type: GdeltQueryType,
) -> list[dict[str, str]]:
    """Extract stable article metadata fields from saved GDELT responses."""

    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    source_language = _as_str(raw_payload.get("source_language")).strip().lower()

    for response_block in raw_payload.get("responses", []):
        if not isinstance(response_block, Mapping):
            continue
        response_date = _as_str(response_block.get("date"))
        response = response_block.get("response", {})
        if not isinstance(response, Mapping):
            continue
        articles = response.get("articles", [])
        if not isinstance(articles, list):
            continue

        for article in articles:
            if not isinstance(article, Mapping):
                continue
            row = _article_row(article, query_type)
            if not _article_matches_source_language(row, source_language):
                continue
            key = _article_dedupe_key(row, response_date)
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)

    return rows


def gdelt_output_path(
    base_dir: Path,
    query_type: GdeltQueryType,
    start_date: date,
    end_date: date,
    extension: str,
) -> Path:
    """Return the deterministic output path for a GDELT collection artifact."""

    filename = f"gdelt_{query_type.value}_{start_date:%Y%m%d}_{end_date:%Y%m%d}.{extension}"
    return base_dir / filename


def _article_row(article: Mapping[str, Any], query_type: GdeltQueryType) -> dict[str, str]:
    domain = _as_str(article.get("domain"))
    source = _as_str(article.get("source") or article.get("sourceName") or domain)
    return {
        "title": _as_str(article.get("title")),
        "url": _as_str(article.get("url") or article.get("url_mobile")),
        "domain": domain,
        "source": source,
        "seendate": _as_str(article.get("seendate")),
        "language": _as_str(article.get("language")),
        "query_type": query_type.value,
    }


def _article_matches_source_language(row: Mapping[str, str], source_language: str) -> bool:
    if source_language not in {"english", "eng", "en"}:
        return True
    language = _as_str(row.get("language")).strip().lower()
    if not language:
        return True
    return language in {"english", "eng", "en"}


def _article_dedupe_key(row: Mapping[str, str], response_date: str) -> tuple[str, str, str]:
    title_key = normalise_title(_as_str(row.get("title"))).replace(" ", "")
    text_key = title_key or normalise_title(_as_str(row.get("url"))).replace(" ", "")
    return (
        response_date,
        text_key,
        _as_str(row.get("domain")).strip().lower(),
    )


def _write_article_csv(path: Path, rows: Iterable[Mapping[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=GDELT_ARTICLE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _ensure_writable(path: Path, overwrite: bool, *, allow_overwrite: bool = True) -> None:
    if path.exists() and not overwrite:
        if not allow_overwrite:
            raise FileExistsError(
                f"{path} already exists. Choose a different date range to avoid overwriting raw data."
            )
        raise FileExistsError(
            f"{path} already exists. Pass overwrite=True or choose a different date range "
            "to avoid overwriting generated data."
        )


def _validate_longrun_spec(spec: GdeltLongRunSpec) -> None:
    if spec.start_date > spec.end_date:
        raise ValueError("start_date must be on or before end_date.")
    if spec.max_records <= 0:
        raise ValueError("max_records must be positive.")
    if spec.chunking_mode != "daily":
        raise ValueError("Only daily GDELT DOC chunking is supported.")
    if not spec.query_types:
        raise ValueError("At least one query type is required.")


def _daily_checkpoint_root(raw_dir: Path, query_type: GdeltQueryType) -> Path:
    return raw_dir / "doc_daily" / query_type.value


def _daily_checkpoint_path(raw_dir: Path, query_type: GdeltQueryType, day: date) -> Path:
    return _daily_checkpoint_root(raw_dir, query_type) / f"{day:%Y}" / f"{day:%Y%m%d}.json"


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise GdeltClientError(f"{path} does not contain a JSON object.")
    return data


def _read_article_count(path: Path) -> int:
    with path.open(encoding="utf-8", newline="") as csv_file:
        return sum(1 for _ in csv.DictReader(csv_file))


def _validate_saved_longrun_payload(
    raw_payload: Mapping[str, Any],
    expected_query: str,
    expected_source_language: str | None,
    path: Path,
) -> None:
    saved_source_language = raw_payload.get("source_language")
    if saved_source_language is not None and _as_str(saved_source_language) != _as_str(expected_source_language):
        raise _incompatible_raw_error(
            path,
            f"saved source_language {_as_str(saved_source_language)!r} does not match "
            f"current source_language {_as_str(expected_source_language)!r}",
        )

    responses = raw_payload.get("responses", [])
    if not isinstance(responses, list):
        raise _incompatible_raw_error(path, "saved responses are not a list")
    for response_block in responses:
        if not isinstance(response_block, Mapping):
            raise _incompatible_raw_error(path, "saved response block is not an object")
        _validate_saved_response_block(response_block, expected_query, path)


def _validate_saved_response_block(
    response_block: Mapping[str, Any],
    expected_query: str,
    path: Path,
) -> None:
    saved_query = response_block.get("query")
    if saved_query is None:
        raise _incompatible_raw_error(path, "saved response block is missing query metadata")
    if _as_str(saved_query) != expected_query:
        raise _incompatible_raw_error(path, "saved query does not match the current GDELT query")


def _incompatible_raw_error(path: Path, reason: str) -> FileExistsError:
    return FileExistsError(
        f"{path} is incompatible with the current GDELT long-run spec: {reason}. "
        "Raw data is not overwritten. Use a different raw_dir, change the date range, "
        "or intentionally reuse the old aggregate as-is."
    )


def _day_count(start_date: date, end_date: date) -> int:
    return (end_date - start_date).days + 1


def _emit_longrun_progress(
    progress_callback: Callable[[Mapping[str, Any]], None] | None,
    **event: Any,
) -> None:
    if progress_callback is not None:
        progress_callback(event)


def _raise_longrun_error(
    error: GdeltClientError,
    spec: GdeltLongRunSpec,
    query_type: GdeltQueryType,
    *,
    completed_days: int,
) -> None:
    status = gdelt_longrun_status(spec)
    query_status = status[status["query_type"] == query_type.value]
    missing = query_status[~query_status["checkpoint_exists"]]
    next_missing = (
        date.fromisoformat(str(missing["date"].iloc[0]))
        if not missing.empty
        else None
    )
    guidance = _resume_guidance(spec, query_type)
    message = (
        f"{error}\n"
        f"Completed checkpoint days for {query_type.value}: {completed_days}.\n"
        f"Next missing day: {next_missing.isoformat() if next_missing else 'none'}.\n"
        f"Resume guidance:\n{guidance}"
    )
    raise GdeltClientError(
        message,
        resume_guidance=guidance,
        completed_days=completed_days,
        next_missing_day=next_missing,
    ) from error


def _resume_guidance(spec: GdeltLongRunSpec, query_type: GdeltQueryType) -> str:
    return (
        "spec = GdeltLongRunSpec(\n"
        f"    start_date=date({spec.start_date.year}, {spec.start_date.month}, {spec.start_date.day}),\n"
        f"    end_date=date({spec.end_date.year}, {spec.end_date.month}, {spec.end_date.day}),\n"
        f"    query_types=(GdeltQueryType.{query_type.name},),\n"
        f"    max_records={spec.max_records},\n"
        f"    source_language={spec.source_language!r},\n"
        f"    raw_dir=Path({str(spec.raw_dir)!r}),\n"
        f"    interim_dir=Path({str(spec.interim_dir)!r}),\n"
        "    resume=True,\n"
        f"    overwrite_interim={spec.overwrite_interim},\n"
        ")\n"
        "collect_gdelt_longrun(spec, progress_callback=print_progress)"
    )


def _retry_after_seconds(response: Any) -> float | None:
    headers = getattr(response, "headers", {}) or {}
    value = headers.get("Retry-After") if isinstance(headers, Mapping) else None
    if value is None:
        return None

    text = str(value).strip()
    try:
        seconds = float(text)
    except ValueError:
        try:
            retry_time = parsedate_to_datetime(text)
        except (TypeError, ValueError):
            return None
        if retry_time.tzinfo is None:
            retry_time = retry_time.replace(tzinfo=timezone.utc)
        seconds = (retry_time - datetime.now(timezone.utc)).total_seconds()
    return max(0.0, seconds)


def _gkg_archive_urls(masterfile_text: str, start_date: date, end_date: date) -> list[str]:
    urls: list[str] = []
    for line in masterfile_text.splitlines():
        parts = line.split()
        if not parts:
            continue
        url = parts[-1]
        if not url.endswith(".gkg.csv.zip"):
            continue
        archive_date = _gkg_archive_date(url)
        if start_date <= archive_date <= end_date:
            urls.append(url)
    return sorted(urls)


def _gkg_archive_date(archive_url: str) -> date:
    name = Path(archive_url).name
    return datetime.strptime(name[:8], "%Y%m%d").date()


def _write_matching_gkg_rows(
    zip_content: bytes,
    archive_url: str,
    terms: Sequence[str],
    writer: csv.DictWriter,
) -> int:
    matched_rows = 0
    with zipfile.ZipFile(io.BytesIO(zip_content)) as archive:
        for member_name in archive.namelist():
            with archive.open(member_name) as member:
                for raw_line in member:
                    text = raw_line.decode("utf-8", errors="replace").rstrip("\n")
                    lower_text = text.lower()
                    if any(term in lower_text for term in terms):
                        writer.writerow({"archive_url": archive_url, "row_text": text})
                        matched_rows += 1
    return matched_rows


def _iter_dates(start_date: date, end_date: date) -> Iterable[date]:
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def _join_terms(terms: Sequence[str]) -> str:
    return " OR ".join(_format_term(term) for term in terms)


def _append_source_language(query: str, source_language: str | None) -> str:
    if source_language is None:
        return query
    language = source_language.strip().lower()
    if not language:
        return query
    if not language.replace("-", "").isalnum():
        raise ValueError("source_language must contain only letters, numbers, or hyphens.")
    return f"{query} sourcelang:{language}"


def _format_term(term: str) -> str:
    escaped = term.replace('"', '\\"')
    if any(character.isspace() for character in escaped) or any(
        character in escaped for character in ("&", "-", ".")
    ):
        return f'"{escaped}"'
    return escaped


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _response_text(response: Any) -> str:
    text = getattr(response, "text", "")
    if not text:
        return "no response body"
    return text[:300]
