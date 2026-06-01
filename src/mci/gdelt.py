"""GDELT DOC 2.0 headline collection utilities."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import requests

from mci.config import CRITICISM_QUERY_TERMS, INTERIM_DATA_DIR, MARKET_QUERY_TERMS, RAW_DATA_DIR

GDELT_DOC_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_ARTICLE_FIELDS = ("title", "url", "domain", "source", "seendate", "language", "query_type")


class GdeltClientError(RuntimeError):
    """Raised when GDELT collection cannot complete cleanly."""


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


class GdeltClient:
    """Small GDELT DOC 2.0 client with retry handling."""

    def __init__(
        self,
        *,
        endpoint: str = GDELT_DOC_ENDPOINT,
        session: Any | None = None,
        timeout: int = 30,
        max_retries: int = 3,
        backoff_seconds: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.endpoint = endpoint
        self.session = session or requests.Session()
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.sleep = sleep

        headers = getattr(self.session, "headers", None)
        if headers is not None:
            headers.setdefault("User-Agent", "market-criticism-index/0.1.0 research")

    def fetch_day(self, query: str, day: date, max_records: int) -> dict[str, Any]:
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
        return self._get_json(params)

    def _get_json(self, params: Mapping[str, str]) -> dict[str, Any]:
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.get(self.endpoint, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                last_error = exc
                if attempt < self.max_retries:
                    self._pause(attempt)
                    continue
                raise GdeltClientError(
                    "GDELT request failed after retries. Please check the network connection "
                    "or try a smaller date range."
                ) from exc

            status_code = getattr(response, "status_code", 0)
            if status_code == 429 or status_code >= 500:
                if attempt < self.max_retries:
                    self._pause(attempt)
                    continue
                raise GdeltClientError(
                    f"GDELT returned HTTP status {status_code} after retries. "
                    "Please wait and try again later."
                )

            if status_code >= 400:
                message = _response_text(response)
                raise GdeltClientError(f"GDELT returned HTTP status {status_code}: {message}")

            try:
                data = response.json()
            except ValueError as exc:
                raise GdeltClientError("GDELT returned a response that was not valid JSON.") from exc

            if not isinstance(data, dict):
                raise GdeltClientError("GDELT returned JSON in an unexpected format.")
            return data

        raise GdeltClientError("GDELT request failed unexpectedly.") from last_error

    def _pause(self, attempt: int) -> None:
        delay = self.backoff_seconds * (2**attempt)
        self.sleep(delay)


def build_gdelt_query(
    query_type: GdeltQueryType,
    *,
    market_terms: Sequence[str] = MARKET_QUERY_TERMS,
    criticism_terms: Sequence[str] = CRITICISM_QUERY_TERMS,
) -> str:
    """Build the GDELT query string for a supported MVP query type."""

    market_query = f"({_join_terms(market_terms)})"
    if query_type is GdeltQueryType.ALL_MARKET:
        return market_query

    criticism_query = f"({_join_terms(criticism_terms)})"
    return f"{market_query} {criticism_query}"


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
        "responses": responses,
    }
    rows = clean_gdelt_articles(raw_payload, spec.query_type)

    spec.raw_dir.mkdir(parents=True, exist_ok=True)
    spec.interim_dir.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(raw_payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_article_csv(interim_path, rows)

    return GdeltCollectionResult(raw_path=raw_path, interim_path=interim_path, article_count=len(rows))


def clean_gdelt_articles(
    raw_payload: Mapping[str, Any],
    query_type: GdeltQueryType,
) -> list[dict[str, str]]:
    """Extract stable article metadata fields from saved GDELT responses."""

    rows: list[dict[str, str]] = []
    for response_block in raw_payload.get("responses", []):
        if not isinstance(response_block, Mapping):
            continue
        response = response_block.get("response", {})
        if not isinstance(response, Mapping):
            continue
        articles = response.get("articles", [])
        if not isinstance(articles, list):
            continue

        for article in articles:
            if not isinstance(article, Mapping):
                continue
            rows.append(_article_row(article, query_type))

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


def _iter_dates(start_date: date, end_date: date) -> Iterable[date]:
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def _join_terms(terms: Sequence[str]) -> str:
    return " OR ".join(_format_term(term) for term in terms)


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
