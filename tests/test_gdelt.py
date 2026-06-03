"""Tests for mocked GDELT collection."""

from __future__ import annotations

import csv
import io
import json
import zipfile
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from mci.data_collection import HeadlineCollectionSpec, collect_market_headlines
from mci.gdelt import (
    GDELT_ARTICLE_FIELDS,
    GdeltClient,
    GdeltClientError,
    GdeltLongRunSpec,
    GkgBulkSpec,
    GdeltQueryType,
    GdeltRequestSpec,
    build_gdelt_query,
    collect_gdelt_headlines,
    collect_gdelt_longrun,
    collect_gkg_bulk_extract,
    dry_run_gdelt_longrun,
    gdelt_longrun_status,
)


class FakeResponse:
    def __init__(
        self,
        status_code: int,
        data: dict[str, Any],
        text: str = "",
        headers: dict[str, str] | None = None,
        content: bytes | None = None,
    ) -> None:
        self.status_code = status_code
        self._data = data
        self.text = text
        self.headers = headers or {}
        self.content = content or b""

    def json(self) -> dict[str, Any]:
        return self._data


class FakeSession:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []
        self.headers: dict[str, str] = {}

    def get(self, url: str, *, params: dict[str, str], timeout: int) -> FakeResponse:
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        return self.responses.pop(0)


class FakeGkgSession:
    def __init__(self, responses: dict[str, FakeResponse]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def get(self, url: str, timeout: int) -> FakeResponse:
        self.calls.append(url)
        return self.responses[url]


def test_builds_supported_queries() -> None:
    market_query = build_gdelt_query(GdeltQueryType.ALL_MARKET)
    criticism_query = build_gdelt_query(GdeltQueryType.CANDIDATE_CRITICISM)

    assert '"US stock market"' in market_query
    assert "bubble" not in market_query
    assert '"US stock market"' in criticism_query
    assert "bubble" in criticism_query


def test_high_level_collection_uses_custom_market_terms(tmp_path: Path) -> None:
    session = FakeSession([FakeResponse(200, {"articles": []})])
    client = GdeltClient(session=session, max_retries=0, sleep=lambda _: None)
    spec = HeadlineCollectionSpec(
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 1),
        query_terms=("custom market signal",),
        raw_output_dir=tmp_path / "raw",
        interim_output_dir=tmp_path / "interim",
    )

    collect_market_headlines(spec, client=client)

    query = session.calls[0]["params"]["query"]
    assert '"custom market signal"' in query
    assert "US stock market" not in query


def test_collects_raw_and_cleaned_outputs_with_mocked_response(tmp_path: Path) -> None:
    session = FakeSession(
        [
            FakeResponse(
                200,
                {
                    "articles": [
                        {
                            "title": "Wall Street warning grows",
                            "url": "https://example.com/story",
                            "domain": "example.com",
                            "seendate": "20240101120000",
                            "language": "English",
                        }
                    ]
                },
            )
        ]
    )
    client = GdeltClient(session=session, max_retries=0, sleep=lambda _: None)
    spec = GdeltRequestSpec(
        query_type=GdeltQueryType.ALL_MARKET,
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 1),
        raw_dir=tmp_path / "raw",
        interim_dir=tmp_path / "interim",
    )

    result = collect_gdelt_headlines(spec, client=client)

    assert result.raw_path.name == "gdelt_all_us_market_20240101_20240101.json"
    assert result.interim_path.name == "gdelt_all_us_market_20240101_20240101.csv"
    assert result.article_count == 1
    assert len(session.calls) == 1

    raw_payload = json.loads(result.raw_path.read_text(encoding="utf-8"))
    assert raw_payload["query_type"] == "all_us_market"
    assert raw_payload["responses"][0]["response"]["articles"][0]["title"] == "Wall Street warning grows"

    with result.interim_path.open(encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert rows == [
        {
            "title": "Wall Street warning grows",
            "url": "https://example.com/story",
            "domain": "example.com",
            "source": "example.com",
            "seendate": "20240101120000",
            "language": "English",
            "query_type": "all_us_market",
        }
    ]
    assert rows[0].keys() == set(GDELT_ARTICLE_FIELDS)


def test_retries_transient_server_errors(tmp_path: Path) -> None:
    session = FakeSession(
        [
            FakeResponse(503, {}, "temporary outage"),
            FakeResponse(200, {"articles": []}),
        ]
    )
    client = GdeltClient(session=session, max_retries=1, sleep=lambda _: None)
    spec = GdeltRequestSpec(
        query_type=GdeltQueryType.CANDIDATE_CRITICISM,
        start_date=date(2024, 1, 2),
        end_date=date(2024, 1, 2),
        raw_dir=tmp_path / "raw",
        interim_dir=tmp_path / "interim",
    )

    result = collect_gdelt_headlines(spec, client=client)

    assert result.article_count == 0
    assert len(session.calls) == 2


def test_respects_retry_after_header_and_reports_progress() -> None:
    session = FakeSession(
        [
            FakeResponse(429, {}, "too many requests", headers={"Retry-After": "120"}),
            FakeResponse(200, {"articles": []}),
        ]
    )
    sleeps: list[float] = []
    events: list[dict[str, Any]] = []
    client = GdeltClient(
        session=session,
        max_retries=1,
        request_pause_seconds=0,
        jitter_seconds=0,
        sleep=sleeps.append,
    )

    client.fetch_day(
        "market",
        date(2024, 1, 1),
        10,
        progress_callback=lambda event: events.append(dict(event)),
        context={"query_type": "all_us_market", "date": "2024-01-01"},
    )

    assert sleeps == [120]
    assert any(event["phase"] == "retry" and event["sleep_seconds"] == 120 for event in events)
    assert events[-1]["phase"] == "success"


def test_uses_capped_exponential_backoff_with_jitter() -> None:
    session = FakeSession(
        [
            FakeResponse(503, {}, "temporary outage"),
            FakeResponse(200, {"articles": []}),
        ]
    )
    sleeps: list[float] = []
    client = GdeltClient(
        session=session,
        max_retries=1,
        backoff_seconds=1000,
        max_backoff_seconds=1005,
        jitter_seconds=10,
        request_pause_seconds=2,
        jitter=lambda _start, _end: 10,
        sleep=sleeps.append,
    )

    client.fetch_day("market", date(2024, 1, 1), 10)

    assert sleeps == [1005, 2]


def test_raises_after_retry_budget_is_exhausted(tmp_path: Path) -> None:
    session = FakeSession(
        [
            FakeResponse(503, {}, "temporary outage"),
            FakeResponse(503, {}, "temporary outage"),
        ]
    )
    client = GdeltClient(session=session, max_retries=1, sleep=lambda _: None)
    spec = GdeltRequestSpec(
        query_type=GdeltQueryType.ALL_MARKET,
        start_date=date(2024, 1, 3),
        end_date=date(2024, 1, 3),
        raw_dir=tmp_path / "raw",
        interim_dir=tmp_path / "interim",
    )

    with pytest.raises(GdeltClientError, match="HTTP status 503"):
        collect_gdelt_headlines(spec, client=client)


def test_does_not_overwrite_existing_outputs(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    interim_dir = tmp_path / "interim"
    raw_dir.mkdir()
    expected_raw = raw_dir / "gdelt_all_us_market_20240104_20240104.json"
    expected_raw.write_text("{}", encoding="utf-8")
    spec = GdeltRequestSpec(
        query_type=GdeltQueryType.ALL_MARKET,
        start_date=date(2024, 1, 4),
        end_date=date(2024, 1, 4),
        raw_dir=raw_dir,
        interim_dir=interim_dir,
        overwrite=True,
    )

    with pytest.raises(FileExistsError, match="raw data"):
        collect_gdelt_headlines(spec, client=GdeltClient(session=FakeSession([])))

    assert expected_raw.read_text(encoding="utf-8") == "{}"


def test_longrun_status_and_dry_run_report_missing_days(tmp_path: Path) -> None:
    spec = GdeltLongRunSpec(
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 2),
        query_types=(GdeltQueryType.ALL_MARKET,),
        raw_dir=tmp_path / "raw",
        interim_dir=tmp_path / "interim",
        request_pause_seconds=10,
    )
    checkpoint = tmp_path / "raw" / "doc_daily" / "all_us_market" / "2024" / "20240101.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text(
        json.dumps({"date": "2024-01-01", "query": "market", "response": {"articles": []}}),
        encoding="utf-8",
    )

    status = gdelt_longrun_status(spec)
    dry_run = dry_run_gdelt_longrun(spec)

    assert status["checkpoint_exists"].tolist() == [True, False]
    assert status["date"].tolist() == ["2024-01-01", "2024-01-02"]
    assert dry_run.loc[0, "request_count"] == 2
    assert dry_run.loc[0, "existing_checkpoints"] == 1
    assert dry_run.loc[0, "missing_checkpoints"] == 1
    assert dry_run.loc[0, "estimated_minimum_duration_seconds"] == 10


def test_longrun_resumes_from_checkpoints_and_writes_aggregate_outputs(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    interim_dir = tmp_path / "interim"
    checkpoint = raw_dir / "doc_daily" / "all_us_market" / "2024" / "20240101.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text(
        json.dumps(
            {
                "date": "2024-01-01",
                "query": "market",
                "response": {
                    "articles": [
                        {
                            "title": "Existing checkpoint",
                            "url": "https://example.com/1",
                            "domain": "example.com",
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    session = FakeSession(
        [
            FakeResponse(
                200,
                {
                    "articles": [
                        {
                            "title": "Fetched checkpoint",
                            "url": "https://example.com/2",
                            "domain": "example.com",
                        }
                    ]
                },
            )
        ]
    )
    client = GdeltClient(session=session, max_retries=0, request_pause_seconds=0, sleep=lambda _: None)
    events: list[dict[str, Any]] = []
    spec = GdeltLongRunSpec(
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 2),
        query_types=(GdeltQueryType.ALL_MARKET,),
        raw_dir=raw_dir,
        interim_dir=interim_dir,
    )

    result = collect_gdelt_longrun(spec, client=client, progress_callback=lambda event: events.append(dict(event)))

    assert result.article_counts == {"all_us_market": 2}
    assert result.skipped_existing_checkpoints == 1
    assert len(session.calls) == 1
    assert (raw_dir / "doc_daily" / "all_us_market" / "2024" / "20240102.json").exists()
    assert result.raw_paths["all_us_market"].exists()
    assert result.interim_paths["all_us_market"].exists()
    assert any(event["phase"] == "skip" for event in events)
    assert any(event["phase"] == "checkpoint" for event in events)


def test_longrun_failure_leaves_completed_checkpoints_and_resume_guidance(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    interim_dir = tmp_path / "interim"
    session = FakeSession(
        [
            FakeResponse(200, {"articles": []}),
            FakeResponse(429, {}, "too many requests"),
        ]
    )
    client = GdeltClient(session=session, max_retries=0, request_pause_seconds=0, sleep=lambda _: None)
    spec = GdeltLongRunSpec(
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 2),
        query_types=(GdeltQueryType.ALL_MARKET,),
        raw_dir=raw_dir,
        interim_dir=interim_dir,
    )

    with pytest.raises(GdeltClientError) as exc_info:
        collect_gdelt_longrun(spec, client=client)

    assert (raw_dir / "doc_daily" / "all_us_market" / "2024" / "20240101.json").exists()
    assert not (raw_dir / "gdelt_all_us_market_20240101_20240102.json").exists()
    assert exc_info.value.completed_days == 1
    assert exc_info.value.next_missing_day == date(2024, 1, 2)
    assert "collect_gdelt_longrun" in (exc_info.value.resume_guidance or "")


def test_longrun_never_overwrites_raw_aggregate_output(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    interim_dir = tmp_path / "interim"
    raw_dir.mkdir()
    aggregate = raw_dir / "gdelt_all_us_market_20240101_20240101.json"
    aggregate.write_text("{}", encoding="utf-8")
    spec = GdeltLongRunSpec(
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 1),
        query_types=(GdeltQueryType.ALL_MARKET,),
        raw_dir=raw_dir,
        interim_dir=interim_dir,
        overwrite_interim=True,
    )

    with pytest.raises(FileExistsError, match="Raw aggregate outputs are never overwritten"):
        collect_gdelt_longrun(spec, client=GdeltClient(session=FakeSession([]), max_retries=0))

    assert aggregate.read_text(encoding="utf-8") == "{}"


def test_collect_gkg_bulk_extract_writes_filtered_rows_and_manifest(tmp_path: Path) -> None:
    archive_url = "https://data.gdeltproject.org/gdeltv2/20240101000000.gkg.csv.zip"
    master_url = "https://example.com/masterfilelist.txt"
    zip_content = _zip_bytes(
        "20240101000000.gkg.csv",
        "1\tUS stock market rally\n2\tunrelated row\n3\tAI bubble warning\n",
    )
    session = FakeGkgSession(
        {
            master_url: FakeResponse(200, {}, text=f"1 2 {archive_url}\n"),
            archive_url: FakeResponse(200, {}, content=zip_content),
        }
    )
    spec = GkgBulkSpec(
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 1),
        filter_terms=("stock market", "bubble"),
        raw_dir=tmp_path / "gkg",
        masterfilelist_url=master_url,
    )

    result = collect_gkg_bulk_extract(spec, session=session)

    assert result.archive_count == 1
    assert result.matched_row_count == 2
    with result.extract_path.open(encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert [row["row_text"] for row in rows] == ["1\tUS stock market rally", "3\tAI bubble warning"]
    with result.manifest_path.open(encoding="utf-8", newline="") as csv_file:
        manifest = list(csv.DictReader(csv_file))
    assert manifest[0]["matched_rows"] == "2"
    assert session.calls == [master_url, archive_url]


def _zip_bytes(name: str, text: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(name, text)
    return buffer.getvalue()
