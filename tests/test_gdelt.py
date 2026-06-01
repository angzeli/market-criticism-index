"""Tests for mocked GDELT collection."""

from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from mci.data_collection import HeadlineCollectionSpec, collect_market_headlines
from mci.gdelt import (
    GDELT_ARTICLE_FIELDS,
    GdeltClient,
    GdeltClientError,
    GdeltQueryType,
    GdeltRequestSpec,
    build_gdelt_query,
    collect_gdelt_headlines,
)


class FakeResponse:
    def __init__(self, status_code: int, data: dict[str, Any], text: str = "") -> None:
        self.status_code = status_code
        self._data = data
        self.text = text

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
