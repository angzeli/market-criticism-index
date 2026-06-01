"""High-level headline data collection functions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Sequence

from mci.config import INTERIM_DATA_DIR, MARKET_QUERY_TERMS, RAW_DATA_DIR
from mci.gdelt import (
    GdeltClient,
    GdeltCollectionResult,
    GdeltQueryType,
    GdeltRequestSpec,
    collect_gdelt_headlines,
)


@dataclass(frozen=True)
class HeadlineCollectionSpec:
    """Parameters for collecting market-related headline metadata."""

    start_date: date
    end_date: date
    query_terms: Sequence[str] = MARKET_QUERY_TERMS
    raw_output_dir: Path = RAW_DATA_DIR / "gdelt"
    interim_output_dir: Path = INTERIM_DATA_DIR
    max_records: int = 250
    overwrite: bool = False


def collect_market_headlines(
    spec: HeadlineCollectionSpec,
    *,
    client: GdeltClient | None = None,
) -> GdeltCollectionResult:
    """Collect market-related headlines and return output metadata."""

    request = GdeltRequestSpec(
        query_type=GdeltQueryType.ALL_MARKET,
        start_date=spec.start_date,
        end_date=spec.end_date,
        market_terms=spec.query_terms,
        raw_dir=spec.raw_output_dir,
        interim_dir=spec.interim_output_dir,
        max_records=spec.max_records,
        overwrite=spec.overwrite,
    )
    return collect_gdelt_headlines(request, client=client)


def collect_candidate_criticism_headlines(
    spec: HeadlineCollectionSpec,
    *,
    client: GdeltClient | None = None,
) -> GdeltCollectionResult:
    """Collect candidate criticism headlines and return output metadata."""

    request = GdeltRequestSpec(
        query_type=GdeltQueryType.CANDIDATE_CRITICISM,
        start_date=spec.start_date,
        end_date=spec.end_date,
        market_terms=spec.query_terms,
        raw_dir=spec.raw_output_dir,
        interim_dir=spec.interim_output_dir,
        max_records=spec.max_records,
        overwrite=spec.overwrite,
    )
    return collect_gdelt_headlines(request, client=client)
