#!/usr/bin/env python3
"""Run the MVP empirical analysis from a daily panel CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mci.config import FIGURES_DIR, PROCESSED_DATA_DIR, TABLES_DIR
from mci.modelling import CORRELATIONAL_WARNING, ModelSpec, run_mvp_analysis


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MVP event studies and baseline regressions.")
    parser.add_argument("--panel", default=PROCESSED_DATA_DIR / "panel_daily.csv", type=Path)
    parser.add_argument("--figures-dir", default=FIGURES_DIR, type=Path)
    parser.add_argument("--tables-dir", default=TABLES_DIR, type=Path)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing analysis outputs.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    spec = ModelSpec(
        panel_path=args.panel,
        figures_dir=args.figures_dir,
        tables_dir=args.tables_dir,
        overwrite=args.overwrite,
    )

    try:
        result = run_mvp_analysis(spec)
    except (FileExistsError, OSError, ValueError) as exc:
        print(f"MVP analysis did not complete: {exc}", file=sys.stderr)
        return 1

    print(CORRELATIONAL_WARNING)
    print(f"Saved event-study figure: {result.event_study.figure_path}")
    print(f"Saved event-study table: {result.event_study.table_path}")
    print(f"Event count: {result.event_study.event_count}")
    print(f"Dropped incomplete events: {result.event_study.dropped_incomplete_events}")
    print(f"Saved regression CSV: {result.regressions.csv_path}")
    print(f"Saved regression markdown: {result.regressions.markdown_path}")
    print(f"Regression rows: {result.regressions.row_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
