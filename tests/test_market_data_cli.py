"""Tests for the optional market-data fetch CLI."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "collect_market_data.py"


def _load_collect_market_data_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("collect_market_data_script", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_collect_market_data_cli_does_not_preflight_by_default(tmp_path: Path, monkeypatch) -> None:
    module = _load_collect_market_data_script()
    calls: list[str] = []
    output_path = tmp_path / "market_prices_spy_20240102_20240103.csv"

    def fake_preflight(_spec):
        calls.append("preflight")
        raise AssertionError("preflight should be opt-in")

    def fake_collect(_spec):
        calls.append("collect")
        return output_path

    monkeypatch.setattr(module, "preflight_market_data_provider", fake_preflight)
    monkeypatch.setattr(module, "collect_market_data", fake_collect)

    result = module.main(
        [
            "--start-date",
            "2024-01-02",
            "--end-date",
            "2024-01-03",
            "--symbols",
            "SPY",
            "--raw-output-dir",
            str(tmp_path),
        ]
    )

    assert result == 0
    assert calls == ["collect"]


def test_collect_market_data_cli_defaults_match_market_data_spec(tmp_path: Path, monkeypatch) -> None:
    module = _load_collect_market_data_script()
    captured_specs = []
    output_path = tmp_path / "market_prices_spy_20240102_20240103.csv"

    def fake_collect(spec):
        captured_specs.append(spec)
        return output_path

    monkeypatch.setattr(module, "collect_market_data", fake_collect)

    result = module.main(
        [
            "--start-date",
            "2024-01-02",
            "--end-date",
            "2024-01-03",
            "--symbols",
            "SPY",
            "--raw-output-dir",
            str(tmp_path),
        ]
    )

    assert result == 0
    assert captured_specs[0].max_retries == module._market_data_default("max_retries")
    assert captured_specs[0].request_pause_seconds == module._market_data_default("request_pause_seconds")


def test_collect_market_data_cli_runs_preflight_when_requested(tmp_path: Path, monkeypatch) -> None:
    module = _load_collect_market_data_script()
    calls: list[str] = []
    output_path = tmp_path / "market_prices_spy_20240102_20240103.csv"

    def fake_preflight(_spec):
        calls.append("preflight")

    def fake_collect(_spec):
        calls.append("collect")
        return output_path

    monkeypatch.setattr(module, "preflight_market_data_provider", fake_preflight)
    monkeypatch.setattr(module, "collect_market_data", fake_collect)

    result = module.main(
        [
            "--start-date",
            "2024-01-02",
            "--end-date",
            "2024-01-03",
            "--symbols",
            "SPY",
            "--raw-output-dir",
            str(tmp_path),
            "--preflight",
        ]
    )

    assert result == 0
    assert calls == ["preflight", "collect"]
