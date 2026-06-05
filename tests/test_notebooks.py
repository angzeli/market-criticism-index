"""Lightweight structure checks for project runbook notebooks."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _notebook_source(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def _read_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _code_cell_source(path: Path, marker: str) -> str:
    notebook = _read_notebook(path)
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if marker in source:
            return source
    raise AssertionError(f"No code cell containing {marker!r}.")


def test_market_data_collection_runbook_is_thin_package_caller() -> None:
    notebook_path = PROJECT_ROOT / "notebooks" / "01_market_data_collection_runbook.ipynb"

    assert notebook_path.exists()

    source = _notebook_source(notebook_path)

    assert "collect_market_data" in source
    assert "build_market_panel" in source
    assert "market_data_output_path" in source
    assert "MARKET_PRICES_CSV" in source
    assert "data/raw/market" in source

    forbidden_core_logic = (
        "def collect_market_data",
        "def build_market_panel",
        "requests.get",
        "requests.Session",
        "YAHOO_DOWNLOAD_ENDPOINT",
        "yfinance",
    )
    for token in forbidden_core_logic:
        assert token not in source


def test_runbook_notebooks_have_no_stored_outputs_execution_counts_or_execute_time() -> None:
    for notebook_path in (
        PROJECT_ROOT / "notebooks" / "00_data_collection_runbook.ipynb",
        PROJECT_ROOT / "notebooks" / "01_market_data_collection_runbook.ipynb",
    ):
        notebook = _read_notebook(notebook_path)
        for cell in notebook["cells"]:
            assert cell.get("outputs", []) == []
            assert cell.get("execution_count") is None
            assert "ExecuteTime" not in cell.get("metadata", {})


def test_market_data_runbook_defaults_to_retry_without_preflight() -> None:
    notebook_path = PROJECT_ROOT / "notebooks" / "01_market_data_collection_runbook.ipynb"
    namespace: dict[str, object] = {}
    source = _notebook_source(notebook_path)

    exec(_code_cell_source(notebook_path, "START_DATE ="), namespace)

    assert namespace["MARKET_MAX_RETRIES"] == 5
    assert namespace["MARKET_REQUEST_PAUSE_SECONDS"] == 5.0
    assert namespace["RUN_MARKET_PREFLIGHT"] is False
    assert "if RUN_MARKET_PREFLIGHT:" in source
    assert "preflight_market_data_provider(market_spec)" in source


def test_market_data_collection_runbook_setup_works_outside_repo_cwd(tmp_path: Path, monkeypatch) -> None:
    notebook_path = PROJECT_ROOT / "notebooks" / "01_market_data_collection_runbook.ipynb"
    namespace: dict[str, object] = {}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MCI_PROJECT_ROOT", str(PROJECT_ROOT))

    exec(_code_cell_source(notebook_path, "START_DATE ="), namespace)
    exec(_code_cell_source(notebook_path, "project_root_candidates"), namespace)

    assert namespace["PROJECT_ROOT"] == PROJECT_ROOT


def test_gdelt_collection_runbook_setup_works_outside_repo_cwd(tmp_path: Path, monkeypatch) -> None:
    notebook_path = PROJECT_ROOT / "notebooks" / "00_data_collection_runbook.ipynb"
    namespace: dict[str, object] = {}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MCI_PROJECT_ROOT", str(PROJECT_ROOT))

    exec(_code_cell_source(notebook_path, "START_DATE ="), namespace)
    exec(_code_cell_source(notebook_path, "def find_project_root"), namespace)

    assert namespace["PROJECT_ROOT"] == PROJECT_ROOT
    assert namespace["SOURCE_LANGUAGE"] == "english"
    assert namespace["RUN_MARKET_DATA"] is False
