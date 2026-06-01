"""Smoke tests for the MVP scaffold."""

from __future__ import annotations

import importlib


def test_package_imports() -> None:
    package = importlib.import_module("mci")

    assert package.__version__ == "0.1.0"


def test_scaffold_modules_import() -> None:
    module_names = [
        "mci.config",
        "mci.data_collection",
        "mci.gdelt",
        "mci.text_processing",
        "mci.market_data",
        "mci.index",
        "mci.modelling",
        "mci.plotting",
    ]

    for module_name in module_names:
        assert importlib.import_module(module_name)
