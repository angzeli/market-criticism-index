# 📈 Market Criticism Index

This repository will implement a reproducible research pipeline for studying whether criticism narratives about the US equity market are related to later market returns, volatility, drawdowns, or reversals.

The current state is an MVP scaffold only. It defines package boundaries, intended workflow, and placeholder tests. It does not yet collect data or run the empirical pipeline.

## 🧭 Research Boundary

The project keeps the discourse and quantitative workflows separate. K's annotation/codebook work defines and validates what counts as market criticism. Automated keyword, NLP, and modelling steps are supporting tools, not replacements for that definition.

The analysis should avoid causal claims. Results should be described as associations, relationships, predictive tests, or event-study patterns.

## 🔄 Intended Workflow

1. Collect US-market-related headlines from structured sources such as GDELT.
2. Collect candidate criticism headlines using transparent market and criticism keyword queries.
3. Normalise headline text and deduplicate records while preserving raw data.
4. Export a mixed annotation sample for K under `data/interim/`.
5. Use K's labelled validation set to evaluate automated candidate extraction.
6. Build daily Market Criticism Index variables and category-specific measures.
7. Merge daily index variables with SPY, QQQ, RSP, and VIX market data.
8. Run baseline event studies and regressions with interpretable controls.
9. Save figures to `outputs/figures/` and regression tables to `outputs/tables/`.

## 🌳 Repository Filetree

```text
market-criticism-index/
├── .gitignore
├── AGENTS.md
├── LICENSE
├── README.md
├── critisism_study_research_brief-2.pdf
├── pyproject.toml
├── data/
│   ├── raw/
│   │   ├── gdelt/
│   │   └── market/
│   ├── interim/
│   └── processed/
├── outputs/
│   ├── figures/
│   └── tables/
├── src/
│   └── mci/
│       ├── __init__.py
│       ├── config.py
│       ├── data_collection.py
│       ├── index.py
│       ├── market_data.py
│       ├── modelling.py
│       ├── plotting.py
│       └── text_processing.py
└── tests/
    ├── test_config.py
    └── test_imports.py
```

## 🗂️ Data And Output Conventions

Raw data should never be overwritten. Generated headline, annotation, and modelling datasets should be written to `data/interim/` or `data/processed/`. Figures belong in `outputs/figures/`; regression tables belong in `outputs/tables/`.

The scaffold exposes these paths in `mci.config` so future implementation can keep outputs deterministic.

## 🛠️ Development

Install development dependencies when needed:

```bash
python -m pip install -e ".[dev]"
```

Install analysis dependencies when implementing the full pipeline:

```bash
python -m pip install -e ".[analysis,dev]"
```

Run tests:

```bash
pytest
```
