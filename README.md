# 📈 Market Criticism Index

This repository will implement a reproducible research pipeline for studying whether criticism narratives about the US equity market are related to later market returns, volatility, drawdowns, or reversals.

The current state includes the MVP scaffold, an initial GDELT data collection layer, and headline cleaning/deduplication utilities. It does not yet implement annotation sample export, index construction, market-data merging, or empirical modelling.

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
│       ├── gdelt.py
│       ├── index.py
│       ├── market_data.py
│       ├── modelling.py
│       ├── plotting.py
│       └── text_processing.py
├── scripts/
│   └── collect_gdelt.py
└── tests/
    ├── test_config.py
    ├── test_gdelt.py
    ├── test_imports.py
    ├── test_market_data.py
    └── test_text_processing.py
```

## 🗂️ Data And Output Conventions

Raw data should never be overwritten. Generated headline, annotation, and modelling datasets should be written to `data/interim/` or `data/processed/`. Figures belong in `outputs/figures/`; regression tables belong in `outputs/tables/`.

The scaffold exposes these paths in `mci.config` so future implementation can keep outputs deterministic.

## 🧹 Cleaning And Alignment Assumptions

Headline titles are normalised for matching by lowercasing, replacing separator punctuation with spaces, removing other punctuation, and collapsing whitespace. Raw title text is preserved on cleaned record copies.

GDELT-style naive `seendate` timestamps with full time components are treated as UTC and converted to `America/New_York`. Date-only values are left unaligned because they are insufficient for the 16:00 New York close rule. Items seen after 16:00 New York time are assigned to the next trading day.

Generated cleaning and alignment outputs are rejected if they point under `data/raw/`. Alignment outputs default to `data/interim/`. When an explicit market calendar is supplied, alignment validates calendar coverage before writing; otherwise it falls back to weekdays only.

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

Collect one month of all US-market-related GDELT headlines:

```bash
python scripts/collect_gdelt.py --query-type all_us_market --start-date 2022-01-01 --end-date 2022-01-31
```
