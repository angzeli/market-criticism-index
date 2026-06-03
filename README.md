# 📈 Market Criticism Index

This repository will implement a reproducible research pipeline for studying whether criticism narratives about the US equity market are related to later market returns, volatility, drawdowns, or reversals.

The current state includes the MVP scaffold, an initial GDELT data collection layer, headline cleaning/deduplication utilities, the K annotation sample/validation workflow, daily Market Criticism Index construction, market-data panel merging, and MVP empirical analysis. Robustness extensions and source-stratified modelling remain future work.

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
├── docs/
│   └── k_annotation_workflow.md
├── notebooks/
│   ├── 00_data_collection_runbook.ipynb
│   └── 01_market_data_collection_runbook.ipynb
├── outputs/
│   ├── figures/
│   └── tables/
├── src/
│   └── mci/
│       ├── __init__.py
│       ├── annotations.py
│       ├── config.py
│       ├── data_collection.py
│       ├── gdelt.py
│       ├── index.py
│       ├── market_data.py
│       ├── modelling.py
│       ├── plotting.py
│       └── text_processing.py
├── scripts/
│   ├── build_annotation_sample.py
│   ├── build_mci_daily.py
│   ├── build_panel_daily.py
│   ├── collect_gdelt.py
│   ├── run_mvp_analysis.py
│   └── validate_annotations.py
└── tests/
    ├── test_annotations.py
    ├── test_config.py
    ├── test_gdelt.py
    ├── test_index.py
    ├── test_imports.py
    ├── test_market_data.py
    ├── test_market_panel.py
    ├── test_modelling.py
    └── test_text_processing.py
```

## 🗂️ Data And Output Conventions

Raw data should never be overwritten. Generated headline, annotation, and modelling datasets should be written to `data/interim/` or `data/processed/`. Figures belong in `outputs/figures/`; regression tables belong in `outputs/tables/`.

The scaffold exposes these paths in `mci.config` so future implementation can keep outputs deterministic.

## 📒 Interactive Data Collection Runbooks

Use [notebooks/00_data_collection_runbook.ipynb](notebooks/00_data_collection_runbook.ipynb) for interactive GDELT/headline collection and lightweight previews. Use [notebooks/01_market_data_collection_runbook.ipynb](notebooks/01_market_data_collection_runbook.ipynb) for interactive SPY, QQQ, RSP, and VIX market-data collection and panel previews. Both notebooks call existing package functions and do not contain core pipeline logic.

The GDELT notebook uses resumable daily DOC checkpoints under `data/raw/gdelt/doc_daily/`, so interrupted long runs can be restarted without losing completed days. It also includes an optional GKG filtered-extract fallback section for audit/candidate discovery; those outputs are labelled separately from DOC headline CSVs.

The market-data notebook writes raw market prices under `data/raw/market/` and can build the merged panel at `data/processed/panel_daily.csv` when `data/processed/mci_daily.csv` is available. Raw market CSVs are reused when present and are never overwritten.

Command-line scripts remain the reproducible path for pipeline runs.

## 🧹 Cleaning And Alignment Assumptions

Headline titles are normalised for matching by lowercasing, replacing separator punctuation with spaces, removing other punctuation, and collapsing whitespace. Raw title text is preserved on cleaned record copies.

GDELT-style naive `seendate` timestamps with full time components are treated as UTC and converted to `America/New_York`. Date-only values are left unaligned because they are insufficient for the 16:00 New York close rule. Items seen after 16:00 New York time are assigned to the next trading day.

Generated cleaning and alignment outputs are rejected if they point under `data/raw/`. Alignment outputs default to `data/interim/`. When an explicit market calendar is supplied, alignment validates calendar coverage before writing; otherwise it falls back to weekdays only.

## 📝 K Annotation Workflow

Use the [K annotation workflow guide](docs/k_annotation_workflow.md) for sample locations, editable columns, accepted label values, and validation steps.

Build a sample from cleaned candidate and all-market headline CSVs:

```bash
python scripts/build_annotation_sample.py --candidate-csv data/interim/gdelt_candidate_criticism_20220101_20260531.csv --all-market-csv data/interim/gdelt_all_us_market_20220101_20260531.csv --seed 1
```

Validate completed labels after K saves CSVs under `data/processed/annotations/labelled/`:

```bash
python scripts/validate_annotations.py
```

## 📊 MCI Calculation

Daily MCI construction expects cleaned, date-aligned all-market and candidate-criticism headline CSVs. Date resolution uses `trading_day`, then `date`, then `published_date_ny`.

The daily index is:

```text
MCI = raw_criticism_count / total_market_article_count
```

The output has one row per observed all-market headline date, sorted ascending. If `total_market_article_count` is zero, `MCI` is left blank.

The rolling z-score column is current-inclusive and named for the configured window, for example:

```text
mci_rolling_60d_zscore = (MCI - rolling_mean_60d) / rolling_std_60d
```

The rolling z-score uses a strict valid-observation window. It is blank until the full window is available, and it remains blank when the rolling standard deviation is zero.

Input rows with missing or unparseable dates fail fast. Candidate counts are also checked against all-market counts by date so `MCI` does not exceed `1.0`.

When completed labels are supplied, matched `criticism_label = 0` candidates are excluded and matched `criticism_label = 1` candidates are counted. Matched labels must be resolved to `0` or `1`, and conflicting duplicate labels are rejected. Unmatched candidate rows still count as automated candidates. Category-specific columns are added only from matched positive labels with configured category labels, so they reflect labelled category coverage rather than automated category classification.

Build the daily MCI CSV:

```bash
python scripts/build_mci_daily.py --market-csv data/interim/gdelt_all_us_market_20220101_20260531.csv --criticism-csv data/interim/gdelt_candidate_criticism_20220101_20260531.csv --labels data/processed/annotations/labelled/ --overwrite
```

## 📈 Market Data And Panel Merge

Daily panel construction expects `data/processed/mci_daily.csv` and a normalized long market-price CSV with required columns `date`, `symbol`, and `close`. The optional `adj_close` column is used for `SPY`, `QQQ`, and `RSP` when available; VIX features use `close`. Input symbols `VIX` and `^VIX` both map to the `vix` output prefix.

Default symbols are `SPY`, `QQQ`, `RSP`, and `^VIX`. Default horizons are `1`, `5`, and `21` observed trading rows. The default realised-volatility window is `21` rows.

ETF forward and lagged returns use log returns:

```text
fwd_log_return_h = log(price[t+h] / price[t])
lag_log_return_h = log(price[t] / price[t-h])
```

Realised volatility is the trailing standard deviation of 1-day log returns, annualized by `sqrt(252)`, and is blank until the full window is available. Lagged realised volatility is the prior-day realised-volatility value.

Forward realised volatility uses the annualized root mean square of the next `h` one-day log returns:

```text
fwd_realized_vol_h = sqrt(mean(next_h_1d_log_returns^2)) * sqrt(252)
```

Forward max drawdown uses the full forward horizon:

```text
fwd_max_drawdown_h = min(0, min(price[t+1:t+h] / price[t] - 1))
```

VIX forward changes are level changes:

```text
vix_fwd_change_h = VIX[t+h] - VIX[t]
```

The panel keeps all MCI rows and merges market features by `date`. It fails if an MCI date lacks current-day market prices, while trailing forward-looking features may be blank near the end of the market sample. The default output is `data/processed/panel_daily.csv`.

Build the daily panel:

```bash
python scripts/build_panel_daily.py --prices-csv data/raw/market/market_prices_spy_qqq_rsp_vix_20220101_20260531.csv --overwrite
```

## 📉 MVP Empirical Analysis

MVP empirical analysis runs from `data/processed/panel_daily.csv`. These estimates are correlational and should not be interpreted as causal effects.

The event study selects nonmissing `mci_rolling_60d_zscore` days in the top decile and aligns SPY cumulative log returns from `-10` to `+21` observed trading days around each event. Events with incomplete windows are dropped. Outputs are:

- `outputs/figures/event_study_top_decile_mci_spy.png`
- `outputs/tables/event_study_top_decile_mci_spy.csv`

Baseline regressions run horizons `1`, `5`, and `21` with OLS and Newey-West/HAC standard errors using `maxlags = horizon - 1`.

Return model:

```text
spy_fwd_log_return_hd ~ mci_rolling_60d_zscore + spy_lag_log_return_hd + vix_level + spy_lag_realized_vol_21d
```

Future volatility model:

```text
spy_fwd_realized_vol_hd ~ mci_rolling_60d_zscore + spy_lag_realized_vol_21d + vix_level
```

Future drawdown model:

```text
spy_fwd_max_drawdown_hd ~ mci_rolling_60d_zscore + spy_lag_log_return_hd + vix_level + spy_lag_realized_vol_21d
```

Regression outputs are:

- `outputs/tables/baseline_regressions.csv`
- `outputs/tables/baseline_regressions.md`

Run the MVP empirical analysis:

```bash
python scripts/run_mvp_analysis.py --overwrite
```

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

## 👥 Authors

- Angze Li
- Shengkai Zhang
