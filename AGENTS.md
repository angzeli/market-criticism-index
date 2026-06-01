# AGENTS.md

## Project

This repository implements a research pipeline for studying whether criticism narratives about the US stock market relate to future market performance.

## Working principles

- Do not change the research definition of "criticism" unless explicitly asked.
- Do not add causal claims in code comments, docs, or reports.
- Prefer transparent, interpretable methods over black-box modelling.
- Treat NLP sentiment scores as auxiliary features, not as the definition of criticism.
- Keep K's annotation/codebook workflow separate from automated classification.
- Never overwrite raw data.
- Save generated data to `data/interim/` or `data/processed/`.
- Save figures to `outputs/figures/`.
- Save regression tables to `outputs/tables/`.

## Python conventions

- Use `pandas`, `numpy`, `requests`, `statsmodels`, `scikit-learn`, and `matplotlib`.
- Prefer small, testable functions.
- Add unit tests for data cleaning, deduplication, date alignment, and index construction.
- Use type hints where practical.
- Avoid notebooks for core logic; notebooks should call functions from `src/mci/`.

## Done means

For any implementation task:
- relevant tests pass;
- changed files are summarised;
- assumptions are documented;
- no raw data files are modified;
- output paths are deterministic.