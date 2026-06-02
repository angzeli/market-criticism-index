# K Annotation Workflow

This guide explains how to use the manual annotation CSVs for the Market Criticism Index MVP.

## Where Files Go

Annotation samples for K are generated under:

```text
data/interim/annotations/samples/
```

Completed labelled files should be saved under:

```text
data/processed/annotations/labelled/
```

Raw input data should not be edited or overwritten.

## Columns K Should Not Change

These columns provide item identity, headline metadata, source provenance, and sampling provenance:

```text
id
date
title
source
domain
url
query_type
sample_stratum
```

`query_type` records where the item came from, such as `candidate_criticism` or `all_us_market`.

`sample_stratum` records why the item was included in the annotation sample. It is sampling provenance only; it is not a criticism label.

The `ambiguous` stratum supports annotation coverage for difficult cases. It does not redefine what counts as criticism.

## Columns K Should Edit

K should fill in only these manual label columns:

```text
market_relevant_label
criticism_label
category_label
intensity_score
annotator_notes
```

Accepted values:

- `market_relevant_label`: use `1` for yes, `0` for no, or leave blank if unresolved.
- `criticism_label`: use `1` for yes, `0` for no, or leave blank if unresolved.
- `category_label`: use one of the configured MVP categories when `criticism_label` is `1`.
- `intensity_score`: use a numeric `0` to `3` scale, where `0` means no criticism and `3` means strong criticism.
- `annotator_notes`: optional short notes for ambiguous or difficult cases.

Leave `category_label` and `intensity_score` blank when `criticism_label` is `0` or unresolved.

Configured MVP categories:

```text
valuation
bubble_speculation
crash_correction_warning
ai_tech_hype
concentration
```

## Recommended Labelling Sequence

1. Check whether the headline is about the US equity market or a major US equity benchmark.
2. Set `market_relevant_label` to `1` or `0`.
3. Decide whether the headline expresses criticism under the project codebook.
4. Set `criticism_label` to `1` or `0`.
5. If `criticism_label` is `1`, add `category_label` and `intensity_score`.
6. Add `annotator_notes` only when the case needs explanation.

## Validation

After labelled CSVs are saved under `data/processed/annotations/labelled/`, run:

```bash
python scripts/validate_annotations.py
```

The validation report summarizes candidate-extraction precision, sample recall, category counts, intensity distribution, missing labels, and inter-annotator agreement when duplicate item IDs appear across labelled files.

Validation fails when no labelled CSVs are found. Use `--allow-empty` only for smoke checks.
