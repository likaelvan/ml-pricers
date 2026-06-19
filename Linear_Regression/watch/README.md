# Watch Linear Regression Baselines

This folder contains watch-specific Ridge and Huber regression baselines adapted from the artwork linear-regression workflow.

## Data Used

Training source:

- `Data/watch/Watch_2026_finetune_19345.json`

Quarterly 2025 holdout sources:

- `Data/watch/watch_sales_2025Q1_dedup_lot_url.json`
- `Data/watch/watch_sales_2025Q2_dedup_lot_url.json`
- `Data/watch/watch_sales_2025Q3_dedup_lot_url.json`
- `Data/watch/watch_sales_2025Q4_dedup_lot_url.json`

The training file currently includes rows through 2025. The run scripts default to filtering training rows with:

```text
sale_date < 2025-01-01
```

This keeps the 2025 Q1-Q4 files as true holdout sets and avoids training/test leakage.

## Target

Default target:

```text
ah_price_usd
```

The models train on:

```text
log1p(ah_price_usd)
```

Predictions are transformed back to USD with:

```text
expm1(predicted_log_price)
```

Only positive, finite target rows are kept.

## Split and Leakage Controls

The pipeline:

- Loads historical training rows from `Watch_2026_finetune_19345.json`.
- Loads explicit 2025 quarterly test rows from the Q1-Q4 watch files.
- De-duplicates train and test rows by non-missing `lot_url`.
- Removes any training row whose `lot_url` appears in the Q1-Q4 test sets.
- Optionally samples train rows or per-quarter test rows for quick experiments.

## Feature Engineering

Categorical watch features:

```text
auction_house
location
brand_clean
model_clean
ref_no_clean
box_clean
papers_clean
condition_clean
case_material_clean
band_material_clean
dial_color_clean
gold_present_clean
gold_type_clean
diamonds_clean
```

Boolean-like fields such as `box`, `papers`, `gold_present`, and `diamonds` are normalized to:

```text
yes
no
__missing__
__other__
```

They are then treated as categorical features and one-hot encoded. This is intentional because source values can arrive as strings, booleans, numeric flags, or missing values.

Numeric watch features:

```text
sale_date_ordinal
sale_year_num
sale_month_num
year_manufactured_num
case_size_num
karats_num
diamonds_quantity_num
GSPC
FTSE
IXIC
DJI
VIX
BZ_F
FVX
```

The numeric fields are coerced from raw source columns. For fields such as `case_size`, the first numeric token is extracted.

Estimate-aware features:

```text
log_auction_low_estimate_usd
log_auction_high_estimate_usd
log_auction_estimate_midpoint_usd
auction_estimate_range_ratio
```

The raw low/high estimates and midpoint are retained for reporting, but only the transformed estimate features are used in the with-estimates models.

## Methods

The Ridge workflow trains two variants:

- `Watch Ridge - No Estimates`
- `Watch Ridge - With Log Estimates`

The Huber workflow trains two variants:

- `Watch Huber - No Estimates`
- `Watch Huber - With Log Estimates`

Both methods use the same preprocessing and feature schema logic. Rare categorical levels are grouped based only on the training set using `min_category_count`, and numeric medians are fit only on the training set before being applied to test rows.

The Huber script defaults to `--max-iter 1000` because the watch feature matrix can be wider than the artwork regression matrix after one-hot encoding.

## Run

Notebook entrypoints:

- `Ridge_Regression/run_watch_ridge_regression_pricer.ipynb`
- `Huber_Regression/run_watch_huber_regression_pricer.ipynb`

The notebooks expose the full training and testing flow step by step: configuration, imports, data loading, train/test leakage checks, feature engineering, model fitting, evaluation, and output writing.

Script entrypoints from the workspace root:

```bash
python Linear_Regression/watch/Ridge_Regression/run_watch_ridge_regression_pricer.py --verbose
```

```bash
python Linear_Regression/watch/Huber_Regression/run_watch_huber_regression_pricer.py --verbose
```

For a quick smoke test:

```bash
python Linear_Regression/watch/Ridge_Regression/run_watch_ridge_regression_pricer.py \
  --train-rows 500 \
  --test-rows-per-quarter 25 \
  --output-dir Linear_Regression/watch/Ridge_Regression/outputs/smoke_test \
  --verbose
```

```bash
python Linear_Regression/watch/Huber_Regression/run_watch_huber_regression_pricer.py \
  --train-rows 500 \
  --test-rows-per-quarter 25 \
  --output-dir Linear_Regression/watch/Huber_Regression/outputs/smoke_test \
  --verbose
```

## Outputs

Ridge outputs:

- `Ridge_Regression/outputs/watch_ridge_test_predictions.csv`
- `Ridge_Regression/outputs/watch_ridge_overall_metrics.csv`
- `Ridge_Regression/outputs/watch_ridge_quarter_metrics.csv`
- `Ridge_Regression/outputs/watch_ridge_price_bucket_metrics.csv`
- `Ridge_Regression/outputs/watch_ridge_feature_report_no_estimates.csv`
- `Ridge_Regression/outputs/watch_ridge_feature_report_with_estimates.csv`
- `Ridge_Regression/outputs/watch_ridge_training_history.csv`
- `Ridge_Regression/outputs/watch_ridge_top_coefficients.csv`

Huber outputs:

- `Huber_Regression/outputs/watch_huber_test_predictions.csv`
- `Huber_Regression/outputs/watch_huber_overall_metrics.csv`
- `Huber_Regression/outputs/watch_huber_quarter_metrics.csv`
- `Huber_Regression/outputs/watch_huber_price_bucket_metrics.csv`
- `Huber_Regression/outputs/watch_huber_feature_report_no_estimates.csv`
- `Huber_Regression/outputs/watch_huber_feature_report_with_estimates.csv`
- `Huber_Regression/outputs/watch_huber_training_history.csv`
- `Huber_Regression/outputs/watch_huber_top_coefficients.csv`
