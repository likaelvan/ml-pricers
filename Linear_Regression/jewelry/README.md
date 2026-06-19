# Jewelry Linear Regression Baselines

This folder contains jewelry-specific Ridge and Huber regression baselines adapted from the artwork and watch linear-regression workflows.

## Data Used

Training source:

- `Data/sampled_sets_jewelry/Jewelry_2025_high_midpoint_error_remainder_training.csv`

Quarterly 2025 holdout sources:

- `Data/sampled_sets_jewelry/Jewelry_2025_high_midpoint_error_q2025Q1_sample1000.csv`
- `Data/sampled_sets_jewelry/Jewelry_2025_high_midpoint_error_q2025Q2_sample1000.csv`
- `Data/sampled_sets_jewelry/Jewelry_2025_high_midpoint_error_q2025Q3_sample1000.csv`
- `Data/sampled_sets_jewelry/Jewelry_2025_high_midpoint_error_q2025Q4_sample1000.csv`

The provided training file is treated as the training pool by default. It spans 2024-2026, so the scripts expose `--train-before` and the notebooks expose `TRAIN_BEFORE` if a strict date cutoff is needed. The default is no date cutoff because `sale_date < 2025-01-01` leaves only a small pre-2025 training set.

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

- Loads the provided jewelry training file.
- Loads explicit 2025 quarterly test rows from the Q1-Q4 jewelry files.
- De-duplicates train and test rows by non-missing `lot_url`.
- Removes any training row whose `lot_url` appears in the Q1-Q4 test sets.
- Optionally samples train rows or per-quarter test rows for quick experiments.

## Dictionary Flattening

Several jewelry columns are stringified dictionaries. The feature pipeline parses these with `ast.literal_eval` only when the value looks dictionary-like. If parsing fails, the value is treated as missing for the flattened fields.

Flattened columns include:

```text
brand_and_model_identifiers
primary_metal_specification
pearl_specification
primary_gemstone_specification
diamond_set_specification
inclusions_and_certification
```

Exact certification report numbers are intentionally excluded from model features.

## Feature Engineering

Categorical jewelry features:

```text
auction_house
location
product_name_clean
brand_clean
collection_clean
model_clean
reference_clean
maker_signatures_clean
metal_type_clean
metal_purity_clean
metal_color_clean
jewelry_category_clean
pearl_type_clean
pearl_color_clean
pearl_luster_clean
pearl_surface_quality_clean
gemstone_type_clean
gemstone_shape_clean
gemstone_color_clean
gemstone_clarity_clean
gemstone_origin_clean
gemstone_treatment_clean
diamond_color_clean
diamond_clarity_clean
diamond_cut_clean
design_style_clean
enamel_color_clean
certification_lab_clean
country_of_origin_clean
```

Numeric jewelry features:

```text
sale_date_ordinal
sale_year_num
sale_month_num
total_item_weight_num
ring_size_num
necklace_length_num
bracelet_length_num
earring_length_num
pendant_length_num
chain_length_num
gemstone_count_num
diamond_count_num
pearl_count_num
item_count_num
primary_gemstone_carat_num
diamond_total_carat_num
diamond_count_from_spec_num
pearl_diameter_num
pearl_count_from_spec_num
GSPC
FTSE
IXIC
DJI
VIX
BZ_F
FVX
```

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

- `Jewelry Ridge - No Estimates`
- `Jewelry Ridge - With Log Estimates`

The Huber workflow trains two variants:

- `Jewelry Huber - No Estimates`
- `Jewelry Huber - With Log Estimates`

Both methods use the same preprocessing and feature schema logic. Rare categorical levels are grouped based only on the training set using `min_category_count`, and numeric medians are fit only on the training set before being applied to test rows.

The Huber script defaults to `--max-iter 1000` because the feature matrix can be wide after one-hot encoding.

## Run

Notebook entrypoints:

- `Ridge_Regression/run_jewelry_ridge_regression_pricer.ipynb`
- `Huber_Regression/run_jewelry_huber_regression_pricer.ipynb`

The notebooks explicitly show `JEWELRY_DATA_DIR`, `TRAIN_FILE`, `TEST_FILES`, `OUTPUT_DIR`, and `TARGET_COL` before loading data.

Script entrypoints from the workspace root:

```bash
python Linear_Regression/jewelry/Ridge_Regression/run_jewelry_ridge_regression_pricer.py --verbose
```

```bash
python Linear_Regression/jewelry/Huber_Regression/run_jewelry_huber_regression_pricer.py --verbose
```

For a quick smoke test:

```bash
python Linear_Regression/jewelry/Ridge_Regression/run_jewelry_ridge_regression_pricer.py \
  --train-rows 500 \
  --test-rows-per-quarter 25 \
  --output-dir Linear_Regression/jewelry/Ridge_Regression/outputs/smoke_test \
  --verbose
```

```bash
python Linear_Regression/jewelry/Huber_Regression/run_jewelry_huber_regression_pricer.py \
  --train-rows 500 \
  --test-rows-per-quarter 25 \
  --output-dir Linear_Regression/jewelry/Huber_Regression/outputs/smoke_test \
  --verbose
```

## Outputs

Ridge outputs:

- `Ridge_Regression/outputs/jewelry_ridge_test_predictions.csv`
- `Ridge_Regression/outputs/jewelry_ridge_overall_metrics.csv`
- `Ridge_Regression/outputs/jewelry_ridge_quarter_metrics.csv`
- `Ridge_Regression/outputs/jewelry_ridge_price_bucket_metrics.csv`
- `Ridge_Regression/outputs/jewelry_ridge_feature_report_no_estimates.csv`
- `Ridge_Regression/outputs/jewelry_ridge_feature_report_with_estimates.csv`
- `Ridge_Regression/outputs/jewelry_ridge_training_history.csv`
- `Ridge_Regression/outputs/jewelry_ridge_top_coefficients.csv`

Huber outputs:

- `Huber_Regression/outputs/jewelry_huber_test_predictions.csv`
- `Huber_Regression/outputs/jewelry_huber_overall_metrics.csv`
- `Huber_Regression/outputs/jewelry_huber_quarter_metrics.csv`
- `Huber_Regression/outputs/jewelry_huber_price_bucket_metrics.csv`
- `Huber_Regression/outputs/jewelry_huber_feature_report_no_estimates.csv`
- `Huber_Regression/outputs/jewelry_huber_feature_report_with_estimates.csv`
- `Huber_Regression/outputs/jewelry_huber_training_history.csv`
- `Huber_Regression/outputs/jewelry_huber_top_coefficients.csv`
