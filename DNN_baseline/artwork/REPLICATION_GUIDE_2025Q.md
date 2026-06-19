# 2025 Quarterly Evaluation Replication Guide

This document describes the full experiment workflow used in this workspace so the process is reproducible (assuming equivalent source data is available).

It is intentionally focused on procedure and configuration, not low-level code internals.

---

## 1) Scope

Goal:
- Train one tabular DL model on historical data (2022-2024).
- Evaluate on four 2025 quarterly holdout test sets (Q1-Q4).
- Run two variants:
1. with estimate features (`ah_low_estimate_usd`, `ah_high_estimate_usd`)
2. without estimate features

Primary script:
- `/srv/notebooks/[6.0] ML Pricers/DNN_baseline/quarter_train_test.py`

---

## 2) Environment

Recommended runtime:
- Python environment: `finetune_env`
- CUDA-enabled PyTorch

Useful runtime setting for CUDA memory stability:
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

---

## 3) Source Data

Primary source CSV (100k):
- `Data/Artwork_2026_finetune_100000_merged_backtest_Q1_results_on_artist_title_auction.csv`

Target column:
- `ah_bp_price_usd`

Required columns for filtering/splits:
- `sale_date`
- `ah_low_estimate_usd`
- `ah_high_estimate_usd`
- `artwork_count`
- `item_count`

---

## 4) Filtering Logic (used consistently)

Rows are kept only if:
1. `ah_low_estimate_usd` is non-null
2. `ah_high_estimate_usd` is non-null
3. `artwork_count` is in `{null, 0, 1}`
4. `item_count` is in `{null, 0, 1}`

Interpretation of count rule:
- If a count field is non-null and non-zero, it must equal `1`.

---

## 5) Dataset Construction

Output folder:
- `Data/sampled_sets_100k`

### 5.1 Train set

Train period:
- all eligible rows where `sale_date.year in {2022, 2023, 2024}`

Current file:
- `Artwork_2026_finetune_100000_2022_2024_train_all_eligible.csv`

Important:
- This train set was deduplicated and overwritten.
- Dedup key/behavior:
  - dedup by normalized `lot_url` where valid
  - plus exact-row dedup safety pass

### 5.2 Test sets (quarterly)

Test period:
- 2025 only

One file per quarter, 1000 sampled rows each:
- `Artwork_2026_finetune_100000_q2025Q1_sample1000.csv`
- `Artwork_2026_finetune_100000_q2025Q2_sample1000.csv`
- `Artwork_2026_finetune_100000_q2025Q3_sample1000.csv`
- `Artwork_2026_finetune_100000_q2025Q4_sample1000.csv`

Important:
- Quarterly test sets were resampled and overwritten to enforce no duplicate `lot_url` within each quarter sample.

---

## 6) Feature Set and Transform Rules

Model uses a strict curated feature set.

### 6.1 Core features

Categorical/text-context features:
- `auction_house`
- `location`
- `material`
- `artwork_title`
- `medium`
- `signed`
- `titled`
- `dated`
- `artist_name`
- `edition` (if effectively available)

Numeric/time/market features:
- `sale_date` (converted to numeric day index)
- `year_of_creation` (parsed to 4-digit year where possible)
- `length`, `height`, `width` (normalized to cm)
- `GSPC`, `FTSE`, `IXIC`, `DJI`, `VIX`, `BZ_F`, `FVX`

Boolean engineered flags:
- `has_provenance`
- `is_editioned`

Estimate features (optional run mode):
- `ah_low_estimate_usd`
- `ah_high_estimate_usd`

### 6.2 Dimension normalization

`unit` is only used for conversion, then removed from model inputs:
- `in -> cm` using 2.54
- `mm -> cm` using 0.1
- `ft -> cm` using 30.48
- default fallback factor 1.0

### 6.3 Target transform

Training target:
- stable `log1p` transform of non-negative target

Inference inverse:
- stable `expm1` with clipping safeguards

---

## 7) Training Protocol

Model family:
- TabM (PyTorch)

GPU:
- auto-selects CUDA when available

Early stopping setup:
- max epochs: 100 (or explicitly overridden in command)
- validation split from train set
- patience-based stopping
- best validation checkpoint restored in-memory before final test inference

Validation handling:
- batched validation forward pass to reduce OOM risk

---

## 8) Canonical Commands

All commands below use absolute paths for reproducibility.

### 8.1 Train once WITH estimates (save checkpoint)

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python /srv/notebooks/[6.0]\ ML\ Pricers/DNN_baseline/quarter_train_test.py \
  --use-estimates \
  --train-file /srv/notebooks/[6.0]\ ML\ Pricers/DNN_baseline/Data/sampled_sets_100k/Artwork_2026_finetune_100000_2022_2024_train_all_eligible.csv \
  --test-file /srv/notebooks/[6.0]\ ML\ Pricers/DNN_baseline/Data/sampled_sets_100k/Artwork_2026_finetune_100000_q2025Q1_sample1000.csv \
  --output-dir /srv/notebooks/[6.0]\ ML\ Pricers/DNN_baseline/outputs/quarter_eval_100k_with_est_es/q2025Q1 \
  --save-model-path /srv/notebooks/[6.0]\ ML\ Pricers/DNN_baseline/outputs/quarter_eval_100k_with_est_es/model_with_estimates.pt \
  --epochs 60 \
  --early-stopping-patience 6 \
  --val-fraction 0.2 \
  --batch-size 512 \
  --k 64 \
  --verbose
```

### 8.2 Evaluate all quarters WITH estimates (no retraining)

```bash
for q in Q1 Q2 Q3 Q4; do
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python /srv/notebooks/[6.0]\ ML\ Pricers/DNN_baseline/quarter_train_test.py \
    --use-estimates \
    --train-file /srv/notebooks/[6.0]\ ML\ Pricers/DNN_baseline/Data/sampled_sets_100k/Artwork_2026_finetune_100000_2022_2024_train_all_eligible.csv \
    --test-file /srv/notebooks/[6.0]\ ML\ Pricers/DNN_baseline/Data/sampled_sets_100k/Artwork_2026_finetune_100000_q2025${q}_sample1000.csv \
    --output-dir /srv/notebooks/[6.0]\ ML\ Pricers/DNN_baseline/outputs/quarter_eval_100k_with_est_es/q2025${q}_loaded \
    --load-model-path /srv/notebooks/[6.0]\ ML\ Pricers/DNN_baseline/outputs/quarter_eval_100k_with_est_es/model_with_estimates.pt \
    --epochs 60 \
    --early-stopping-patience 6 \
    --val-fraction 0.2 \
    --batch-size 512 \
    --k 64 \
    --verbose
done
```

### 8.3 Train once WITHOUT estimates (save checkpoint)

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python /srv/notebooks/[6.0]\ ML\ Pricers/DNN_baseline/quarter_train_test.py \
  --train-file /srv/notebooks/[6.0]\ ML\ Pricers/DNN_baseline/Data/sampled_sets_100k/Artwork_2026_finetune_100000_2022_2024_train_all_eligible.csv \
  --test-file /srv/notebooks/[6.0]\ ML\ Pricers/DNN_baseline/Data/sampled_sets_100k/Artwork_2026_finetune_100000_q2025Q1_sample1000.csv \
  --output-dir /srv/notebooks/[6.0]\ ML\ Pricers/DNN_baseline/outputs/quarter_eval_100k_no_est_es/q2025Q1 \
  --save-model-path /srv/notebooks/[6.0]\ ML\ Pricers/DNN_baseline/outputs/quarter_eval_100k_no_est_es/model_no_estimates.pt \
  --epochs 60 \
  --early-stopping-patience 6 \
  --val-fraction 0.2 \
  --batch-size 512 \
  --k 64 \
  --verbose
```

### 8.4 Evaluate all quarters WITHOUT estimates (no retraining)

```bash
for q in Q1 Q2 Q3 Q4; do
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python /srv/notebooks/[6.0]\ ML\ Pricers/DNN_baseline/quarter_train_test.py \
    --train-file /srv/notebooks/[6.0]\ ML\ Pricers/DNN_baseline/Data/sampled_sets_100k/Artwork_2026_finetune_100000_2022_2024_train_all_eligible.csv \
    --test-file /srv/notebooks/[6.0]\ ML\ Pricers/DNN_baseline/Data/sampled_sets_100k/Artwork_2026_finetune_100000_q2025${q}_sample1000.csv \
    --output-dir /srv/notebooks/[6.0]\ ML\ Pricers/DNN_baseline/outputs/quarter_eval_100k_no_est_es/q2025${q}_loaded \
    --load-model-path /srv/notebooks/[6.0]\ ML\ Pricers/DNN_baseline/outputs/quarter_eval_100k_no_est_es/model_no_estimates.pt \
    --epochs 60 \
    --early-stopping-patience 6 \
    --val-fraction 0.2 \
    --batch-size 512 \
    --k 64 \
    --verbose
done
```

---

## 9) Output Structure

Per run (per quarter folder):
- `metrics_by_quarter.csv`
- `summary.json`
- `predictions_<quarter>.csv`

Model checkpoints:
- with estimates: `outputs/quarter_eval_100k_with_est_es/model_with_estimates.pt`
- no estimates: `outputs/quarter_eval_100k_no_est_es/model_no_estimates.pt`

Compiled comparisons:
- `outputs/comparison_100k_es/metrics_comparison_by_quarter.csv`
- `outputs/comparison_100k_es/collapse_comparison_by_quarter.csv`
- `outputs/comparison_100k_es/min_median_max_comparison_by_quarter.csv`

---

## 10) Baseline and Diagnostics

Estimate-midpoint baseline artifacts:
- `outputs/estimate_midpoint_baseline/metrics_by_quarter.csv`
- `outputs/estimate_midpoint_baseline/summary.json`

Distribution/collapse diagnostics artifacts:
- `outputs/quarter_eval_100k_with_est_es/collapse_check_latest.csv`
- `outputs/quarter_eval_100k_with_est_es/quarter_price_distribution_raw_vs_log.csv`

---

## 11) Known Failure Modes and Mitigations

### CUDA OOM during validation
- Cause: full validation forward pass on GPU
- Mitigation in current workflow: validation is batched
- Additional knobs: reduce `batch-size`, reduce `k`, keep `expandable_segments:True`

### Duplicate rows/duplicate `lot_url`
- Train and quarterly tests were explicitly deduplicated
- Re-run dedup checks if source data changes

### Distribution collapse risk
- Use early stopping with validation
- Monitor spread diagnostics (`std_ratio`, `iqr_ratio`, tail metrics)
- Compare with/without estimates and against midpoint baseline

---

## 12) Minimal Reproduction Checklist

1. Confirm source CSV exists and schema matches expected columns.
2. Apply the exact filtering logic.
3. Build deduplicated train set (2022-2024 all eligible).
4. Build deduplicated quarter test sets (2025 Q1-Q4, 1000 each).
5. Train one checkpoint per variant (with estimates / without estimates).
6. Evaluate all 4 quarters by loading checkpoint (no retraining).
7. Compile metrics and collapse diagnostics.
8. Compare variants and baseline.

---

## 13) Current Paths (quick reference)

Project root:
- `/srv/notebooks/[6.0] ML Pricers/DNN_baseline`

Train set:
- `Data/sampled_sets_100k/Artwork_2026_finetune_100000_2022_2024_train_all_eligible.csv`

Quarter tests:
- `Data/sampled_sets_100k/Artwork_2026_finetune_100000_q2025Q1_sample1000.csv`
- `Data/sampled_sets_100k/Artwork_2026_finetune_100000_q2025Q2_sample1000.csv`
- `Data/sampled_sets_100k/Artwork_2026_finetune_100000_q2025Q3_sample1000.csv`
- `Data/sampled_sets_100k/Artwork_2026_finetune_100000_q2025Q4_sample1000.csv`

---

## 14) Reported Results (Current Run Artifacts)

This section records the concrete numbers currently generated in this workspace.

### 14.1 Main model comparison (early-stopping setup)

Source:
- `outputs/comparison_100k_es/metrics_comparison_by_quarter.csv`

Per-quarter metrics (`with_est_es` vs `no_est_es`):

| Quarter | With MAE | No MAE | With RMSE | No RMSE | With MAPE% | No MAPE% | With R2 | No R2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2025Q1 | 89,506.49 | 90,679.35 | 493,403.06 | 478,618.43 | 217.31 | 402.90 | -0.0416 | 0.0198 |
| 2025Q2 | 211,180.01 | 225,657.99 | 1,172,844.83 | 1,184,380.89 | 153.17 | 144.64 | -0.0338 | -0.0543 |
| 2025Q3 | 52,462.59 | 38,372.76 | 260,120.70 | 207,508.61 | 212.95 | 176.52 | -0.5938 | -0.0142 |
| 2025Q4 | 263,125.29 | 206,652.12 | 1,107,272.21 | 1,054,817.96 | 231.03 | 137.67 | -0.1411 | -0.0355 |

Overall average:
- `with_est_es`: MAE `154,068.59`, RMSE `758,410.20`, MAPE `203.62`, R2 `-0.2026`
- `no_est_es`: MAE `140,340.55`, RMSE `731,331.47`, MAPE `215.43`, R2 `-0.0210`

Interpretation:
- With-estimates improves MAPE on average but worsens MAE/RMSE/R2 in this run.

### 14.2 Baseline (estimate midpoint)

Source:
- `outputs/estimate_midpoint_baseline/metrics_by_quarter.csv`
- `outputs/estimate_midpoint_baseline/summary.json`

Per-quarter:
- 2025Q1: MAE `28,535.32`, RMSE `164,792.77`, MAPE `54.77`, R2 `0.8777`
- 2025Q2: MAE `79,491.11`, RMSE `1,211,286.64`, MAPE `62.26`, R2 `0.4768`
- 2025Q3: MAE `105,120.85`, RMSE `1,819,281.37`, MAPE `51.29`, R2 `0.0213`
- 2025Q4: MAE `53,969.42`, RMSE `394,249.82`, MAPE `49.50`, R2 `0.8316`

Overall:
- MAE `66,779.18`, RMSE `1,113,506.80`, MAPE `54.45`, R2 `0.3245`

### 14.3 Distribution-collapse diagnostics

Source:
- `outputs/comparison_100k_es/collapse_comparison_by_quarter.csv`
- `outputs/quarter_eval_100k_with_est_es/collapse_check_latest.csv`

Heuristic indicators (higher usually means less collapse):
- `std_ratio = std(pred) / std(true)`
- `iqr_ratio = IQR(pred) / IQR(true)`

Observed for `with_est_es`:
- Q1: std_ratio `0.2638`, iqr_ratio `0.3090`
- Q2: std_ratio `0.0885`, iqr_ratio `0.3132`
- Q3: std_ratio `1.0567`, iqr_ratio `0.2766`
- Q4: std_ratio `0.3968`, iqr_ratio `0.7409`

Observed for `no_est_es`:
- Q1: std_ratio `0.0897`, iqr_ratio `0.4680`
- Q2: std_ratio `0.1687`, iqr_ratio `0.1399`
- Q3: std_ratio `0.1671`, iqr_ratio `0.2401`
- Q4: std_ratio `0.0082`, iqr_ratio `0.0851`

### 14.4 Min / Median / Max comparison

Source:
- `outputs/comparison_100k_es/min_median_max_comparison_by_quarter.csv`

For each quarter, true min/median/max and predicted min/median/max are recorded for both variants.

---

## 15) Paper-Readiness Checklist (must report)

To make the study publication-grade, include the following in your methods/results section:

1. Data provenance and extraction date for each source.
2. Exact filtering logic (as in Section 4) and resulting counts at each stage.
3. Dedup policy (key and tie-break rule), plus before/after row counts.
4. Split policy:
   - Train period (2022-2024)
   - Test periods (2025 Q1-Q4)
   - Exact sample sizes per quarter
5. Feature list with explicit typing and transformations (Section 6).
6. Target transform and inverse-transform definitions.
7. Training hyperparameters and early-stopping criteria:
   - epochs, patience, val_fraction, batch_size, k, optimizer, lr, weight_decay
8. Hardware/software metadata:
   - GPU model and VRAM
   - CUDA/PyTorch versions
   - Python version
9. Randomness controls:
   - random seed values for sampling and training
10. Metrics definitions:
   - MAE/RMSE/MAPE/R2 formulas and aggregation method.
11. Baseline definitions:
   - midpoint baseline and any other comparators.
12. Diagnostics:
   - distribution checks (raw vs log)
   - collapse indicators (std_ratio/iqr_ratio, tail underreach).
13. Artifact paths / files used for reproducibility (Section 9, 14).

Note:
- This guide now includes concrete result values from current artifacts, but for a final paper you should rerun once with locked seeds and frozen data snapshot and then freeze those final numbers.
