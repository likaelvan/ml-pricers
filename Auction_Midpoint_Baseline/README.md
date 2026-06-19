# Auction Midpoint Baseline

This folder computes the auction-house estimate midpoint baseline used in the comparative valuation report.

## Definition

`auction_midpoint_usd = (ah_low_estimate_usd + ah_high_estimate_usd) / 2`

The script evaluates the midpoint against two targets:

- `ah_price_usd`: hammer-price target used by Ridge, Huber, and Base_OpenAI outputs.
- `ah_bp_price_usd`: buyer-premium target used by the current DNN pipeline.

## Run

```bash
python Auction_Midpoint_Baseline/auction_midpoint_baseline.py
```

Or run the notebook:

```text
Auction_Midpoint_Baseline/notebooks/run_auction_midpoint_baseline.ipynb
```

## Outputs

- `outputs/auction_midpoint_predictions.csv`: row-level midpoint predictions and targets.
- `outputs/auction_midpoint_overall_metrics.csv`: overall MAE, RMSE, MAPE, R2.
- `outputs/auction_midpoint_quarter_metrics.csv`: quarterly metrics.
- `outputs/auction_midpoint_price_bucket_metrics.csv`: price-bucket metrics.
- `outputs/auction_midpoint_bias_overall.csv`: overall over/undervaluation bias.
- `outputs/auction_midpoint_bias_by_price_bucket.csv`: price-bucket bias.
