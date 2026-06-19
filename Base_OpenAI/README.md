# Base OpenAI Pricers

This folder contains shared OpenAI pricing utilities and asset-specific GPT-4.1 Mini pricers for artwork, watches, and jewelry.

## Structure

```text
Base_OpenAI/
  openai_pricer/      # shared OpenAI client, data loading, runner, and metrics
  artwork/            # artwork-specific prompt, runner, notebook, outputs
  watch/              # watch-specific prompt, runner, notebook, outputs
  jewelry/            # jewelry-specific prompt, runner, notebook, outputs
  outputs/            # legacy artwork outputs from the original root runner
```

The legacy root-level artwork runner is still present for compatibility. New asset-specific work should use the folders above.

Each asset supports two variants:

- `with_estimates`: prompt includes auction-house low/high estimate fields.
- `no_estimates`: prompt excludes auction-house low/high estimate fields.

## Asset Runners

Artwork:

```bash
python Base_OpenAI/artwork/run_openai_artwork_pricer.py \
  --variant with_estimates \
  --include-auction-estimates \
  --rows-per-quarter 10 \
  --no-resume
```

Watches:

```bash
python Base_OpenAI/watch/run_openai_watch_pricer.py \
  --variant with_estimates \
  --include-auction-estimates \
  --rows-per-quarter 10 \
  --no-resume
```

Jewelry:

```bash
python Base_OpenAI/jewelry/run_openai_jewelry_pricer.py \
  --variant with_estimates \
  --include-auction-estimates \
  --rows-per-quarter 10 \
  --no-resume
```

Omit `--rows-per-quarter` for a full run. Outputs are written to each asset folder's `outputs/` directory.

## Notebooks

Each asset folder contains a notebook with explicit quarter input paths:

- `artwork/run_openai_artwork_pricer.ipynb`
- `watch/run_openai_watch_pricer.ipynb`
- `jewelry/run_openai_jewelry_pricer.ipynb`

## Legacy Artwork Outputs

The original completed with-estimates outputs are preserved as:

- `outputs/openai_gpt41mini_with_estimates_predictions.csv`
- `outputs/openai_gpt41mini_with_estimates_overall_metrics.csv`
- `outputs/openai_gpt41mini_with_estimates_quarter_metrics.csv`
- `outputs/openai_gpt41mini_with_estimates_price_bucket_metrics.csv`

## Legacy Root Runner

The root-level runner still supports the original artwork workflow from the workspace root:

```bash
python Base_OpenAI/run_openai_pricer.py \
  --variant no_estimates \
  --max-workers 50 \
  --checkpoint-every 10 \
  --no-resume
```

If the run is interrupted, resume with:

```bash
python Base_OpenAI/run_openai_pricer.py \
  --variant no_estimates \
  --max-workers 50 \
  --checkpoint-every 10 \
  --resume
```

## Build With vs Without Estimate Comparison

After the no-estimates run finishes:

```bash
python Base_OpenAI/compare_openai_estimate_variants.py
```

This writes:

- `outputs/openai_gpt41mini_estimate_comparison_overall_metrics.csv`
- `outputs/openai_gpt41mini_estimate_comparison_quarter_metrics.csv`
- `outputs/openai_gpt41mini_estimate_comparison_price_bucket_metrics.csv`

## Notebook

`run_openai_gpt41mini_pricer.ipynb` has also been updated to use variant-specific output paths. Set `INCLUDE_AUCTION_ESTIMATES = False` for the no-estimates variant.
