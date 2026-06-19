# Artwork OpenAI Pricer

Asset-specific GPT-4.1 Mini pricer for artwork auction lots.

## Input Files

```text
Data/sampled_sets_100k/Artwork_2026_finetune_100000_q2025Q1_sample1000.csv
Data/sampled_sets_100k/Artwork_2026_finetune_100000_q2025Q2_sample1000.csv
Data/sampled_sets_100k/Artwork_2026_finetune_100000_q2025Q3_sample1000.csv
Data/sampled_sets_100k/Artwork_2026_finetune_100000_q2025Q4_sample1000.csv
```

## Prompt Inputs

The artwork prompt uses artist, title, creation year, medium/material, dimensions, edition, subject matter, provenance, description, auction house, location, sale date, lot number, currency, optional auction estimates, and optional market indicators.

The prompt excludes realized price, buyer-premium price, Barker/model outputs, price regime/bin fields, and other leakage columns.

## Run

From the workspace root:

```bash
python Base_OpenAI/artwork/run_openai_artwork_pricer.py \
  --variant with_estimates \
  --include-auction-estimates \
  --rows-per-quarter 10 \
  --no-resume
```

Full run:

```bash
python Base_OpenAI/artwork/run_openai_artwork_pricer.py \
  --variant with_estimates \
  --include-auction-estimates
```

No-estimates run:

```bash
python Base_OpenAI/artwork/run_openai_artwork_pricer.py \
  --variant no_estimates
```

Outputs are written to `Base_OpenAI/artwork/outputs/`.
