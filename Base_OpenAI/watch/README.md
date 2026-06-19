# Watch OpenAI Pricer

Asset-specific GPT-4.1 Mini pricer for watch auction lots.

## Input Files

```text
Data/watch/watch_sales_2025Q1_dedup_lot_url.csv
Data/watch/watch_sales_2025Q2_dedup_lot_url.csv
Data/watch/watch_sales_2025Q3_dedup_lot_url.csv
Data/watch/watch_sales_2025Q4_dedup_lot_url.csv
```

## Prompt Inputs

The watch prompt uses brand, model, reference number, year manufactured, case size, box/papers, condition, case material, band material, gold/diamond indicators, description, auction house, auction title, location, sale date, lot number, currency, optional auction estimates, and optional market indicators.

The prompt excludes realized price, buyer-premium price, existing watch LLM valuation columns, midpoint error fields, and other leakage columns.

## Run

Smoke test from the workspace root:

```bash
python Base_OpenAI/watch/run_openai_watch_pricer.py \
  --variant with_estimates \
  --include-auction-estimates \
  --rows-per-quarter 10 \
  --no-resume
```

Full run:

```bash
python Base_OpenAI/watch/run_openai_watch_pricer.py \
  --variant with_estimates \
  --include-auction-estimates
```

No-estimates run:

```bash
python Base_OpenAI/watch/run_openai_watch_pricer.py \
  --variant no_estimates
```

Outputs are written to `Base_OpenAI/watch/outputs/`.
