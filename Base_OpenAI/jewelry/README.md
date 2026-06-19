# Jewelry OpenAI Pricer

Asset-specific GPT-4.1 Mini pricer for jewelry auction lots.

## Input Files

```text
Data/sampled_sets_jewelry/Jewelry_2025_high_midpoint_error_q2025Q1_sample1000.csv
Data/sampled_sets_jewelry/Jewelry_2025_high_midpoint_error_q2025Q2_sample1000.csv
Data/sampled_sets_jewelry/Jewelry_2025_high_midpoint_error_q2025Q3_sample1000.csv
Data/sampled_sets_jewelry/Jewelry_2025_high_midpoint_error_q2025Q4_sample1000.csv
```

## Prompt Inputs

The jewelry prompt uses product name, maker/signature fields, metal specification, jewelry category, measurements, gemstone/diamond/pearl counts and specifications, certifications, origin, description, auction house, sale date, optional auction estimates, and optional market indicators.

Dictionary-like jewelry fields are flattened into readable prompt text before being sent to the model. The prompt excludes realized price, buyer-premium price, existing jewelry LLM valuation columns, price regime/bin fields, and other leakage columns.

## Run

Smoke test from the workspace root:

```bash
python Base_OpenAI/jewelry/run_openai_jewelry_pricer.py \
  --variant with_estimates \
  --include-auction-estimates \
  --rows-per-quarter 10 \
  --no-resume
```

Full run:

```bash
python Base_OpenAI/jewelry/run_openai_jewelry_pricer.py \
  --variant with_estimates \
  --include-auction-estimates
```

No-estimates run:

```bash
python Base_OpenAI/jewelry/run_openai_jewelry_pricer.py \
  --variant no_estimates
```

Outputs are written to `Base_OpenAI/jewelry/outputs/`.
