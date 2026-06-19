from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
VARIANTS = {
    "with_estimates": "OpenAI GPT-4.1 Mini - With Estimates",
    "no_estimates": "OpenAI GPT-4.1 Mini - No Estimates",
}


def read_variant_metric(variant: str, suffix: str) -> pd.DataFrame | None:
    path = OUTPUT_DIR / f"openai_gpt41mini_{variant}_{suffix}.csv"
    if not path.exists():
        print(f"[compare] missing {path}")
        return None
    df = pd.read_csv(path)
    df["variant"] = variant
    df["model"] = VARIANTS[variant]
    return df


def write_combined(suffix: str, output_name: str) -> None:
    frames = []
    missing = []
    for variant in VARIANTS:
        df = read_variant_metric(variant, suffix)
        if df is not None:
            frames.append(df)
        else:
            missing.append(variant)
    if missing:
        print(f"[compare] skipping {output_name}; missing variants: {', '.join(missing)}")
        print("[compare] run the no-estimates pricing pass first, then rerun this comparison script.")
        return
    if not frames:
        print(f"[compare] no inputs found for {suffix}")
        return
    combined = pd.concat(frames, ignore_index=True)
    out_path = OUTPUT_DIR / output_name
    combined.to_csv(out_path, index=False)
    print(f"[compare] wrote {out_path}: {len(combined):,} rows")


def main() -> None:
    write_combined("overall_metrics", "openai_gpt41mini_estimate_comparison_overall_metrics.csv")
    write_combined("quarter_metrics", "openai_gpt41mini_estimate_comparison_quarter_metrics.csv")
    write_combined("price_bucket_metrics", "openai_gpt41mini_estimate_comparison_price_bucket_metrics.csv")


if __name__ == "__main__":
    main()
