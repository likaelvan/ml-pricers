from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
BASE_OPENAI = PROJECT_DIR.parent
WORKSPACE_ROOT = BASE_OPENAI.parent
sys.path.insert(0, str(BASE_OPENAI))

from jewelry_prompt_builder import build_prompt, build_system_prompt  # noqa: E402
from openai_pricer.runner import run_asset_pricer  # noqa: E402

QUARTERS = ("2025Q1", "2025Q2", "2025Q3", "2025Q4")
DEFAULT_DATA_DIR = WORKSPACE_ROOT / "Data" / "sampled_sets_jewelry"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GPT-4.1 Mini jewelry pricing.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--include-auction-estimates", action="store_true")
    parser.add_argument("--include-market-indicators", action="store_true", default=True)
    parser.add_argument("--no-market-indicators", dest="include_market_indicators", action="store_false")
    parser.add_argument("--variant", choices=["with_estimates", "no_estimates"], default=None)
    parser.add_argument("--target-col", default="ah_price_usd")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--rows-per-quarter", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--max-workers", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=300)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--env-file", type=Path, default=BASE_OPENAI / "env")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variant = args.variant or ("with_estimates" if args.include_auction_estimates else "no_estimates")
    include_estimates = args.include_auction_estimates or variant == "with_estimates"
    quarter_files = {
        quarter: args.data_dir / f"Jewelry_2025_high_midpoint_error_q{quarter}_sample1000.csv"
        for quarter in QUARTERS
    }
    run_asset_pricer(
        asset_name="jewelry",
        quarter_files=quarter_files,
        output_dir=args.output_dir,
        output_prefix=f"openai_gpt41mini_jewelry_{variant}",
        model=args.model,
        model_label=f"OpenAI GPT-4.1 Mini Jewelry - {'With Estimates' if include_estimates else 'No Estimates'}",
        build_prompt=build_prompt,
        build_system_prompt=build_system_prompt,
        actual_col=args.target_col,
        include_auction_estimates=include_estimates,
        include_market_indicators=args.include_market_indicators,
        max_rows=args.max_rows,
        rows_per_quarter=args.rows_per_quarter,
        random_state=args.random_state,
        resume=args.resume,
        checkpoint_every=args.checkpoint_every,
        max_workers=args.max_workers,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        max_retries=args.max_retries,
        env_file=args.env_file,
    )


if __name__ == "__main__":
    main()
