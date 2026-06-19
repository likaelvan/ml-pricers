from __future__ import annotations

import argparse
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from openai_pricer import (
    OpenAIPricer,
    build_artwork_prompt,
    build_system_prompt,
    evaluate_by_price_bucket,
    evaluate_by_quarter,
    evaluate_price_predictions,
    load_quarter_files,
    prepare_pricing_frame,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT.parent / "Data" / "sampled_sets_100k"
DEFAULT_OUTPUT_DIR = ROOT / "outputs"
QUARTERS = ["2025Q1", "2025Q2", "2025Q3", "2025Q4"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GPT-4.1 Mini artwork pricing.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--model-label", default=None)
    parser.add_argument("--include-auction-estimates", action="store_true")
    parser.add_argument("--include-market-indicators", action="store_true", default=True)
    parser.add_argument("--no-market-indicators", dest="include_market_indicators", action="store_false")
    parser.add_argument("--variant", choices=["with_estimates", "no_estimates"], default=None)
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
    parser.add_argument("--env-file", type=Path, default=ROOT / "env")
    return parser.parse_args()


def output_paths(output_dir: Path, variant: str) -> dict[str, Path]:
    prefix = f"openai_gpt41mini_{variant}"
    return {
        "predictions": output_dir / f"{prefix}_predictions.csv",
        "failures": output_dir / f"{prefix}_failures.csv",
        "overall": output_dir / f"{prefix}_overall_metrics.csv",
        "quarter": output_dir / f"{prefix}_quarter_metrics.csv",
        "bucket": output_dir / f"{prefix}_price_bucket_metrics.csv",
    }


def main() -> None:
    args = parse_args()
    load_dotenv(args.env_file)
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set. Set it before running API pricing.")

    variant = args.variant or ("with_estimates" if args.include_auction_estimates else "no_estimates")
    model_label = args.model_label or (
        "OpenAI GPT-4.1 Mini - With Estimates"
        if args.include_auction_estimates
        else "OpenAI GPT-4.1 Mini - No Estimates"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = output_paths(args.output_dir, variant)

    print(
        "[openai run] "
        f"variant={variant}, include_auction_estimates={args.include_auction_estimates}, "
        f"include_market_indicators={args.include_market_indicators}, workers={args.max_workers}"
    )
    print(f"[openai run] predictions={paths['predictions']}")

    raw_df = load_quarter_files(args.data_dir, quarters=QUARTERS, verbose=True)
    pricing_df = prepare_pricing_frame(
        raw_df,
        rows_per_quarter=args.rows_per_quarter,
        max_rows=args.max_rows,
        random_state=args.random_state,
        dedupe_key_col="lot_url",
        verbose=True,
    )
    print(f"[openai run] rows selected: {len(pricing_df):,}")
    print(pricing_df["source_quarter"].value_counts().sort_index().to_string())

    system_prompt = build_system_prompt(include_auction_estimates=args.include_auction_estimates)
    thread_local = threading.local()

    def get_thread_pricer() -> OpenAIPricer:
        if not hasattr(thread_local, "pricer"):
            thread_local.pricer = OpenAIPricer(
                model=args.model,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
                max_retries=args.max_retries,
                use_structured_output=True,
            )
        return thread_local.pricer

    def price_one(row_index: int, row_dict: dict) -> dict:
        user_prompt = build_artwork_prompt(
            row_dict,
            include_auction_estimates=args.include_auction_estimates,
            include_market_indicators=args.include_market_indicators,
        )
        result = get_thread_pricer().price(system_prompt=system_prompt, user_prompt=user_prompt)
        base = dict(row_dict)
        base.update(result.as_dict())
        base["openai_model"] = args.model
        base["openai_variant"] = variant
        base["openai_include_auction_estimates"] = args.include_auction_estimates
        base["openai_include_market_indicators"] = args.include_market_indicators
        base["openai_input_row_index"] = row_index
        return base

    existing = pd.DataFrame()
    completed_keys: set[str] = set()
    if args.resume and paths["predictions"].exists():
        existing = pd.read_csv(paths["predictions"], low_memory=False)
        if "lot_url" in existing.columns and "openai_error" in existing.columns:
            completed_keys = set(existing.loc[existing["openai_error"].isna(), "lot_url"].dropna().astype(str))
        print(f"[openai run] resume found successful rows: {len(completed_keys):,}")

    pending = []
    for idx, row in pricing_df.iterrows():
        key = str(row.get("lot_url"))
        if args.resume and key in completed_keys:
            continue
        pending.append((idx, row.to_dict()))

    print(f"[openai run] pending rows: {len(pending):,}")
    rows: list[dict] = []
    started = time.perf_counter()

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_payload = {
            executor.submit(price_one, row_index, row_dict): (row_index, row_dict)
            for row_index, row_dict in pending
        }
        for completed_count, future in enumerate(as_completed(future_to_payload), start=1):
            row_index, row_dict = future_to_payload[future]
            try:
                base = future.result()
            except Exception as exc:  # noqa: BLE001 - keep resumable rows for failed calls.
                base = dict(row_dict)
                base.update(
                    {
                        "openai_predicted_price_usd": None,
                        "openai_confidence": None,
                        "openai_pricing_basis": None,
                        "openai_notes": None,
                        "openai_raw_output": None,
                        "openai_error": f"{type(exc).__name__}: {exc}",
                        "openai_latency_seconds": None,
                        "openai_input_tokens": None,
                        "openai_output_tokens": None,
                        "openai_total_tokens": None,
                        "openai_model": args.model,
                        "openai_variant": variant,
                        "openai_include_auction_estimates": args.include_auction_estimates,
                        "openai_include_market_indicators": args.include_market_indicators,
                        "openai_input_row_index": row_index,
                    }
                )

            rows.append(base)
            elapsed = time.perf_counter() - started
            print(
                f"[{completed_count}/{len(pending)}] row={row_index} "
                f"quarter={base.get('source_quarter')} price={base.get('openai_predicted_price_usd')} "
                f"error={base.get('openai_error')} elapsed={elapsed:.1f}s",
                flush=True,
            )

            if args.checkpoint_every and completed_count % args.checkpoint_every == 0:
                write_predictions(existing, rows, paths["predictions"])

    results = write_predictions(existing, rows, paths["predictions"])
    write_metrics(results, paths, model_label)
    print(f"[openai run] final rows written: {len(results):,}")


def write_predictions(existing: pd.DataFrame, rows: list[dict], path: Path) -> pd.DataFrame:
    results = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    if not results.empty:
        results = results.drop_duplicates(subset=["lot_url"], keep="last")
        if "openai_input_row_index" in results.columns:
            results = results.sort_values("openai_input_row_index", na_position="last")
        results.to_csv(path, index=False)
    print(f"[openai run] checkpoint wrote {len(results):,} rows to {path}", flush=True)
    return results


def write_metrics(results: pd.DataFrame, paths: dict[str, Path], model_label: str) -> None:
    if results.empty:
        return
    failures = results[results["openai_error"].notna()].copy()
    failures.to_csv(paths["failures"], index=False)

    scored = results[results["openai_predicted_price_usd"].notna()].copy()
    scored["ah_price_usd"] = pd.to_numeric(scored["ah_price_usd"], errors="coerce")
    scored["openai_predicted_price_usd"] = pd.to_numeric(scored["openai_predicted_price_usd"], errors="coerce")
    overall_metrics = pd.DataFrame(
        [evaluate_price_predictions(scored["ah_price_usd"], scored["openai_predicted_price_usd"], label=model_label)]
    )
    quarter_metrics = evaluate_by_quarter(
        scored,
        "openai_predicted_price_usd",
        actual_col="ah_price_usd",
        label=model_label,
    )
    price_bucket_metrics = evaluate_by_price_bucket(
        scored,
        "openai_predicted_price_usd",
        actual_col="ah_price_usd",
        label=model_label,
    )
    overall_metrics.to_csv(paths["overall"], index=False)
    quarter_metrics.to_csv(paths["quarter"], index=False)
    price_bucket_metrics.to_csv(paths["bucket"], index=False)
    print("[openai run] overall metrics:")
    print(overall_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
