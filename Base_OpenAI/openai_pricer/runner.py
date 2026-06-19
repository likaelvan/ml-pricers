from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import pandas as pd
from dotenv import load_dotenv

from .evaluation import evaluate_by_price_bucket, evaluate_by_quarter, evaluate_price_predictions
from .openai_client import OpenAIPricer
from .prompt_builder import load_explicit_quarter_files, prepare_pricing_frame

PromptBuilder = Callable[[dict, bool, bool], str]
SystemPromptBuilder = Callable[[bool], str]


def variant_output_paths(output_dir: Path, prefix: str) -> dict[str, Path]:
    return {
        "predictions": output_dir / f"{prefix}_predictions.csv",
        "failures": output_dir / f"{prefix}_failures.csv",
        "overall": output_dir / f"{prefix}_overall_metrics.csv",
        "quarter": output_dir / f"{prefix}_quarter_metrics.csv",
        "bucket": output_dir / f"{prefix}_price_bucket_metrics.csv",
    }


def run_asset_pricer(
    *,
    asset_name: str,
    quarter_files: dict[str, Path],
    output_dir: Path,
    output_prefix: str,
    model: str,
    model_label: str,
    build_prompt: PromptBuilder,
    build_system_prompt: SystemPromptBuilder,
    actual_col: str,
    include_auction_estimates: bool,
    include_market_indicators: bool,
    max_rows: int | None,
    rows_per_quarter: int | None,
    random_state: int,
    resume: bool,
    checkpoint_every: int,
    max_workers: int,
    temperature: float,
    max_output_tokens: int,
    max_retries: int,
    env_file: Path,
    dedupe_key_col: str = "lot_url",
) -> None:
    load_dotenv(env_file)
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set. Set it before running API pricing.")

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = variant_output_paths(output_dir, output_prefix)

    print(
        f"[openai {asset_name}] output_prefix={output_prefix}, model={model}, "
        f"include_auction_estimates={include_auction_estimates}, "
        f"include_market_indicators={include_market_indicators}, workers={max_workers}"
    )
    print(f"[openai {asset_name}] predictions={paths['predictions']}")

    raw_df = load_explicit_quarter_files(quarter_files, verbose=True)
    pricing_df = prepare_pricing_frame(
        raw_df,
        rows_per_quarter=rows_per_quarter,
        max_rows=max_rows,
        random_state=random_state,
        dedupe_key_col=dedupe_key_col,
        verbose=True,
    )
    print(f"[openai {asset_name}] rows selected: {len(pricing_df):,}")
    print(pricing_df["source_quarter"].value_counts().sort_index().to_string())

    system_prompt = build_system_prompt(include_auction_estimates=include_auction_estimates)
    thread_local = threading.local()

    def get_thread_pricer() -> OpenAIPricer:
        if not hasattr(thread_local, "pricer"):
            thread_local.pricer = OpenAIPricer(
                model=model,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                max_retries=max_retries,
                use_structured_output=True,
            )
        return thread_local.pricer

    def price_one(row_index: int, row_dict: dict) -> dict:
        user_prompt = build_prompt(row_dict, include_auction_estimates, include_market_indicators)
        result = get_thread_pricer().price(system_prompt=system_prompt, user_prompt=user_prompt)
        base = dict(row_dict)
        base.update(result.as_dict())
        base["openai_model"] = model
        base["openai_asset"] = asset_name
        base["openai_include_auction_estimates"] = include_auction_estimates
        base["openai_include_market_indicators"] = include_market_indicators
        base["openai_input_row_index"] = row_index
        return base

    existing = pd.DataFrame()
    completed_keys: set[str] = set()
    if resume and paths["predictions"].exists():
        existing = pd.read_csv(paths["predictions"], low_memory=False)
        if dedupe_key_col in existing.columns and "openai_error" in existing.columns:
            completed_keys = set(existing.loc[existing["openai_error"].isna(), dedupe_key_col].dropna().astype(str))
        print(f"[openai {asset_name}] resume found successful rows: {len(completed_keys):,}")

    pending = []
    for idx, row in pricing_df.iterrows():
        key = str(row.get(dedupe_key_col))
        if resume and key in completed_keys:
            continue
        pending.append((idx, row.to_dict()))

    print(f"[openai {asset_name}] pending rows: {len(pending):,}")
    rows: list[dict] = []
    started = time.perf_counter()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_payload = {
            executor.submit(price_one, row_index, row_dict): (row_index, row_dict)
            for row_index, row_dict in pending
        }
        for completed_count, future in enumerate(as_completed(future_to_payload), start=1):
            row_index, row_dict = future_to_payload[future]
            try:
                base = future.result()
            except Exception as exc:  # noqa: BLE001 - checkpoint failed rows for resumability.
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
                        "openai_model": model,
                        "openai_asset": asset_name,
                        "openai_include_auction_estimates": include_auction_estimates,
                        "openai_include_market_indicators": include_market_indicators,
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

            if checkpoint_every and completed_count % checkpoint_every == 0:
                write_predictions(existing, rows, paths["predictions"], dedupe_key_col=dedupe_key_col)

    results = write_predictions(existing, rows, paths["predictions"], dedupe_key_col=dedupe_key_col)
    write_metrics(results, paths, model_label=model_label, actual_col=actual_col)
    print(f"[openai {asset_name}] final rows written: {len(results):,}")


def write_predictions(existing: pd.DataFrame, rows: list[dict], path: Path, *, dedupe_key_col: str) -> pd.DataFrame:
    results = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    if not results.empty:
        if dedupe_key_col in results.columns:
            results = results.drop_duplicates(subset=[dedupe_key_col], keep="last")
        if "openai_input_row_index" in results.columns:
            results = results.sort_values("openai_input_row_index", na_position="last")
        results.to_csv(path, index=False)
    print(f"[openai run] checkpoint wrote {len(results):,} rows to {path}", flush=True)
    return results


def write_metrics(results: pd.DataFrame, paths: dict[str, Path], *, model_label: str, actual_col: str) -> None:
    if results.empty:
        return
    failures = results[results["openai_error"].notna()].copy()
    failures.to_csv(paths["failures"], index=False)

    scored = results[results["openai_predicted_price_usd"].notna()].copy()
    scored[actual_col] = pd.to_numeric(scored[actual_col], errors="coerce")
    scored["openai_predicted_price_usd"] = pd.to_numeric(scored["openai_predicted_price_usd"], errors="coerce")
    overall_metrics = pd.DataFrame(
        [evaluate_price_predictions(scored[actual_col], scored["openai_predicted_price_usd"], label=model_label)]
    )
    quarter_metrics = evaluate_by_quarter(
        scored,
        "openai_predicted_price_usd",
        actual_col=actual_col,
        label=model_label,
    )
    price_bucket_metrics = evaluate_by_price_bucket(
        scored,
        "openai_predicted_price_usd",
        actual_col=actual_col,
        label=model_label,
    )
    overall_metrics.to_csv(paths["overall"], index=False)
    quarter_metrics.to_csv(paths["quarter"], index=False)
    price_bucket_metrics.to_csv(paths["bucket"], index=False)
    print("[openai run] overall metrics:")
    print(overall_metrics.to_string(index=False))
