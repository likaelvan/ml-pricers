from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data" / "sampled_sets_100k"
OUT_DIR = ROOT / "Auction_Midpoint_Baseline" / "outputs"
QUARTERS = ["2025Q1", "2025Q2", "2025Q3", "2025Q4"]
PRICE_BUCKETS = [0, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000, np.inf]
PRICE_LABELS = ["<$1k", "$1k-$5k", "$5k-$10k", "$10k-$50k", "$50k-$100k", "$100k-$500k", "$500k+"]


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true != 0)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def metrics_frame(df: pd.DataFrame, actual_col: str, pred_col: str, model: str) -> dict:
    mask = df[[actual_col, pred_col]].notna().all(axis=1)
    y_true = df.loc[mask, actual_col].astype(float).to_numpy()
    y_pred = df.loc[mask, pred_col].astype(float).to_numpy()
    if len(y_true) == 0:
        return {"model": model, "n": 0, "MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "R2": np.nan}
    return {
        "model": model,
        "n": int(len(y_true)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE": safe_mape(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
    }


def bias_frame(df: pd.DataFrame, actual_col: str, pred_col: str, model: str) -> dict:
    mask = df[[actual_col, pred_col]].notna().all(axis=1)
    d = df.loc[mask, [actual_col, pred_col]].astype(float).copy()
    if d.empty:
        return {"model": model, "n": 0}
    err = d[pred_col] - d[actual_col]
    pct_err = np.where(d[actual_col] != 0, err / d[actual_col], np.nan)
    return {
        "model": model,
        "n": int(len(d)),
        "mean_error_usd": float(np.nanmean(err)),
        "median_error_usd": float(np.nanmedian(err)),
        "mean_pct_error": float(np.nanmean(pct_err)),
        "median_pct_error": float(np.nanmedian(pct_err)),
        "overvaluation_rate": float(np.mean(err > 0)),
        "undervaluation_rate": float(np.mean(err < 0)),
    }


def deviation_from_midpoint(df: pd.DataFrame, pred_col: str, midpoint_col: str, model: str) -> dict:
    mask = df[[pred_col, midpoint_col]].notna().all(axis=1) & (df[midpoint_col].astype(float) != 0)
    d = df.loc[mask, [pred_col, midpoint_col]].astype(float)
    if d.empty:
        return {"model": model, "n": 0}
    diff = d[pred_col] - d[midpoint_col]
    ratio = d[pred_col] / d[midpoint_col]
    return {
        "model": model,
        "n": int(len(d)),
        "mean_model_minus_midpoint_usd": float(np.nanmean(diff)),
        "median_model_minus_midpoint_usd": float(np.nanmedian(diff)),
        "mean_abs_model_minus_midpoint_usd": float(np.nanmean(np.abs(diff))),
        "median_model_to_midpoint_ratio": float(np.nanmedian(ratio)),
    }


def add_price_bucket(df: pd.DataFrame, actual_col: str) -> pd.DataFrame:
    out = df.copy()
    out["price_bucket"] = pd.cut(out[actual_col], bins=PRICE_BUCKETS, labels=PRICE_LABELS, right=False)
    return out


def load_quarter_data() -> pd.DataFrame:
    frames = []
    for quarter in QUARTERS:
        path = DATA_DIR / f"Artwork_2026_finetune_100000_q{quarter}_sample1000.csv"
        df = pd.read_csv(path, low_memory=False)
        df["source_quarter"] = quarter
        df["auction_midpoint_usd"] = (
            pd.to_numeric(df["ah_low_estimate_usd"], errors="coerce")
            + pd.to_numeric(df["ah_high_estimate_usd"], errors="coerce")
        ) / 2.0
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def grouped_metrics(df: pd.DataFrame, actual_col: str, pred_col: str, model: str, group_col: str) -> pd.DataFrame:
    rows = []
    for key, part in df.groupby(group_col, dropna=False, observed=True):
        row = metrics_frame(part, actual_col, pred_col, model)
        row[group_col] = key
        rows.append(row)
    return pd.DataFrame(rows)


def grouped_bias(df: pd.DataFrame, actual_col: str, pred_col: str, model: str, group_col: str) -> pd.DataFrame:
    rows = []
    for key, part in df.groupby(group_col, dropna=False, observed=True):
        row = bias_frame(part, actual_col, pred_col, model)
        row[group_col] = key
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_quarter_data()
    keep_cols = [
        "ID", "lot_url", "source_quarter", "sale_date", "artist_name", "artwork_title",
        "ah_price_usd", "ah_bp_price_usd", "ah_low_estimate_usd", "ah_high_estimate_usd",
        "auction_midpoint_usd",
    ]
    pred = df[keep_cols].copy()
    pred.to_csv(OUT_DIR / "auction_midpoint_predictions.csv", index=False)

    configs = [
        ("ah_price_usd", "Auction Midpoint - Hammer Target"),
        ("ah_bp_price_usd", "Auction Midpoint - Buyer Premium Target"),
    ]

    overall_rows = []
    quarter_rows = []
    bucket_rows = []
    bias_rows = []
    bucket_bias_rows = []
    for actual_col, model in configs:
        work = add_price_bucket(pred, actual_col)
        overall_rows.append(metrics_frame(work, actual_col, "auction_midpoint_usd", model))
        quarter_rows.append(grouped_metrics(work, actual_col, "auction_midpoint_usd", model, "source_quarter"))
        bucket_rows.append(grouped_metrics(work, actual_col, "auction_midpoint_usd", model, "price_bucket"))
        bias_rows.append(bias_frame(work, actual_col, "auction_midpoint_usd", model))
        bucket_bias_rows.append(grouped_bias(work, actual_col, "auction_midpoint_usd", model, "price_bucket"))

    pd.DataFrame(overall_rows).to_csv(OUT_DIR / "auction_midpoint_overall_metrics.csv", index=False)
    pd.concat(quarter_rows, ignore_index=True).to_csv(OUT_DIR / "auction_midpoint_quarter_metrics.csv", index=False)
    pd.concat(bucket_rows, ignore_index=True).to_csv(OUT_DIR / "auction_midpoint_price_bucket_metrics.csv", index=False)
    pd.DataFrame(bias_rows).to_csv(OUT_DIR / "auction_midpoint_bias_overall.csv", index=False)
    pd.concat(bucket_bias_rows, ignore_index=True).to_csv(OUT_DIR / "auction_midpoint_bias_by_price_bucket.csv", index=False)

    print(f"[auction midpoint] rows={len(pred):,}")
    print(f"[auction midpoint] outputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
