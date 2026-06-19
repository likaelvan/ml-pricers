from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_price_predictions(
    actual: pd.Series,
    predicted: pd.Series | np.ndarray,
    *,
    label: str,
) -> dict[str, float | str | int]:
    """Return MAE, RMSE, MAPE, and R2."""

    actual = pd.Series(actual, dtype=float)
    predicted = pd.Series(predicted, index=actual.index, dtype=float)
    mask = actual.gt(0) & predicted.notna() & np.isfinite(predicted)
    actual = actual[mask]
    predicted = predicted[mask]
    if len(actual) == 0:
        return {"model": label, "n": 0, "MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "R2": np.nan}
    return {
        "model": label,
        "n": int(len(actual)),
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "MAPE": float(((predicted - actual).abs() / actual).mean()),
        "R2": float(r2_score(actual, predicted)) if len(actual) > 1 else np.nan,
    }


def evaluate_by_quarter(
    frame: pd.DataFrame,
    predicted_col: str,
    *,
    actual_col: str = "ah_price_usd",
    quarter_col: str = "source_quarter",
    label: str = "OpenAI GPT-4.1 Mini",
) -> pd.DataFrame:
    rows = []
    for quarter, group in frame.groupby(quarter_col, dropna=False):
        row = evaluate_price_predictions(group[actual_col], group[predicted_col], label=label)
        row["quarter"] = quarter
        rows.append(row)
    return pd.DataFrame(rows).sort_values("quarter").reset_index(drop=True)


def evaluate_by_price_bucket(
    frame: pd.DataFrame,
    predicted_col: str,
    *,
    actual_col: str = "ah_price_usd",
    label: str = "OpenAI GPT-4.1 Mini",
) -> pd.DataFrame:
    work = frame.copy()
    work["price_bucket"] = pd.cut(
        pd.to_numeric(work[actual_col], errors="coerce"),
        bins=[0, 1_000, 5_000, 25_000, 100_000, 500_000, np.inf],
        labels=["<$1k", "$1k-$5k", "$5k-$25k", "$25k-$100k", "$100k-$500k", "$500k+"],
        include_lowest=True,
    )
    rows = []
    for bucket, group in work.groupby("price_bucket", observed=False):
        if group.empty:
            continue
        row = evaluate_price_predictions(group[actual_col], group[predicted_col], label=label)
        row["price_bucket"] = str(bucket)
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)
