from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from .features import price_bucket


DEFAULT_LGBM_PARAMS = {
    "objective": "regression",
    "n_estimators": 800,
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_child_samples": 30,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.1,
    "reg_lambda": 2.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}


def get_lightgbm_regressor(params: dict | None = None):
    """Return an LGBMRegressor, raising a clear dependency message if missing."""

    try:
        from lightgbm import LGBMRegressor
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "lightgbm is not installed. Install it in this environment before running "
            "the training cells, for example: pip install lightgbm"
        ) from exc

    merged = dict(DEFAULT_LGBM_PARAMS)
    if params:
        merged.update(params)
    return LGBMRegressor(**merged)


def fit_lgbm_log_price_model(
    X_train: pd.DataFrame,
    y_train_log: pd.Series,
    *,
    categorical_cols: Iterable[str],
    params: dict | None = None,
    validation_fraction: float = 0.10,
    early_stopping_rounds: int | None = 50,
    log_period: int | None = 50,
    eval_metric: str = "rmse",
    random_state: int = 42,
    verbose: bool = True,
):
    """Train LightGBM on log1p USD prices with optional diagnostics."""

    model = get_lightgbm_regressor(params)
    categorical_cols = list(categorical_cols)
    fit_kwargs = {"categorical_feature": categorical_cols}
    callbacks = []

    if validation_fraction and validation_fraction > 0:
        X_fit, X_valid, y_fit, y_valid = train_test_split(
            X_train,
            y_train_log,
            test_size=validation_fraction,
            random_state=random_state,
        )
        fit_kwargs["eval_set"] = [(X_valid, y_valid)]
        fit_kwargs["eval_metric"] = eval_metric

        try:
            import lightgbm as lgb
        except ModuleNotFoundError:
            lgb = None

        if lgb is not None and log_period and log_period > 0:
            callbacks.append(lgb.log_evaluation(period=log_period))
        if lgb is not None and early_stopping_rounds and early_stopping_rounds > 0:
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=verbose))
        if callbacks:
            fit_kwargs["callbacks"] = callbacks

        _log(
            verbose,
            "Training LightGBM with validation diagnostics: "
            f"train_fit={len(X_fit):,}, validation={len(X_valid):,}, "
            f"features={X_train.shape[1]:,}, categorical={len(categorical_cols):,}, "
            f"eval_metric={eval_metric}, log_period={log_period}, "
            f"early_stopping_rounds={early_stopping_rounds}",
        )
        model.fit(X_fit, y_fit, **fit_kwargs)
    else:
        _log(
            verbose,
            "Training LightGBM without validation diagnostics: "
            f"train={len(X_train):,}, features={X_train.shape[1]:,}, categorical={len(categorical_cols):,}",
        )
        model.fit(X_train, y_train_log, **fit_kwargs)

    if hasattr(model, "best_iteration_") and model.best_iteration_:
        _log(verbose, f"Best iteration: {model.best_iteration_}")
    if hasattr(model, "best_score_") and model.best_score_:
        _log(verbose, f"Best validation score: {model.best_score_}")
    return model


def lightgbm_training_history(model) -> pd.DataFrame:
    """Return LightGBM eval history as a tidy dataframe when available."""

    result = getattr(model, "evals_result_", None)
    if not result:
        return pd.DataFrame()

    rows = []
    for dataset_name, metrics in result.items():
        for metric_name, values in metrics.items():
            for iteration, value in enumerate(values, start=1):
                rows.append(
                    {
                        "dataset": dataset_name,
                        "metric": metric_name,
                        "iteration": iteration,
                        "value": value,
                    }
                )
    return pd.DataFrame(rows)


def predict_price_from_log_model(model, X: pd.DataFrame) -> np.ndarray:
    """Predict USD prices from a model trained on log1p prices."""

    pred_log = model.predict(X)
    return np.expm1(pred_log).clip(min=0)


def evaluate_price_predictions(
    actual: pd.Series,
    predicted: pd.Series | np.ndarray,
    *,
    label: str,
) -> dict[str, float | str | int]:
    """Return the four approved accuracy metrics for price predictions."""

    actual = pd.Series(actual, dtype=float)
    predicted = pd.Series(predicted, index=actual.index, dtype=float)
    mask = actual.gt(0) & predicted.notna() & np.isfinite(predicted)
    actual = actual[mask]
    predicted = predicted[mask]
    abs_pct_error = ((predicted - actual) / actual).abs()

    return {
        "model": label,
        "n": int(len(actual)),
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "MAPE": float(abs_pct_error.mean()),
        "R2": float(r2_score(actual, predicted)) if len(actual) > 1 else np.nan,
    }


def compare_model_to_auction_midpoint(
    frame: pd.DataFrame,
    predicted_price: pd.Series | np.ndarray,
    *,
    actual_col: str = "target_price_usd",
    auction_col: str = "auction_midpoint_usd",
    model_label: str = "LightGBM",
) -> pd.DataFrame:
    """Compare model predictions with the auction midpoint baseline."""

    rows = [
        evaluate_price_predictions(frame[actual_col], predicted_price, label=model_label),
        evaluate_price_predictions(frame[actual_col], frame[auction_col], label="Auction midpoint"),
    ]
    return pd.DataFrame(rows)


def evaluate_by_quarter(
    frame: pd.DataFrame,
    predicted_price: pd.Series | np.ndarray,
    *,
    actual_col: str = "target_price_usd",
    quarter_col: str = "source_quarter",
    label: str = "LightGBM",
) -> pd.DataFrame:
    """Evaluate predictions separately for each explicit quarterly holdout."""

    work = frame.copy()
    work["predicted_price"] = predicted_price
    rows = []
    for quarter, group in work.groupby(quarter_col, dropna=False):
        row = evaluate_price_predictions(group[actual_col], group["predicted_price"], label=label)
        row["quarter"] = quarter
        rows.append(row)
    return pd.DataFrame(rows).sort_values("quarter").reset_index(drop=True)


def evaluate_bias_by_price_bucket(
    frame: pd.DataFrame,
    predicted_price: pd.Series | np.ndarray,
    *,
    actual_col: str = "target_price_usd",
    label: str = "LightGBM",
) -> pd.DataFrame:
    """Evaluate bias by actual-price range."""

    work = frame.copy()
    work["predicted_price"] = predicted_price
    work["price_bucket"] = price_bucket(work[actual_col])
    rows = []
    for bucket, group in work.groupby("price_bucket", observed=False):
        if group.empty:
            continue
        row = evaluate_price_predictions(group[actual_col], group["predicted_price"], label=label)
        row["price_bucket"] = str(bucket)
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def feature_importance_frame(model, feature_cols: Iterable[str]) -> pd.DataFrame:
    """Return split/gain feature importances when available."""

    feature_cols = list(feature_cols)
    if hasattr(model, "booster_"):
        booster = model.booster_
        split = booster.feature_importance(importance_type="split")
        gain = booster.feature_importance(importance_type="gain")
    elif hasattr(model, "feature_importances_"):
        split = model.feature_importances_
        gain = np.full(len(split), np.nan)
    else:
        raise AttributeError("Model does not expose LightGBM feature importance.")

    return (
        pd.DataFrame({"feature": feature_cols, "split_importance": split, "gain_importance": gain})
        .sort_values(["gain_importance", "split_importance"], ascending=False)
        .reset_index(drop=True)
    )


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(f"[lightgbm model] {message}")
