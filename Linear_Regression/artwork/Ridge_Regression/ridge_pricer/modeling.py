from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_ALPHAS = np.logspace(-3, 4, 16)


def build_ridge_pipeline(
    categorical_cols: Iterable[str],
    numeric_cols: Iterable[str],
    *,
    alphas: Iterable[float] = DEFAULT_ALPHAS,
    alpha: float | None = None,
) -> Pipeline:
    """Build a regularized log-linear Ridge regression pipeline."""

    categorical_cols = list(categorical_cols)
    numeric_cols = list(numeric_cols)
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=True)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
                        ("onehot", encoder),
                    ]
                ),
                categorical_cols,
            ),
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler(with_mean=False)),
                    ]
                ),
                numeric_cols,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    estimator = RidgeCV(alphas=list(alphas)) if alpha is None else Ridge(alpha=float(alpha))
    return Pipeline(steps=[("preprocessor", preprocessor), ("ridge", estimator)])


def fit_ridge_log_price_model(
    X_train: pd.DataFrame,
    y_train_log: pd.Series,
    *,
    categorical_cols: Iterable[str],
    numeric_cols: Iterable[str],
    alphas: Iterable[float] = DEFAULT_ALPHAS,
    validation_fraction: float = 0.10,
    log_period: int | None = 1,
    eval_metric: str = "log_RMSE",
    random_state: int = 42,
    verbose: bool = False,
) -> Pipeline:
    """Fit a log-price Ridge model with optional alpha-by-alpha diagnostics."""

    alphas = [float(alpha) for alpha in alphas]
    categorical_cols = list(categorical_cols)
    numeric_cols = list(numeric_cols)
    _log(verbose, f"Training Ridge on {len(X_train):,} rows and {X_train.shape[1]:,} features")
    _log(verbose, f"Alpha grid: {alphas}")

    history = pd.DataFrame()
    selected_alpha: float | None = None

    if validation_fraction and validation_fraction > 0:
        X_fit, X_valid, y_fit, y_valid = train_test_split(
            X_train,
            y_train_log,
            test_size=validation_fraction,
            random_state=random_state,
        )
        _log(
            verbose,
            "Running alpha diagnostics: "
            f"train_fit={len(X_fit):,}, validation={len(X_valid):,}, "
            f"eval_metric={eval_metric}, log_period={log_period}",
        )
        rows = []
        for idx, alpha in enumerate(alphas, start=1):
            candidate = build_ridge_pipeline(categorical_cols, numeric_cols, alpha=alpha)
            candidate.fit(X_fit, y_fit)
            pred_log = candidate.predict(X_valid)
            actual_price = np.expm1(y_valid)
            predicted_price = np.expm1(pred_log).clip(min=0)
            row = {
                "iteration": idx,
                "alpha": alpha,
                "log_MAE": float(mean_absolute_error(y_valid, pred_log)),
                "log_RMSE": float(np.sqrt(mean_squared_error(y_valid, pred_log))),
                "log_R2": float(r2_score(y_valid, pred_log)) if len(y_valid) > 1 else np.nan,
                "MAE": float(mean_absolute_error(actual_price, predicted_price)),
                "RMSE": float(np.sqrt(mean_squared_error(actual_price, predicted_price))),
                "MAPE": float(np.mean(np.abs((predicted_price - actual_price) / actual_price))),
                "R2": float(r2_score(actual_price, predicted_price)) if len(actual_price) > 1 else np.nan,
            }
            rows.append(row)
            if verbose and log_period and (idx == 1 or idx % log_period == 0 or idx == len(alphas)):
                _log(
                    True,
                    f"[{idx:02d}/{len(alphas):02d}] alpha={alpha:.6g} "
                    f"log_RMSE={row['log_RMSE']:.5f} log_MAE={row['log_MAE']:.5f} "
                    f"RMSE={row['RMSE']:.2f} MAE={row['MAE']:.2f}",
                )

        history = pd.DataFrame(rows)
        selected_alpha = _select_alpha_from_history(history, eval_metric)
        _log(verbose, f"Selected alpha from validation diagnostics: {selected_alpha}")
        model = build_ridge_pipeline(categorical_cols, numeric_cols, alpha=selected_alpha)
        model.fit(X_train, y_train_log)
        model.named_steps["ridge"].alpha_ = selected_alpha
    else:
        _log(verbose, "Training RidgeCV without explicit validation diagnostics")
        model = build_ridge_pipeline(categorical_cols, numeric_cols, alphas=alphas)
        model.fit(X_train, y_train_log)
        selected_alpha = float(model.named_steps["ridge"].alpha_)
        _log(verbose, f"Selected alpha from RidgeCV: {selected_alpha}")

    model.training_history_ = history
    model.selected_alpha_ = selected_alpha
    return model


def ridge_training_history(model: Pipeline) -> pd.DataFrame:
    """Return alpha diagnostics from fit_ridge_log_price_model when available."""

    history = getattr(model, "training_history_", None)
    if history is None:
        return pd.DataFrame()
    return history.copy()


def predict_price_from_log_model(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Back-transform log1p predictions to USD prices."""

    pred_log = model.predict(X)
    return np.expm1(pred_log).clip(min=0)


def evaluate_price_predictions(
    actual: pd.Series,
    predicted: pd.Series | np.ndarray,
    *,
    label: str,
) -> dict[str, float | str | int]:
    """Return only MAE, RMSE, MAPE, and R2."""

    actual = pd.Series(actual, dtype=float)
    predicted = pd.Series(predicted, index=actual.index, dtype=float)
    mask = actual.gt(0) & predicted.notna() & np.isfinite(predicted)
    actual = actual[mask]
    predicted = predicted[mask]
    mape = (((predicted - actual) / actual).abs()).mean()
    return {
        "model": label,
        "n": int(len(actual)),
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "MAPE": float(mape),
        "R2": float(r2_score(actual, predicted)) if len(actual) > 1 else np.nan,
    }


def evaluate_by_quarter(
    frame: pd.DataFrame,
    predicted_price: pd.Series | np.ndarray,
    *,
    actual_col: str = "target_price_usd",
    quarter_col: str = "source_quarter",
    label: str = "Ridge Regression",
) -> pd.DataFrame:
    """Evaluate Ridge predictions by quarter."""

    work = frame.copy()
    work["predicted_price"] = predicted_price
    rows = []
    for quarter, group in work.groupby(quarter_col, dropna=False):
        row = evaluate_price_predictions(group[actual_col], group["predicted_price"], label=label)
        row["quarter"] = quarter
        rows.append(row)
    return pd.DataFrame(rows).sort_values("quarter").reset_index(drop=True)


def evaluate_by_price_bucket(
    frame: pd.DataFrame,
    predicted_price: pd.Series | np.ndarray,
    *,
    actual_col: str = "target_price_usd",
    label: str = "Ridge Regression",
) -> pd.DataFrame:
    """Evaluate Ridge predictions by actual-price bucket."""

    work = frame.copy()
    work["predicted_price"] = predicted_price
    work["price_bucket"] = pd.cut(
        work[actual_col],
        bins=[0, 1_000, 5_000, 25_000, 100_000, 500_000, np.inf],
        labels=["<$1k", "$1k-$5k", "$5k-$25k", "$25k-$100k", "$100k-$500k", "$500k+"],
        include_lowest=True,
    )
    rows = []
    for bucket, group in work.groupby("price_bucket", observed=False):
        if group.empty:
            continue
        row = evaluate_price_predictions(group[actual_col], group["predicted_price"], label=label)
        row["price_bucket"] = str(bucket)
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def get_top_coefficients(
    model: Pipeline,
    *,
    top_n: int = 30,
) -> pd.DataFrame:
    """Return largest positive/negative Ridge coefficients after preprocessing."""

    preprocessor = model.named_steps["preprocessor"]
    ridge = model.named_steps["ridge"]
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = np.array([f"feature_{idx}" for idx in range(len(ridge.coef_))])

    coefs = pd.DataFrame({"feature": feature_names, "coefficient": ridge.coef_})
    coefs["abs_coefficient"] = coefs["coefficient"].abs()
    return coefs.sort_values("abs_coefficient", ascending=False).head(top_n).reset_index(drop=True)


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(f"[ridge model] {message}")


def _select_alpha_from_history(history: pd.DataFrame, eval_metric: str) -> float:
    metric = eval_metric if eval_metric in history.columns else "log_RMSE"
    ascending = metric not in {"R2", "log_R2"}
    best_idx = history[metric].astype(float).sort_values(ascending=ascending).index[0]
    return float(history.loc[best_idx, "alpha"])
