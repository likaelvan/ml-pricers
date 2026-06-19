from __future__ import annotations

from itertools import product
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_ALPHAS = [0.0001, 0.001, 0.01]
DEFAULT_EPSILONS = [1.35, 1.75]


def build_huber_pipeline(
    categorical_cols: Iterable[str],
    numeric_cols: Iterable[str],
    *,
    alpha: float = 0.0001,
    epsilon: float = 1.35,
    max_iter: int = 300,
) -> Pipeline:
    """Build a robust log-linear Huber regression pipeline."""

    categorical_cols = list(categorical_cols)
    numeric_cols = list(numeric_cols)
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

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
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "huber",
                HuberRegressor(
                    alpha=float(alpha),
                    epsilon=float(epsilon),
                    max_iter=int(max_iter),
                ),
            ),
        ]
    )


def fit_huber_log_price_model(
    X_train: pd.DataFrame,
    y_train_log: pd.Series,
    *,
    categorical_cols: Iterable[str],
    numeric_cols: Iterable[str],
    alphas: Iterable[float] = DEFAULT_ALPHAS,
    epsilons: Iterable[float] = DEFAULT_EPSILONS,
    validation_fraction: float = 0.10,
    eval_metric: str = "log_RMSE",
    random_state: int = 42,
    max_iter: int = 300,
    verbose: bool = False,
) -> Pipeline:
    """Fit Huber on log1p USD prices with grid diagnostics."""

    alphas = [float(alpha) for alpha in alphas]
    epsilons = [float(epsilon) for epsilon in epsilons]
    categorical_cols = list(categorical_cols)
    numeric_cols = list(numeric_cols)
    _log(verbose, f"Training Huber on {len(X_train):,} rows and {X_train.shape[1]:,} features")
    _log(verbose, f"Alpha grid: {alphas}")
    _log(verbose, f"Epsilon grid: {epsilons}")

    history = pd.DataFrame()
    selected_alpha = alphas[0]
    selected_epsilon = epsilons[0]

    if validation_fraction and validation_fraction > 0:
        X_fit, X_valid, y_fit, y_valid = train_test_split(
            X_train,
            y_train_log,
            test_size=validation_fraction,
            random_state=random_state,
        )
        rows = []
        combos = list(product(alphas, epsilons))
        _log(
            verbose,
            "Running Huber diagnostics: "
            f"train_fit={len(X_fit):,}, validation={len(X_valid):,}, "
            f"candidates={len(combos):,}, eval_metric={eval_metric}",
        )
        for iteration, (alpha, epsilon) in enumerate(combos, start=1):
            candidate = build_huber_pipeline(
                categorical_cols,
                numeric_cols,
                alpha=alpha,
                epsilon=epsilon,
                max_iter=max_iter,
            )
            candidate.fit(X_fit, y_fit)
            pred_log = candidate.predict(X_valid)
            actual_price = np.expm1(y_valid)
            predicted_price = np.expm1(pred_log).clip(min=0)
            row = {
                "iteration": iteration,
                "alpha": alpha,
                "epsilon": epsilon,
                "log_MAE": float(mean_absolute_error(y_valid, pred_log)),
                "log_RMSE": float(np.sqrt(mean_squared_error(y_valid, pred_log))),
                "log_R2": float(r2_score(y_valid, pred_log)) if len(y_valid) > 1 else np.nan,
                "MAE": float(mean_absolute_error(actual_price, predicted_price)),
                "RMSE": float(np.sqrt(mean_squared_error(actual_price, predicted_price))),
                "MAPE": float(np.mean(np.abs((predicted_price - actual_price) / actual_price))),
                "R2": float(r2_score(actual_price, predicted_price)) if len(actual_price) > 1 else np.nan,
                "n_iter": int(candidate.named_steps["huber"].n_iter_),
                "outlier_fraction": float(np.mean(candidate.named_steps["huber"].outliers_)),
            }
            rows.append(row)
            _log(
                verbose,
                f"[{iteration:02d}/{len(combos):02d}] alpha={alpha:.6g} epsilon={epsilon:.3g} "
                f"log_RMSE={row['log_RMSE']:.5f} log_MAE={row['log_MAE']:.5f} "
                f"RMSE={row['RMSE']:.2f} MAE={row['MAE']:.2f} "
                f"outlier_fraction={row['outlier_fraction']:.3f}",
            )

        history = pd.DataFrame(rows)
        selected = _select_candidate_from_history(history, eval_metric)
        selected_alpha = float(selected["alpha"])
        selected_epsilon = float(selected["epsilon"])
        _log(verbose, f"Selected alpha={selected_alpha}, epsilon={selected_epsilon}")

    model = build_huber_pipeline(
        categorical_cols,
        numeric_cols,
        alpha=selected_alpha,
        epsilon=selected_epsilon,
        max_iter=max_iter,
    )
    model.fit(X_train, y_train_log)
    huber = model.named_steps["huber"]
    model.training_history_ = history
    model.selected_alpha_ = selected_alpha
    model.selected_epsilon_ = selected_epsilon
    model.outlier_fraction_ = float(np.mean(huber.outliers_))
    model.n_iter_ = int(huber.n_iter_)
    _log(
        verbose,
        f"Final fit: alpha={selected_alpha}, epsilon={selected_epsilon}, "
        f"n_iter={model.n_iter_}, outlier_fraction={model.outlier_fraction_:.3f}",
    )
    return model


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
    label: str = "Huber Regression",
) -> pd.DataFrame:
    """Evaluate predictions by quarter."""

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
    label: str = "Huber Regression",
) -> pd.DataFrame:
    """Evaluate predictions by actual-price bucket."""

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


def huber_training_history(model: Pipeline) -> pd.DataFrame:
    """Return validation diagnostics from fit_huber_log_price_model."""

    history = getattr(model, "training_history_", None)
    if history is None:
        return pd.DataFrame()
    return history.copy()


def get_top_coefficients(model: Pipeline, *, top_n: int = 30) -> pd.DataFrame:
    """Return largest positive/negative Huber coefficients after preprocessing."""

    preprocessor = model.named_steps["preprocessor"]
    huber = model.named_steps["huber"]
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = np.array([f"feature_{idx}" for idx in range(len(huber.coef_))])

    coefs = pd.DataFrame({"feature": feature_names, "coefficient": huber.coef_})
    coefs["abs_coefficient"] = coefs["coefficient"].abs()
    return coefs.sort_values("abs_coefficient", ascending=False).head(top_n).reset_index(drop=True)


def _select_candidate_from_history(history: pd.DataFrame, eval_metric: str) -> pd.Series:
    metric = eval_metric if eval_metric in history.columns else "log_RMSE"
    ascending = metric not in {"R2", "log_R2"}
    best_idx = history[metric].astype(float).sort_values(ascending=ascending).index[0]
    return history.loc[best_idx]


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(f"[huber model] {message}")
