from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from watch_ridge_pricer import (
    apply_feature_schema,
    build_feature_report,
    evaluate_by_price_bucket,
    evaluate_by_quarter,
    evaluate_price_predictions,
    fit_feature_schema,
    fit_ridge_log_price_model,
    get_top_coefficients,
    load_watch_data,
    make_model_frame,
    predict_price_from_log_model,
    ridge_training_history,
    split_train_test,
)


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = HERE / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train watch Ridge regression baselines.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-col", default="ah_price_usd")
    parser.add_argument("--train-before", default="2025-01-01")
    parser.add_argument("--train-rows", type=int, default=None)
    parser.add_argument("--test-rows-per-quarter", type=int, default=None)
    parser.add_argument("--min-category-count", type=int, default=20)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_watch_data(train_before=args.train_before, verbose=args.verbose)
    train_raw, test_raw = split_train_test(
        raw,
        train_rows=args.train_rows,
        test_rows_per_quarter=args.test_rows_per_quarter,
        random_state=args.random_state,
        verbose=args.verbose,
    )

    predictions = test_raw[
        [c for c in ["ID", "lot_url", "source_file", "source_quarter", "sale_date", args.target_col] if c in test_raw]
    ].copy()
    predictions = predictions.rename(columns={args.target_col: "actual_price_usd"})

    overall_rows = []
    quarter_rows = []
    bucket_rows = []
    history_frames = []
    coef_frames = []

    for include_estimates, suffix, label in [
        (False, "no_estimates", "Watch Ridge - No Estimates"),
        (True, "with_estimates", "Watch Ridge - With Log Estimates"),
    ]:
        train_frame, feature_cols, categorical_cols, numeric_cols = make_model_frame(
            train_raw,
            target_col=args.target_col,
            include_estimates=include_estimates,
            verbose=args.verbose,
        )
        test_frame, _, _, _ = make_model_frame(
            test_raw,
            target_col=args.target_col,
            include_estimates=include_estimates,
            verbose=args.verbose,
        )

        schema = fit_feature_schema(
            train_frame,
            feature_cols,
            categorical_cols,
            min_category_count=args.min_category_count,
            verbose=args.verbose,
        )
        x_train = apply_feature_schema(train_frame, feature_cols, categorical_cols, schema, verbose=args.verbose)
        x_test = apply_feature_schema(test_frame, feature_cols, categorical_cols, schema, verbose=args.verbose)

        model = fit_ridge_log_price_model(
            x_train,
            train_frame["log_target_price_usd"],
            categorical_cols=categorical_cols,
            numeric_cols=numeric_cols,
            validation_fraction=args.validation_fraction,
            random_state=args.random_state,
            verbose=args.verbose,
        )
        pred = predict_price_from_log_model(model, x_test)
        predictions[f"ridge_pred_{suffix}_usd"] = pred

        overall_rows.append(evaluate_price_predictions(test_frame["target_price_usd"], pred, label=label))
        quarter_rows.append(
            evaluate_by_quarter(test_frame, pred, actual_col="target_price_usd", label=label)
        )
        bucket_rows.append(
            evaluate_by_price_bucket(test_frame, pred, actual_col="target_price_usd", label=label)
        )

        feature_report = build_feature_report(train_frame, feature_cols, categorical_cols, schema=schema)
        feature_report.to_csv(args.output_dir / f"watch_ridge_feature_report_{suffix}.csv", index=False)

        history = ridge_training_history(model)
        if not history.empty:
            history.insert(0, "model", label)
            history_frames.append(history)

        coefs = get_top_coefficients(model, top_n=40)
        coefs.insert(0, "model", label)
        coef_frames.append(coefs)

    predictions.to_csv(args.output_dir / "watch_ridge_test_predictions.csv", index=False)
    pd.DataFrame(overall_rows).to_csv(args.output_dir / "watch_ridge_overall_metrics.csv", index=False)
    pd.concat(quarter_rows, ignore_index=True).to_csv(args.output_dir / "watch_ridge_quarter_metrics.csv", index=False)
    pd.concat(bucket_rows, ignore_index=True).to_csv(args.output_dir / "watch_ridge_price_bucket_metrics.csv", index=False)
    if history_frames:
        pd.concat(history_frames, ignore_index=True).to_csv(args.output_dir / "watch_ridge_training_history.csv", index=False)
    pd.concat(coef_frames, ignore_index=True).to_csv(args.output_dir / "watch_ridge_top_coefficients.csv", index=False)

    print(f"[watch ridge] outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
