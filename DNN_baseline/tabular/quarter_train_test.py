#!/usr/bin/env python3
import argparse
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RUNNER_DIR = Path(__file__).resolve().parent
if str(RUNNER_DIR) not in sys.path:
    sys.path.insert(0, str(RUNNER_DIR))

TARGET_COL = "ah_bp_price_usd"
QUARTER_FILE_RE = re.compile(r"(?P<quarter>\d{4}Q[1-4])")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train TabM on one CSV/JSON file and test on one quarter CSV/JSON file."
        )
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path(
            "Data/sampled_sets/"
            "Artwork_2026_finetune_40000_2023_2024_train_sample10000.csv"
        ),
        help="Training CSV/JSON file.",
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        default=Path(
            "Data/sampled_sets/"
            "Artwork_2026_finetune_40000_q2025Q1_sample300.csv"
        ),
        help="Testing CSV/JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for metrics and predictions. Defaults to tabular/outputs/<asset-type>/quarter_eval.",
    )
    parser.add_argument(
        "--drop-cols",
        nargs="*",
        default=["ID", "image_url", "lot_url", "auction_url"],
        help="Columns to drop from features when present.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--k", type=int, default=32, help="TabM ensemble size.")
    parser.add_argument(
        "--use-estimates",
        action="store_true",
        help="Include ah_low_estimate_usd and ah_high_estimate_usd as features.",
    )
    parser.add_argument(
        "--asset-type",
        choices=["artwork", "jewelry", "watch"],
        default="artwork",
        help="Feature preprocessing schema to use.",
    )
    parser.add_argument(
        "--target-col",
        default=TARGET_COL,
        help=f"Target column to predict. Defaults to {TARGET_COL}.",
    )
    parser.add_argument(
        "--save-model-path",
        type=Path,
        default=None,
        help="Where to save model checkpoint. Defaults to <output-dir>/model.pt.",
    )
    parser.add_argument(
        "--load-model-path",
        type=Path,
        default=None,
        help="Load model checkpoint and skip training.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed step-by-step logging.",
    )
    return parser.parse_args()


def quarter_from_filename(path: Path) -> str | None:
    match = QUARTER_FILE_RE.search(path.name)
    if not match:
        return None
    return match.group("quarter")


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported input format for {path}. Use .csv or .json")


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def stable_target_log1p(y: np.ndarray) -> np.ndarray:
    """Numerically stable log transform for non-negative target."""
    y = np.asarray(y, dtype=np.float64)
    y = np.clip(y, a_min=0.0, a_max=None)
    return np.log1p(y)


def stable_pred_expm1(y_log: np.ndarray) -> np.ndarray:
    """Numerically stable inverse transform with overflow guard."""
    y_log = np.asarray(y_log, dtype=np.float64)
    # exp(40) is already far beyond observed target scales here.
    y_log = np.clip(y_log, a_min=-20.0, a_max=40.0)
    y = np.expm1(y_log)
    return np.clip(y, a_min=0.0, a_max=None)


def load_tabm_class(repo_root: Path):
    tabm_candidates = [
        repo_root / "tabm" / "tabm.py",
        repo_root.parent / "tabm" / "tabm.py",
    ]
    tabm_path = next((path for path in tabm_candidates if path.exists()), None)
    if tabm_path is None:
        searched = ", ".join(str(path) for path in tabm_candidates)
        raise FileNotFoundError(f"TabM source not found. Searched: {searched}")
    spec = importlib.util.spec_from_file_location("tabm_local", tabm_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load tabm module spec.")
    module = importlib.util.module_from_spec(spec)
    sys.modules["tabm_local"] = module
    spec.loader.exec_module(module)
    return module.TabM


from asset_features import preprocess_features


def evaluate_once(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_quarter: str,
    target_col: str,
    drop_cols: list[str],
    output_dir: Path,
    epochs: int,
    early_stopping_patience: int,
    val_fraction: float,
    batch_size: int,
    lr: float,
    weight_decay: float,
    k: int,
    use_estimates: bool,
    asset_type: str,
    verbose: bool,
    save_model_path: Path | None,
    load_model_path: Path | None,
) -> dict[str, float | int | str]:
    train_df = train_df[train_df[target_col].notna()].copy()
    test_df = test_df[test_df[target_col].notna()].copy()

    if verbose:
        print(f"[INFO] Rows after target filter: train={len(train_df)}, test={len(test_df)}")

    prep = preprocess_features(
        train_df,
        test_df,
        target_col=target_col,
        drop_cols=drop_cols,
        use_estimates=use_estimates,
        asset_type=asset_type,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"[INFO] Device: {device}")
        print(
            f"[INFO] Features: n_num={prep['n_num_features']}, "
            f"n_cat={0 if prep['cat_cardinalities'] is None else len(prep['cat_cardinalities'])}"
        )

    repo_root = Path(__file__).resolve().parent
    TabM = load_tabm_class(repo_root)

    model = TabM.make(
        n_num_features=prep["n_num_features"],
        cat_cardinalities=prep["cat_cardinalities"],
        d_out=1,
        k=k,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    x_num_train = (
        None
        if prep["x_num_train"] is None
        else torch.as_tensor(prep["x_num_train"], dtype=torch.float32, device=device)
    )
    x_cat_train = (
        None
        if prep["x_cat_train"] is None
        else torch.as_tensor(prep["x_cat_train"], dtype=torch.long, device=device)
    )
    y_train = torch.as_tensor(
        stable_target_log1p(train_df[target_col].astype(float).to_numpy()),
        dtype=torch.float32,
        device=device,
    )

    x_num_test = (
        None
        if prep["x_num_test"] is None
        else torch.as_tensor(prep["x_num_test"], dtype=torch.float32, device=device)
    )
    x_cat_test = (
        None
        if prep["x_cat_test"] is None
        else torch.as_tensor(prep["x_cat_test"], dtype=torch.long, device=device)
    )
    y_test = test_df[target_col].astype(float).to_numpy()

    if load_model_path is not None:
        checkpoint = torch.load(load_model_path, map_location=device)
        state = (
            checkpoint["model_state_dict"]
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
            else checkpoint
        )
        model.load_state_dict(state)
        if verbose:
            print(f"[INFO] Loaded model checkpoint: {load_model_path}")
    else:
        model.train()
        n = y_train.shape[0]
        n_val = max(1, int(n * val_fraction))
        n_train_idx = max(1, n - n_val)
        perm_all = torch.randperm(n, device=device)
        idx_train = perm_all[:n_train_idx]
        idx_val = perm_all[n_train_idx:]

        best_val = float("inf")
        best_epoch = 0
        bad_epochs = 0
        best_state = None
        t0 = time.time()

        for epoch in range(1, epochs + 1):
            epoch_t0 = time.time()
            perm = idx_train[torch.randperm(n_train_idx, device=device)]
            epoch_loss = 0.0
            seen = 0

            for idx in perm.split(batch_size):
                xb_num = None if x_num_train is None else x_num_train[idx]
                xb_cat = None if x_cat_train is None else x_cat_train[idx]
                yb = y_train[idx]

                pred = model(xb_num, xb_cat).squeeze(-1)  # (B, k)
                loss = ((pred - yb[:, None]) ** 2).mean()

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                batch_n = idx.shape[0]
                epoch_loss += float(loss.item()) * batch_n
                seen += batch_n

            # Validation loss for early stopping (batched to avoid GPU OOM).
            model.eval()
            val_num = 0.0
            val_den = 0
            with torch.no_grad():
                for vidx in idx_val.split(batch_size):
                    vb_num = None if x_num_train is None else x_num_train[vidx]
                    vb_cat = None if x_cat_train is None else x_cat_train[vidx]
                    yv = y_train[vidx]
                    pv = model(vb_num, vb_cat).squeeze(-1)
                    vloss = ((pv - yv[:, None]) ** 2).mean()
                    bsz = int(vidx.shape[0])
                    val_num += float(vloss.item()) * bsz
                    val_den += bsz
            val_loss = val_num / max(val_den, 1)
            model.train()

            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                bad_epochs = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad_epochs += 1

            if verbose and (
                epoch == 1
                or epoch % max(1, early_stopping_patience) == 0
                or epoch == epochs
                or bad_epochs >= early_stopping_patience
            ):
                elapsed = time.time() - t0
                epoch_sec = time.time() - epoch_t0
                avg_epoch = elapsed / max(epoch, 1)
                eta = max(0.0, (epochs - epoch) * avg_epoch)
                print(
                    f"[INFO] Epoch {epoch}/{epochs} "
                    f"train_mse={epoch_loss / max(seen, 1):.6f} "
                    f"val_mse={val_loss:.6f} best_val={best_val:.6f} bad_epochs={bad_epochs} "
                    f"epoch_sec={epoch_sec:.1f}s elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m"
                )

            if bad_epochs >= early_stopping_patience:
                if verbose:
                    print(f"[INFO] Early stopping at epoch {epoch}, restoring best epoch {best_epoch}")
                break

        if best_state is not None:
            model.load_state_dict(best_state)

    saved_model = str(load_model_path) if load_model_path is not None else None
    if save_model_path is not None and load_model_path is None:
        save_model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "k": k,
                "use_estimates": use_estimates,
                "asset_type": asset_type,
                "target": target_col,
            },
            save_model_path,
        )
        saved_model = str(save_model_path)
        if verbose:
            print(f"[INFO] Saved model checkpoint: {saved_model}")

    model.eval()
    with torch.no_grad():
        pred_log = model(x_num_test, x_cat_test).squeeze(-1).mean(dim=1).detach().cpu().numpy()

    y_pred = stable_pred_expm1(pred_log)

    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))
    mape = safe_mape(y_test, y_pred)

    pred_df = pd.DataFrame(
        {
            "quarter_tested": test_quarter,
            "y_true": y_test,
            "y_pred": y_pred,
            "abs_error": np.abs(y_test - y_pred),
        }
    )
    pred_path = output_dir / f"predictions_{test_quarter}.csv"
    pred_df.to_csv(pred_path, index=False)

    return {
        "quarter_tested": test_quarter,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "mae": mae,
        "rmse": rmse,
        "mape_pct": mape,
        "r2": r2,
        "device": str(device),
        "predictions_file": str(pred_path),
        "model_file": saved_model,
    }


def main() -> None:
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = RUNNER_DIR / "outputs" / args.asset_type / "quarter_eval"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_table(args.train_file)
    test_df = load_table(args.test_file)
    test_quarter = quarter_from_filename(args.test_file) or "Q_UNKNOWN"

    if args.verbose:
        print(f"[INFO] Train file: {args.train_file}")
        print(f"[INFO] Test file:  {args.test_file}")
        print(f"[INFO] Test quarter tag: {test_quarter}")

    result = evaluate_once(
        train_df=train_df,
        test_df=test_df,
        test_quarter=test_quarter,
        target_col=args.target_col,
        drop_cols=args.drop_cols,
        output_dir=args.output_dir,
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        val_fraction=args.val_fraction,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        k=args.k,
        use_estimates=args.use_estimates,
        asset_type=args.asset_type,
        verbose=args.verbose,
        save_model_path=args.save_model_path or (args.output_dir / "model.pt"),
        load_model_path=args.load_model_path,
    )

    print(
        f"{test_quarter} | device={result['device']} | "
        f"n_train={result['n_train']} n_test={result['n_test']} "
        f"MAE={result['mae']:.2f} RMSE={result['rmse']:.2f} "
        f"MAPE={result['mape_pct']:.2f}% R2={result['r2']:.4f}"
    )

    metrics_df = pd.DataFrame([result]).sort_values("quarter_tested")
    metrics_csv = args.output_dir / "metrics_by_quarter.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    summary = {
        "target": args.target_col,
        "train_file": str(args.train_file),
        "test_file": str(args.test_file),
        "test_quarter": test_quarter,
        "epochs": args.epochs,
        "early_stopping_patience": args.early_stopping_patience,
        "val_fraction": args.val_fraction,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "k": args.k,
        "use_estimates": args.use_estimates,
        "asset_type": args.asset_type,
        "device": result["device"],
        "mae": float(metrics_df["mae"].iloc[0]),
        "rmse": float(metrics_df["rmse"].iloc[0]),
        "mape_pct": float(metrics_df["mape_pct"].iloc[0]),
        "r2": float(metrics_df["r2"].iloc[0]),
        "metrics_file": str(metrics_csv),
        "model_file": result["model_file"],
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nSaved: {metrics_csv}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
