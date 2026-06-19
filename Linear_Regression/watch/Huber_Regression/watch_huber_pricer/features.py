from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
WATCH_DATA_DIR = ROOT / "Data" / "watch"
TRAIN_FILE = WATCH_DATA_DIR / "Watch_2026_finetune_19345.json"
TARGET_COL = "ah_price_usd"
MARKET_COLS = ["GSPC", "FTSE", "IXIC", "DJI", "VIX", "BZ_F", "FVX"]

CATEGORICAL_FEATURES = [
    "auction_house",
    "location",
    "brand_clean",
    "model_clean",
    "ref_no_clean",
    "box_clean",
    "papers_clean",
    "condition_clean",
    "case_material_clean",
    "band_material_clean",
    "dial_color_clean",
    "gold_present_clean",
    "gold_type_clean",
    "diamonds_clean",
]

NUMERIC_FEATURES = [
    "sale_date_ordinal",
    "sale_year_num",
    "sale_month_num",
    "year_manufactured_num",
    "case_size_num",
    "karats_num",
    "diamonds_quantity_num",
    *MARKET_COLS,
]

ESTIMATE_NUMERIC_FEATURES = [
    "log_auction_low_estimate_usd",
    "log_auction_high_estimate_usd",
    "log_auction_estimate_midpoint_usd",
    "auction_estimate_range_ratio",
]

ESTIMATE_REPORTING_FEATURES = [
    "auction_low_estimate_usd",
    "auction_high_estimate_usd",
    "auction_estimate_midpoint_usd",
    "auction_estimate_range_usd",
    *ESTIMATE_NUMERIC_FEATURES,
]

FEATURE_TRANSFORMATIONS = {
    "auction_house": "Cleaned categorical from auction_house; rare levels grouped after schema fitting.",
    "location": "Cleaned categorical from sale location; rare levels grouped after schema fitting.",
    "brand_clean": "Cleaned categorical from watch brand; rare brands grouped after schema fitting.",
    "model_clean": "Cleaned categorical from watch model; rare models grouped after schema fitting.",
    "ref_no_clean": "Cleaned categorical from reference number; rare references grouped after schema fitting.",
    "box_clean": "Normalized box indicator to yes/no/missing/other and one-hot encoded.",
    "papers_clean": "Normalized papers indicator to yes/no/missing/other and one-hot encoded.",
    "condition_clean": "Cleaned categorical from condition.",
    "case_material_clean": "Cleaned categorical from case material.",
    "band_material_clean": "Cleaned categorical from band material.",
    "dial_color_clean": "Cleaned categorical from dial color.",
    "gold_present_clean": "Normalized gold-present indicator to yes/no/missing/other and one-hot encoded.",
    "gold_type_clean": "Cleaned categorical from gold type.",
    "diamonds_clean": "Normalized diamonds indicator to yes/no/missing/other and one-hot encoded.",
    "sale_date_ordinal": "Numeric ordinal day derived from sale_date.",
    "sale_year_num": "Numeric sale year derived from sale_date.",
    "sale_month_num": "Numeric sale month derived from sale_date.",
    "year_manufactured_num": "Numeric year manufactured.",
    "case_size_num": "Numeric case size coerced from case_size.",
    "karats_num": "Numeric gold karat value coerced from karats.",
    "diamonds_quantity_num": "Numeric diamond count coerced from diamonds_quantity.",
    "GSPC": "Numeric market index level.",
    "FTSE": "Numeric market index level.",
    "IXIC": "Numeric market index level.",
    "DJI": "Numeric market index level.",
    "VIX": "Numeric market/risk index level.",
    "BZ_F": "Numeric Brent crude proxy.",
    "FVX": "Numeric 5-year treasury yield proxy.",
    "auction_low_estimate_usd": "Cleaned auction-house low estimate in USD; retained for reporting only.",
    "auction_high_estimate_usd": "Cleaned auction-house high estimate in USD; retained for reporting only.",
    "auction_estimate_midpoint_usd": "Cleaned midpoint of auction-house low/high estimates; retained for reporting only.",
    "auction_estimate_range_usd": "Cleaned spread between auction-house high and low estimates; retained for reporting only.",
    "log_auction_low_estimate_usd": "log1p(cleaned auction-house low estimate USD); used in the with-estimates model.",
    "log_auction_high_estimate_usd": "log1p(cleaned auction-house high estimate USD); used in the with-estimates model.",
    "log_auction_estimate_midpoint_usd": "log1p(cleaned auction estimate midpoint USD); used in the with-estimates model.",
    "auction_estimate_range_ratio": "Cleaned estimate spread divided by cleaned midpoint; used in the with-estimates model.",
}


@dataclass(frozen=True)
class FeatureSchema:
    categorical_levels: dict[str, list[str]]
    numeric_medians: dict[str, float]


def load_watch_data(
    train_file: str | Path = TRAIN_FILE,
    test_dir: str | Path = WATCH_DATA_DIR,
    *,
    train_before: str | None = "2025-01-01",
    quarters: Iterable[str] = ("2025Q1", "2025Q2", "2025Q3", "2025Q4"),
    verbose: bool = False,
) -> pd.DataFrame:
    """Load historical watch training data and explicit 2025 quarterly holdout files."""

    train_file = Path(train_file)
    test_dir = Path(test_dir)
    if not train_file.exists():
        raise FileNotFoundError(f"Training file not found: {train_file.resolve()}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir.resolve()}")

    train = _load_watch_file(train_file)
    train["source_file"] = train_file.name
    train["source_quarter"] = None
    train["is_quarter_test_file"] = False
    train["sale_date"] = pd.to_datetime(train["sale_date"], errors="coerce")
    before_filter = len(train)
    if train_before:
        train = train[train["sale_date"].lt(pd.Timestamp(train_before))].copy()
        _log(verbose, f"Filtered train sale_date < {train_before}: {before_filter:,}->{len(train):,}")

    frames = [train]
    for quarter in quarters:
        path = test_dir / f"watch_sales_{quarter}_dedup_lot_url.json"
        if not path.exists():
            raise FileNotFoundError(f"Quarter test file not found: {path.resolve()}")
        frame = _load_watch_file(path)
        frame["source_file"] = path.name
        frame["source_quarter"] = quarter
        frame["is_quarter_test_file"] = True
        frame["sale_date"] = pd.to_datetime(frame["sale_date"], errors="coerce")
        frames.append(frame)
        _log(verbose, f"Loaded {path.name}: {len(frame):,} rows, {len(frame.columns):,} columns")

    df = pd.concat(frames, ignore_index=True, sort=False)
    _log(verbose, f"Loaded watch frame: {len(df):,} rows, {len(df.columns):,} columns")
    return df


def split_train_test(
    df: pd.DataFrame,
    *,
    train_rows: int | None = None,
    test_rows_per_quarter: int | None = None,
    dedupe_key_col: str = "lot_url",
    cap_oversized_requests: bool = True,
    random_state: int = 42,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split train/test samples using lot_url as the leakage/duplicate key."""

    train = df[~df["is_quarter_test_file"]].copy()
    test = df[df["is_quarter_test_file"]].copy()
    _log(verbose, f"Initial split: train={len(train):,}, explicit test={len(test):,}")

    if dedupe_key_col in train.columns:
        before = len(train)
        train = _drop_duplicate_nonmissing_keys(train, dedupe_key_col)
        _log(verbose, f"De-duplicated train {dedupe_key_col}: {before:,}->{len(train):,}")

    if dedupe_key_col in test.columns:
        before = len(test)
        test = _drop_duplicate_nonmissing_keys(test, dedupe_key_col)
        _log(verbose, f"De-duplicated test {dedupe_key_col}: {before:,}->{len(test):,}")

        before = len(train)
        test_keys = set(test[dedupe_key_col].dropna().astype(str))
        train = train[~train[dedupe_key_col].astype(str).isin(test_keys)].copy()
        _log(verbose, f"Removed test-overlapping {dedupe_key_col} values from train: {before:,}->{len(train):,}")

    if train_rows is not None:
        if len(train) < train_rows:
            if not cap_oversized_requests:
                raise ValueError(f"Requested {train_rows:,} train rows, only {len(train):,} available.")
            _log(verbose, f"Requested {train_rows:,} train rows, only {len(train):,} available; using all.")
        elif len(train) > train_rows:
            train = train.sample(n=train_rows, random_state=random_state).copy()
        _log(verbose, f"Using train rows: {len(train):,}")

    if test_rows_per_quarter is not None:
        sampled = []
        for quarter, group in test.groupby("source_quarter", dropna=False):
            quarter_n = min(test_rows_per_quarter, len(group)) if cap_oversized_requests else test_rows_per_quarter
            if len(group) < test_rows_per_quarter and not cap_oversized_requests:
                raise ValueError(f"Requested {test_rows_per_quarter:,} rows for {quarter}, only {len(group):,}.")
            if len(group) < test_rows_per_quarter:
                _log(verbose, f"Requested {test_rows_per_quarter:,} rows for {quarter}; using {quarter_n:,}.")
            sampled.append(group.sample(n=quarter_n, random_state=random_state))
        test = pd.concat(sampled, ignore_index=True)
        _log(verbose, f"Actual test rows per quarter: {test['source_quarter'].value_counts().sort_index().to_dict()}")

    _log(verbose, f"Final split: train={len(train):,}, test={len(test):,}")
    return train.reset_index(drop=True), test.reset_index(drop=True)


def make_model_frame(
    df: pd.DataFrame,
    *,
    target_col: str = TARGET_COL,
    include_estimates: bool = False,
    verbose: bool = False,
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    """Create the constrained watch regression feature frame."""

    frame = df.copy()
    _log(
        verbose,
        f"Building watch model frame from {len(frame):,} rows using "
        f"target_col={target_col!r}, include_estimates={include_estimates}",
    )

    before = len(frame)
    frame["target_price_usd"] = _to_numeric(frame.get(target_col))
    frame = frame[np.isfinite(frame["target_price_usd"]) & (frame["target_price_usd"] > 0)].copy()
    frame["log_target_price_usd"] = np.log1p(frame["target_price_usd"])
    _log(verbose, f"Filtered invalid/non-positive targets: {before:,}->{len(frame):,}")

    frame["sale_date_ordinal"] = frame["sale_date"].map(lambda value: value.toordinal() if pd.notna(value) else np.nan)
    frame["sale_year_num"] = frame["sale_date"].dt.year.astype(float)
    frame["sale_month_num"] = frame["sale_date"].dt.month.astype(float)
    frame["year_manufactured_num"] = _year_numeric(frame.get("year_manufactured"))
    frame["case_size_num"] = _first_number(frame.get("case_size"))
    frame["karats_num"] = _first_number(frame.get("karats"))
    frame["diamonds_quantity_num"] = _first_number(frame.get("diamonds_quantity"))

    frame["auction_low_estimate_usd"] = _positive_numeric(frame.get("ah_low_estimate_usd"))
    frame["auction_high_estimate_usd"] = _positive_numeric(frame.get("ah_high_estimate_usd"))
    frame["auction_estimate_midpoint_usd"] = (
        frame["auction_low_estimate_usd"] + frame["auction_high_estimate_usd"]
    ) / 2
    frame["auction_estimate_range_usd"] = (
        frame["auction_high_estimate_usd"] - frame["auction_low_estimate_usd"]
    ).where(frame["auction_high_estimate_usd"].ge(frame["auction_low_estimate_usd"]))
    frame["log_auction_low_estimate_usd"] = np.log1p(frame["auction_low_estimate_usd"])
    frame["log_auction_high_estimate_usd"] = np.log1p(frame["auction_high_estimate_usd"])
    frame["log_auction_estimate_midpoint_usd"] = np.log1p(frame["auction_estimate_midpoint_usd"])
    frame["auction_estimate_range_ratio"] = (
        frame["auction_estimate_range_usd"] / frame["auction_estimate_midpoint_usd"]
    )

    frame["auction_house"] = _clean_category(frame.get("auction_house"))
    frame["location"] = _clean_category(frame.get("location"))
    frame["brand_clean"] = _clean_category(frame.get("brand"))
    frame["model_clean"] = _clean_category(frame.get("model"))
    frame["ref_no_clean"] = _clean_category(frame.get("ref_no"))
    frame["box_clean"] = _clean_yes_no(frame.get("box"))
    frame["papers_clean"] = _clean_yes_no(frame.get("papers"))
    frame["condition_clean"] = _clean_category(frame.get("condition"))
    frame["case_material_clean"] = _clean_category(frame.get("case_material"))
    frame["band_material_clean"] = _clean_category(frame.get("band_material"))
    frame["dial_color_clean"] = _clean_category(frame.get("dial_color"))
    frame["gold_present_clean"] = _clean_yes_no(frame.get("gold_present"))
    frame["gold_type_clean"] = _clean_category(frame.get("gold_type"))
    frame["diamonds_clean"] = _clean_yes_no(frame.get("diamonds"))

    for col in MARKET_COLS:
        if col not in frame.columns:
            frame[col] = np.nan
        frame[col] = _to_numeric(frame[col])

    numeric_features = list(NUMERIC_FEATURES)
    if include_estimates:
        numeric_features += ESTIMATE_NUMERIC_FEATURES

    feature_cols = CATEGORICAL_FEATURES + numeric_features
    keep_cols = [
        "ID",
        "lot_url",
        "source_file",
        "source_quarter",
        "is_quarter_test_file",
        "sale_date",
        "target_price_usd",
        "log_target_price_usd",
        *ESTIMATE_REPORTING_FEATURES,
        *feature_cols,
    ]
    keep_cols = list(dict.fromkeys(col for col in keep_cols if col in frame.columns))
    _log(
        verbose,
        f"Engineered {len(feature_cols):,} features "
        f"({len(CATEGORICAL_FEATURES):,} categorical, {len(numeric_features):,} numeric).",
    )
    return frame[keep_cols].copy(), feature_cols, list(CATEGORICAL_FEATURES), numeric_features


def fit_feature_schema(
    frame: pd.DataFrame,
    feature_cols: Iterable[str],
    categorical_cols: Iterable[str],
    *,
    min_category_count: int = 20,
    verbose: bool = False,
) -> FeatureSchema:
    """Fit rare-category grouping and numeric medians on training data only."""

    categorical_cols = list(categorical_cols)
    feature_cols = list(feature_cols)
    _log(verbose, f"Fitting watch feature schema with min_category_count={min_category_count}")

    categorical_levels: dict[str, list[str]] = {}
    for col in categorical_cols:
        values = _clean_category(frame[col] if col in frame.columns else pd.Series(dtype=object))
        counts = values.value_counts(dropna=False)
        kept = counts[counts >= min_category_count].index.tolist()
        levels = sorted(set(kept + ["__missing__", "__other__"]))
        categorical_levels[col] = levels
        _log(verbose, f"  categorical {col}: {len(levels):,} kept levels from {counts.size:,} observed")

    numeric_medians = {}
    for col in feature_cols:
        if col in categorical_cols:
            continue
        values = _to_numeric(frame[col] if col in frame.columns else pd.Series(dtype=float))
        median = values.replace([np.inf, -np.inf], np.nan).median()
        numeric_medians[col] = 0.0 if pd.isna(median) else float(median)
    _log(verbose, f"  numeric medians stored for {len(numeric_medians):,} features")

    return FeatureSchema(categorical_levels=categorical_levels, numeric_medians=numeric_medians)


def apply_feature_schema(
    frame: pd.DataFrame,
    feature_cols: Iterable[str],
    categorical_cols: Iterable[str],
    schema: FeatureSchema,
    *,
    verbose: bool = False,
) -> pd.DataFrame:
    """Apply train-fitted schema to train/test model matrices."""

    feature_cols = list(feature_cols)
    categorical_cols = set(categorical_cols)
    out = pd.DataFrame(index=frame.index)
    _log(verbose, f"Applying watch feature schema to {len(frame):,} rows and {len(feature_cols):,} features")

    for col in feature_cols:
        if col in categorical_cols:
            levels = schema.categorical_levels[col]
            values = _clean_category(frame[col] if col in frame.columns else pd.Series(index=frame.index, dtype=object))
            values = values.where(values.isin(levels), "__other__")
            out[col] = pd.Categorical(values.fillna("__missing__"), categories=levels).astype(str)
        else:
            values = _to_numeric(frame[col] if col in frame.columns else pd.Series(index=frame.index, dtype=float))
            out[col] = values.replace([np.inf, -np.inf], np.nan).fillna(schema.numeric_medians[col]).astype(float)

    return out


def build_feature_report(
    frame: pd.DataFrame,
    feature_cols: Iterable[str],
    categorical_cols: Iterable[str],
    *,
    schema: FeatureSchema | None = None,
) -> pd.DataFrame:
    rows = []
    categorical_cols = set(categorical_cols)
    for feature in feature_cols:
        role = "categorical" if feature in categorical_cols else "numeric"
        series = frame[feature] if feature in frame.columns else pd.Series(index=frame.index, dtype=object)
        missing_rate = float(series.isna().mean()) if len(series) else 0.0
        if role == "categorical":
            cleaned = _clean_category(series)
            unique_count = int(cleaned.nunique(dropna=False))
            schema_levels = len(schema.categorical_levels.get(feature, [])) if schema else np.nan
            examples = ", ".join(cleaned.drop_duplicates().head(3).astype(str).tolist())
        else:
            numeric = _to_numeric(series).replace([np.inf, -np.inf], np.nan)
            unique_count = int(numeric.nunique(dropna=True))
            schema_levels = np.nan
            examples = ", ".join(numeric.dropna().head(3).round(4).astype(str).tolist())
        rows.append(
            {
                "feature": feature,
                "role": role,
                "transformation": FEATURE_TRANSFORMATIONS.get(feature, "No transformation note defined."),
                "missing_rate": missing_rate,
                "unique_count_before_grouping": unique_count,
                "train_schema_levels": schema_levels,
                "example_values": examples,
            }
        )
    return pd.DataFrame(rows)


def print_feature_summary(
    frame: pd.DataFrame,
    feature_cols: Iterable[str],
    categorical_cols: Iterable[str],
    numeric_cols: Iterable[str],
    *,
    schema: FeatureSchema | None = None,
) -> pd.DataFrame:
    feature_cols = list(feature_cols)
    categorical_cols = list(categorical_cols)
    numeric_cols = list(numeric_cols)
    print(f"Training rows: {len(frame):,}")
    print(f"Training features: {len(feature_cols):,}")
    print(f"Categorical features ({len(categorical_cols):,}): {categorical_cols}")
    print(f"Numeric features ({len(numeric_cols):,}): {numeric_cols}")
    return build_feature_report(frame, feature_cols, categorical_cols, schema=schema)


def _load_watch_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data)
    raise ValueError(f"Unsupported watch file type: {path}")


def _drop_duplicate_nonmissing_keys(df: pd.DataFrame, key_col: str) -> pd.DataFrame:
    key = df[key_col]
    missing = key.isna()
    deduped = df[~missing].drop_duplicates(subset=[key_col], keep="first")
    if missing.any():
        deduped = pd.concat([deduped, df[missing]], ignore_index=False)
    return deduped.sort_index().copy()


def _to_numeric(values) -> pd.Series:
    if values is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(values, errors="coerce")


def _positive_numeric(values) -> pd.Series:
    numeric = _to_numeric(values)
    return numeric.where(numeric > 0)


def _first_number(values) -> pd.Series:
    if values is None:
        return pd.Series(dtype=float)
    series = pd.Series(values).astype("string")
    extracted = series.str.extract(r"(-?\d+(?:\.\d+)?)", expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def _year_numeric(values) -> pd.Series:
    numeric = _first_number(values)
    return numeric.where((numeric >= 1500) & (numeric <= 2035))


def _clean_category(values) -> pd.Series:
    if values is None:
        return pd.Series(dtype=object)
    series = pd.Series(values).astype("string").str.strip()
    series = series.replace(
        {
            "": pd.NA,
            "N/A": pd.NA,
            "n/a": pd.NA,
            "None": pd.NA,
            "none": pd.NA,
            "nan": pd.NA,
            "<NA>": pd.NA,
        }
    )
    return series.fillna("__missing__").astype(str)


def _clean_yes_no(values) -> pd.Series:
    if values is None:
        return pd.Series(dtype=object)
    series = pd.Series(values).astype("string").str.strip().str.lower()
    yes_values = {"1", "1.0", "true", "yes", "y", "present", "with", "included"}
    no_values = {"0", "0.0", "false", "no", "n", "absent", "without", "none"}
    missing_values = {"", "nan", "n/a", "na", "null", "<na>", "missing"}
    out = pd.Series("__other__", index=series.index, dtype=object)
    out[series.isin(yes_values)] = "yes"
    out[series.isin(no_values)] = "no"
    out[series.isna() | series.isin(missing_values)] = "__missing__"
    return out


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(f"[watch features] {message}")
