from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DATA_DIR = Path("../../Data/sampled_sets")
TARGET_COL = "ah_price_usd"
MARKET_COLS = ["GSPC", "FTSE", "IXIC", "DJI", "VIX", "BZ_F", "FVX"]

CATEGORICAL_FEATURES = [
    "auction_house",
    "location",
    "material_clean",
    "artwork_title_clean",
    "medium",
    "signed",
    "titled",
    "dated",
    "artist_name_grouped",
    "edition_clean",
]

NUMERIC_FEATURES = [
    "sale_date_ordinal",
    "sale_year_num",
    "sale_month_num",
    "year_of_creation_num",
    "length_cm",
    "height_cm",
    "width_cm",
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
    "auction_house": "Cleaned categorical from raw auction_house; rare levels grouped after schema fitting.",
    "location": "Cleaned categorical from raw location; rare levels grouped after schema fitting.",
    "material_clean": "Lexically cleaned raw material text; rare cleaned phrases grouped after schema fitting.",
    "artwork_title_clean": "Cleaned categorical from artwork_title; rare titles grouped after schema fitting.",
    "medium": "Cleaned categorical from raw medium; rare levels grouped after schema fitting.",
    "signed": "Cleaned categorical yes/no value from signed.",
    "titled": "Cleaned categorical yes/no value from titled.",
    "dated": "Cleaned categorical yes/no value from dated.",
    "artist_name_grouped": "Cleaned categorical from artist_name; rare artists grouped after schema fitting.",
    "edition_clean": "Cleaned categorical from edition; rare levels grouped after schema fitting.",
    "sale_date_ordinal": "Numeric ordinal day derived from sale_date.",
    "sale_year_num": "Numeric sale year derived from sale_date.",
    "sale_month_num": "Numeric sale month derived from sale_date.",
    "year_of_creation_num": "Numeric conversion of year_of_creation.",
    "length_cm": "Numeric length converted to centimeters using unit.",
    "height_cm": "Numeric height converted to centimeters using unit.",
    "width_cm": "Numeric width converted to centimeters using unit.",
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


def load_artwork_data(data_dir: str | Path = DATA_DIR, *, verbose: bool = False) -> pd.DataFrame:
    """Load sampled artwork CSV/JSON files."""

    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir.resolve()}")

    _log(verbose, f"Loading artwork files from {data_dir.resolve()}")
    paths = sorted([*data_dir.glob("*.csv"), *data_dir.glob("*.json")])
    frames = []
    for path in paths:
        frame = _load_artwork_file(path)
        frame["source_file"] = path.name
        frame["source_quarter"] = _quarter_from_filename(path.name)
        frame["is_quarter_test_file"] = frame["source_quarter"].notna()
        frames.append(frame)
        _log(verbose, f"  loaded {path.name}: {len(frame):,} rows, {len(frame.columns):,} columns")

    if not frames:
        raise FileNotFoundError(f"No CSV or JSON files found in {data_dir.resolve()}")

    df = pd.concat(frames, ignore_index=True)
    df["sale_date"] = pd.to_datetime(df["sale_date"], errors="coerce")
    _log(verbose, f"Loaded combined dataframe: {len(df):,} rows, {len(df.columns):,} columns")
    return df


def split_train_test(
    df: pd.DataFrame,
    *,
    train_rows: int | None = 10_000,
    test_rows_per_quarter: int | None = 250,
    dedupe_key_col: str = "lot_url",
    cap_oversized_requests: bool = True,
    random_state: int = 42,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split train/test samples using lot_url as the leakage/duplicate key."""

    train = df[~df["is_quarter_test_file"]].copy()
    test = df[df["is_quarter_test_file"]].copy()
    _log(verbose, f"Initial split: train={len(train):,}, explicit test={len(test):,}")

    if train_rows is not None:
        if len(train) < train_rows:
            if not cap_oversized_requests:
                raise ValueError(f"Requested {train_rows:,} train rows, only {len(train):,} available.")
            _log(
                verbose,
                f"Requested {train_rows:,} train rows, only {len(train):,} available; using all available train rows.",
            )
        elif len(train) > train_rows:
            train = train.sample(n=train_rows, random_state=random_state).copy()
        _log(verbose, f"Using train rows: {len(train):,}")

    if dedupe_key_col in train.columns:
        duplicate_train = int(train[dedupe_key_col].dropna().duplicated().sum())
        if duplicate_train:
            _log(
                verbose,
                f"Train contains {duplicate_train:,} duplicate {dedupe_key_col} rows; "
                "keeping them to preserve exact train size.",
            )

    if dedupe_key_col in test.columns:
        before = len(test)
        test = _drop_duplicate_nonmissing_keys(test, dedupe_key_col)
        _log(verbose, f"De-duplicated test {dedupe_key_col}: {before:,}->{len(test):,}")

        before = len(test)
        train_keys = set(train[dedupe_key_col].dropna())
        test = test[~test[dedupe_key_col].isin(train_keys)].copy()
        _log(verbose, f"Removed train/test overlapping {dedupe_key_col} values from test: {before:,}->{len(test):,}")

    if test_rows_per_quarter is not None:
        sampled = []
        for quarter, group in test.groupby("source_quarter", dropna=False):
            quarter_n = min(test_rows_per_quarter, len(group)) if cap_oversized_requests else test_rows_per_quarter
            if len(group) < test_rows_per_quarter and not cap_oversized_requests:
                raise ValueError(
                    f"Requested {test_rows_per_quarter:,} rows for {quarter}, only {len(group):,} available."
                )
            if len(group) < test_rows_per_quarter:
                _log(
                    verbose,
                    f"Requested {test_rows_per_quarter:,} test rows for {quarter}, "
                    f"only {len(group):,} available; using {quarter_n:,}.",
                )
            sampled.append(group.sample(n=quarter_n, random_state=random_state))
        test = pd.concat(sampled, ignore_index=True)
        _log(verbose, f"Requested test rows per quarter: {test_rows_per_quarter:,}")
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
    """Create the constrained Ridge feature frame."""

    frame = df.copy()
    _log(
        verbose,
        f"Building Ridge model frame from {len(frame):,} rows using "
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
    frame["year_of_creation_num"] = _to_numeric(frame.get("year_of_creation"))

    frame["length_cm"] = _dimension_to_cm(frame.get("length"), frame.get("unit"))
    frame["height_cm"] = _dimension_to_cm(frame.get("height"), frame.get("unit"))
    frame["width_cm"] = _dimension_to_cm(frame.get("width"), frame.get("unit"))

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
    frame["medium"] = _clean_category(frame.get("medium"))
    frame["signed"] = _clean_category(frame.get("signed"))
    frame["titled"] = _clean_category(frame.get("titled"))
    frame["dated"] = _clean_category(frame.get("dated"))
    frame["artist_name_grouped"] = _clean_category(frame.get("artist_name"))
    frame["edition_clean"] = _clean_category(frame.get("edition"))
    frame["artwork_title_clean"] = _clean_category(frame.get("artwork_title"))
    frame["material_clean"] = _clean_material_text(frame.get("material"))

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
    if include_estimates:
        _log(verbose, f"  included auction estimate numeric features: {ESTIMATE_NUMERIC_FEATURES}")
    else:
        _log(verbose, "  auction estimate fields retained for reporting but excluded from model features")
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
    _log(verbose, f"Fitting Ridge feature schema with min_category_count={min_category_count}")

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
    _log(verbose, f"Applying Ridge feature schema to {len(frame):,} rows and {len(feature_cols):,} features")

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


def _load_artwork_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            return pd.DataFrame(json.load(f))
    raise ValueError(f"Unsupported artwork file type: {path}")


def _drop_duplicate_nonmissing_keys(df: pd.DataFrame, key_col: str) -> pd.DataFrame:
    key = df[key_col]
    missing = key.isna()
    deduped = df[~missing].drop_duplicates(subset=[key_col], keep="first")
    if missing.any():
        deduped = pd.concat([deduped, df[missing]], ignore_index=False)
    return deduped.sort_index().copy()


def _quarter_from_filename(filename: str) -> str | None:
    match = re.search(r"q(20\d{2}Q[1-4])", filename)
    return match.group(1) if match else None


def _to_numeric(values) -> pd.Series:
    if values is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(values, errors="coerce")


def _positive_numeric(values) -> pd.Series:
    numeric = _to_numeric(values)
    return numeric.where(numeric > 0)


def _clean_category(values) -> pd.Series:
    if values is None:
        return pd.Series(dtype=object)
    series = pd.Series(values).astype("string").str.strip()
    series = series.replace({"": pd.NA, "N/A": pd.NA, "None": pd.NA, "nan": pd.NA, "<NA>": pd.NA})
    return series.fillna("__missing__").astype(str)


def _clean_material_text(values) -> pd.Series:
    if values is None:
        return pd.Series(dtype=object)
    series = pd.Series(values).astype("string").str.lower().fillna("")
    replacements = {
        r"&amp;": " and ",
        r"\bcolours\b": " colors ",
        r"\bcolour\b": " color ",
    }
    for pattern, replacement in replacements.items():
        series = series.str.replace(pattern, replacement, regex=True)
    series = series.str.replace(r"[^a-z0-9]+", " ", regex=True)
    series = series.str.replace(r"\b(on|in|with|and|the|a|an|of|mounted|laid|down)\b", " ", regex=True)
    series = series.str.replace(r"\s+", " ", regex=True).str.strip()
    series = series.replace({"": "__missing__", "n a": "__missing__", "none": "__missing__"})
    return series.astype(str)


def _dimension_to_cm(values, units) -> pd.Series:
    values = _to_numeric(values)
    units = pd.Series(units if units is not None else pd.Series(index=values.index, dtype=object))
    units = units.astype("string").str.lower().str.strip()
    out = values.copy().astype(float)
    out = np.where(units.eq("in").fillna(False), out * 2.54, out)
    out = np.where(units.eq("mm").fillna(False), out / 10.0, out)
    out = pd.Series(out, index=values.index)
    return out.where(out > 0, np.nan)


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(f"[ridge features] {message}")
