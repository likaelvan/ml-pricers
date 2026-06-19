from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DATA_DIR = Path("../Data/sampled_sets")
TARGET_COL = "ah_price_usd"
PREMIUM_TARGET_COL = "ah_bp_price_usd"

MARKET_COLS = ["GSPC", "FTSE", "IXIC", "DJI", "VIX", "BZ_F", "FVX"]

BASE_CATEGORICAL_FEATURES = [
    "auction_house",
    "location",
    "material_clean",
    "material_cluster",
    "artwork_title_clean",
    "medium",
    "signed",
    "titled",
    "dated",
    "artist_name_grouped",
    "edition_clean",
]

BASE_NUMERIC_FEATURES = [
    "sale_date_ordinal",
    "sale_year_num",
    "sale_month_num",
    "year_of_creation_num",
    "length_cm",
    "height_cm",
    "width_cm",
    *MARKET_COLS,
]

AUCTION_ESTIMATE_FEATURES = [
    "ah_low_estimate_usd",
    "ah_high_estimate_usd",
    "auction_midpoint_usd",
    "log_auction_midpoint_usd",
    "estimate_range_ratio",
]

FEATURE_TRANSFORMATIONS = {
    "auction_house": "Cleaned categorical from raw auction_house; rare levels grouped after schema fitting.",
    "location": "Cleaned categorical from raw location; rare levels grouped after schema fitting.",
    "medium": "Cleaned categorical from raw medium; rare levels grouped after schema fitting.",
    "material_clean": "Lexically cleaned raw material text; frequent cleaned phrases retained, rare phrases grouped.",
    "material_cluster": "Train-fitted TF-IDF char n-gram + MiniBatchKMeans cluster over material_clean.",
    "artwork_title_clean": "Cleaned categorical from artwork_title; rare titles grouped after schema fitting.",
    "artist_name_grouped": "Cleaned categorical from artist_name; rare artists grouped after schema fitting.",
    "edition_clean": "Cleaned categorical from edition; rare levels grouped after schema fitting.",
    "signed": "Cleaned categorical yes/no value from signed.",
    "titled": "Cleaned categorical yes/no value from titled.",
    "dated": "Cleaned categorical yes/no value from dated.",
    "sale_date_ordinal": "Numeric ordinal day derived from sale_date.",
    "sale_year_num": "Numeric sale year derived from sale_date.",
    "sale_month_num": "Numeric sale month derived from sale_date.",
    "year_of_creation_num": "Numeric conversion of year_of_creation.",
    "height_cm": "Numeric height converted to centimeters using unit.",
    "width_cm": "Numeric width converted to centimeters using unit.",
    "length_cm": "Numeric length converted to centimeters using unit.",
    "GSPC": "Numeric market index level.",
    "FTSE": "Numeric market index level.",
    "IXIC": "Numeric market index level.",
    "DJI": "Numeric market index level.",
    "VIX": "Numeric market/risk index level.",
    "BZ_F": "Numeric Brent crude proxy.",
    "FVX": "Numeric 5-year treasury yield proxy.",
    "ah_low_estimate_usd": "Auction-house low estimate in USD; optional estimate-aware feature.",
    "ah_high_estimate_usd": "Auction-house high estimate in USD; optional estimate-aware feature.",
    "auction_midpoint_usd": "Mean of auction low/high estimate USD; optional estimate-aware feature.",
    "log_auction_midpoint_usd": "log1p(auction_midpoint_usd); optional estimate-aware feature.",
    "estimate_range_ratio": "(high - low) / midpoint; optional estimate-aware feature.",
}


@dataclass(frozen=True)
class FeatureSchema:
    """Train-fitted categorical levels used to apply stable categories to holdouts."""

    categorical_levels: dict[str, list[str]]
    numeric_medians: dict[str, float]
    material_vectorizer: object | None = None
    material_cluster_model: object | None = None
    material_cluster_feature: str = "material_cluster"


def load_artwork_data(data_dir: str | Path = DATA_DIR, *, verbose: bool = False) -> pd.DataFrame:
    """Load artwork CSV/JSON files from the configured train/test data directory."""

    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir.resolve()}")

    _log(verbose, f"Loading artwork files from {data_dir.resolve()}")
    frames = []
    paths = sorted([*data_dir.glob("*.csv"), *data_dir.glob("*.json")])
    for path in paths:
        frame = _load_artwork_file(path)
        frame["source_file"] = path.name
        frame["source_quarter"] = _quarter_from_filename(path.name)
        frame["is_quarter_holdout_file"] = frame["source_quarter"].notna()
        frames.append(frame)
        _log(verbose, f"  loaded {path.name}: {len(frame):,} rows, {len(frame.columns):,} columns")

    if not frames:
        raise FileNotFoundError(f"No CSV or JSON files found in {data_dir.resolve()}")

    df = pd.concat(frames, ignore_index=True)
    df["sale_date"] = pd.to_datetime(df["sale_date"], errors="coerce")
    df["sale_quarter"] = df["sale_date"].dt.to_period("Q").astype(str)
    df["sale_month"] = df["sale_date"].dt.month.astype("Int64").astype(str)
    _log(verbose, f"Loaded combined dataframe: {len(df):,} rows, {len(df.columns):,} columns")
    return df


def split_train_test(
    df: pd.DataFrame,
    *,
    train_rows: int | None = 10_000,
    test_rows_per_quarter: int | None = 250,
    dedupe_key_col: str = "lot_url",
    dedupe_test_ids: bool = True,
    remove_train_ids_from_test: bool = True,
    cap_oversized_requests: bool = True,
    random_state: int = 42,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split sampled train rows from explicit quarterly test files.

    The sampled data folder contains one 2023-2024 train CSV and explicit
    2025 quarterly test CSVs. By default this keeps exactly 10,000 train rows
    and samples exactly 250 rows per quarter from de-duplicated test candidates.
    Duplicate/leakage checks use lot_url by default because it is a better lot
    identity key than scraper-generated ID.
    """

    df = df.copy()
    train = df[~df["is_quarter_holdout_file"]].copy()
    test = df[df["is_quarter_holdout_file"]].copy()
    _log(verbose, f"Initial split: train={len(train):,}, explicit test={len(test):,}")

    if train_rows is not None:
        if len(train) < train_rows:
            if not cap_oversized_requests:
                raise ValueError(f"Requested {train_rows:,} train rows, but only {len(train):,} are available.")
            _log(
                verbose,
                f"Requested {train_rows:,} train rows, only {len(train):,} available; using all available train rows.",
            )
        elif len(train) > train_rows:
            train = train.sample(n=train_rows, random_state=random_state).copy()
        _log(verbose, f"Using train rows: {len(train):,}")

    if dedupe_key_col in train.columns:
        duplicate_train_keys = int(train[dedupe_key_col].dropna().duplicated().sum())
        if duplicate_train_keys:
            _log(
                verbose,
                f"Train contains {duplicate_train_keys:,} duplicate {dedupe_key_col} rows; "
                "keeping them to preserve exact train size.",
            )

    if dedupe_test_ids and dedupe_key_col in test.columns:
        test_before = len(test)
        test = _drop_duplicate_nonmissing_keys(test, dedupe_key_col)
        _log(verbose, f"De-duplicated test {dedupe_key_col}: {test_before:,}->{len(test):,}")

    if remove_train_ids_from_test and dedupe_key_col in df.columns:
        test_before = len(test)
        train_keys = set(train[dedupe_key_col].dropna())
        test = test[~test[dedupe_key_col].isin(train_keys)].copy()
        _log(
            verbose,
            f"Removed train/test overlapping {dedupe_key_col} values from test: {test_before:,}->{len(test):,}",
        )

    if test_rows_per_quarter is not None:
        sampled = []
        for quarter, group in test.groupby("source_quarter", dropna=False):
            quarter_n = min(test_rows_per_quarter, len(group)) if cap_oversized_requests else test_rows_per_quarter
            if len(group) < test_rows_per_quarter and not cap_oversized_requests:
                raise ValueError(
                    f"Requested {test_rows_per_quarter:,} test rows for {quarter}, "
                    f"but only {len(group):,} are available after filtering."
                )
            if len(group) < test_rows_per_quarter:
                _log(
                    verbose,
                    f"Requested {test_rows_per_quarter:,} test rows for {quarter}, "
                    f"only {len(group):,} are available after filtering; using {quarter_n:,}.",
                )
            sampled.append(group.sample(n=quarter_n, random_state=random_state))
        test = pd.concat(sampled, ignore_index=True)
        _log(verbose, f"Requested test rows per quarter: {test_rows_per_quarter:,}")
        _log(verbose, f"Actual test rows per quarter: {test['source_quarter'].value_counts().sort_index().to_dict()}")

    _log(verbose, f"Final split: train={len(train):,}, test={len(test):,}")
    return train.reset_index(drop=True), test.reset_index(drop=True)


def split_train_holdout(*args, **kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backward-compatible alias for split_train_test."""

    if "dedupe_id_col" in kwargs and "dedupe_key_col" not in kwargs:
        kwargs["dedupe_key_col"] = kwargs.pop("dedupe_id_col")
    return split_train_test(*args, **kwargs)


def make_model_frame(
    df: pd.DataFrame,
    *,
    target_col: str = TARGET_COL,
    include_auction_estimates: bool = False,
    verbose: bool = False,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Create model-ready engineered features and return feature column lists."""

    frame = df.copy()
    _log(verbose, f"Building model frame from {len(frame):,} rows using target_col={target_col!r}")

    for col in [
        "artist_birth",
        "artist_death",
        "year_of_creation",
        "sale_year",
        "height",
        "width",
        "length",
        "ah_low_estimate_usd",
        "ah_high_estimate_usd",
        "ah_price_usd",
        "ah_bp_price_usd",
        "artwork_count",
        "item_count",
        *MARKET_COLS,
    ]:
        if col in frame.columns:
            frame[f"{col}_raw"] = frame[col]

    before_target_filter = len(frame)
    frame["target_price_usd"] = _to_numeric(frame.get(target_col))
    frame = frame[np.isfinite(frame["target_price_usd"]) & (frame["target_price_usd"] > 0)].copy()
    frame["log_target_price_usd"] = np.log1p(frame["target_price_usd"])
    _log(verbose, f"Filtered invalid/non-positive targets: {before_target_filter:,}->{len(frame):,}")

    frame["year_of_creation_num"] = _to_numeric(frame.get("year_of_creation"))
    frame["sale_date_ordinal"] = frame["sale_date"].map(lambda value: value.toordinal() if pd.notna(value) else np.nan)
    frame["sale_year_num"] = frame["sale_date"].dt.year.astype("float")
    frame["sale_month_num"] = frame["sale_date"].dt.month.astype("float")

    frame["height_cm"] = _dimension_to_cm(frame.get("height"), frame.get("unit"))
    frame["width_cm"] = _dimension_to_cm(frame.get("width"), frame.get("unit"))
    frame["length_cm"] = _dimension_to_cm(frame.get("length"), frame.get("unit"))

    frame["artist_name_grouped"] = _clean_category(frame.get("artist_name"))
    frame["auction_house"] = _clean_category(frame.get("auction_house"))
    frame["location"] = _clean_category(frame.get("location"))
    frame["medium"] = _clean_category(frame.get("medium"))
    frame["signed"] = _clean_category(frame.get("signed"))
    frame["titled"] = _clean_category(frame.get("titled"))
    frame["dated"] = _clean_category(frame.get("dated"))
    frame["edition_clean"] = _clean_category(frame.get("edition"))
    frame["artwork_title_clean"] = _clean_category(frame.get("artwork_title"))
    frame["material_clean"] = _clean_material_text(frame.get("material"))

    frame["ah_low_estimate_usd"] = _to_numeric(frame.get("ah_low_estimate_usd"))
    frame["ah_high_estimate_usd"] = _to_numeric(frame.get("ah_high_estimate_usd"))
    frame["auction_midpoint_usd"] = (frame["ah_low_estimate_usd"] + frame["ah_high_estimate_usd"]) / 2
    frame["log_auction_midpoint_usd"] = np.log1p(frame["auction_midpoint_usd"])
    frame["estimate_range_ratio"] = (
        (frame["ah_high_estimate_usd"] - frame["ah_low_estimate_usd"]) / frame["auction_midpoint_usd"]
    )

    for col in MARKET_COLS:
        if col not in frame.columns:
            frame[col] = np.nan
        frame[col] = _to_numeric(frame[col])

    categorical_cols = list(BASE_CATEGORICAL_FEATURES)
    numeric_cols = list(BASE_NUMERIC_FEATURES)
    if include_auction_estimates:
        numeric_cols += AUCTION_ESTIMATE_FEATURES

    feature_cols = categorical_cols + numeric_cols
    _log(
        verbose,
        f"Engineered {len(feature_cols):,} features "
        f"({len(categorical_cols):,} categorical, {len(numeric_cols):,} numeric). "
        f"include_auction_estimates={include_auction_estimates}",
    )
    if include_auction_estimates:
        _log(verbose, f"  included auction estimate numeric features: {AUCTION_ESTIMATE_FEATURES}")
    else:
        _log(verbose, "  auction estimate fields retained for reporting but excluded from model features")
    keep_cols = [
        "ID",
        "lot_url",
        "source_file",
        "source_quarter",
        "is_quarter_holdout_file",
        "sale_date",
        "target_price_usd",
        "log_target_price_usd",
        "auction_midpoint_usd",
        *AUCTION_ESTIMATE_FEATURES,
        *feature_cols,
    ]
    keep_cols = list(dict.fromkeys(col for col in keep_cols if col in frame.columns))
    return frame[keep_cols].copy(), feature_cols, categorical_cols


def fit_feature_schema(
    frame: pd.DataFrame,
    feature_cols: Iterable[str],
    categorical_cols: Iterable[str],
    *,
    min_category_count: int = 20,
    n_material_clusters: int = 32,
    verbose: bool = False,
) -> FeatureSchema:
    """Fit category levels and numeric medians from training data only."""

    categorical_cols = list(categorical_cols)
    feature_cols = list(feature_cols)
    _log(
        verbose,
        f"Fitting feature schema with min_category_count={min_category_count}, "
        f"n_material_clusters={n_material_clusters}",
    )
    material_vectorizer, material_cluster_model = _fit_material_clusterer(
        frame.get("material_clean"),
        n_clusters=n_material_clusters,
    )
    if material_cluster_model is not None:
        _log(verbose, f"  fitted material clusters: {material_cluster_model.n_clusters}")
    else:
        _log(verbose, "  material clustering skipped: insufficient usable material text")

    categorical_levels: dict[str, list[str]] = {}
    for col in categorical_cols:
        if col == "material_cluster":
            if material_cluster_model is None:
                levels = ["__missing__", "__other__"]
            else:
                levels = [f"material_cluster_{idx:02d}" for idx in range(material_cluster_model.n_clusters)]
                levels += ["__missing__", "__other__"]
            categorical_levels[col] = levels
            _log(verbose, f"  categorical {col}: {len(levels):,} levels")
            continue

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

    return FeatureSchema(
        categorical_levels=categorical_levels,
        numeric_medians=numeric_medians,
        material_vectorizer=material_vectorizer,
        material_cluster_model=material_cluster_model,
    )


def apply_feature_schema(
    frame: pd.DataFrame,
    feature_cols: Iterable[str],
    categorical_cols: Iterable[str],
    schema: FeatureSchema,
    *,
    verbose: bool = False,
) -> pd.DataFrame:
    """Apply train-fitted schema to produce stable LightGBM feature matrices."""

    feature_cols = list(feature_cols)
    categorical_cols = list(categorical_cols)
    out = pd.DataFrame(index=frame.index)
    _log(verbose, f"Applying feature schema to {len(frame):,} rows and {len(feature_cols):,} features")

    for col in feature_cols:
        if col in categorical_cols:
            levels = schema.categorical_levels[col]
            if col == schema.material_cluster_feature:
                values = _predict_material_cluster(frame, schema)
            else:
                values = _clean_category(frame[col] if col in frame.columns else pd.Series(index=frame.index, dtype=object))
            values = values.where(values.isin(levels), "__other__")
            values = values.fillna("__missing__")
            out[col] = pd.Categorical(values, categories=levels)
        else:
            values = _to_numeric(frame[col] if col in frame.columns else pd.Series(index=frame.index, dtype=float))
            values = values.replace([np.inf, -np.inf], np.nan).fillna(schema.numeric_medians[col])
            out[col] = values.astype(float)

    return out


def build_feature_report(
    frame: pd.DataFrame,
    feature_cols: Iterable[str],
    categorical_cols: Iterable[str],
    *,
    schema: FeatureSchema | None = None,
) -> pd.DataFrame:
    """Return a compact audit table for the features used in training."""

    categorical_cols = set(categorical_cols)
    rows = []
    for feature in feature_cols:
        role = "categorical" if feature in categorical_cols else "numeric"
        if feature == "material_cluster" and schema is not None:
            series = _predict_material_cluster(frame, schema)
        else:
            series = frame[feature] if feature in frame.columns else pd.Series(index=frame.index, dtype=object)
        missing_rate = float(series.isna().mean()) if len(series) else 0.0
        if role == "categorical":
            cleaned = _clean_category(series)
            unique_count = int(cleaned.nunique(dropna=False))
            kept_levels = len(schema.categorical_levels.get(feature, [])) if schema else np.nan
            example = ", ".join(cleaned.drop_duplicates().head(3).astype(str).tolist())
        else:
            numeric = _to_numeric(series).replace([np.inf, -np.inf], np.nan)
            unique_count = int(numeric.nunique(dropna=True))
            kept_levels = np.nan
            example = ", ".join(numeric.dropna().head(3).round(4).astype(str).tolist())

        rows.append(
            {
                "feature": feature,
                "role": role,
                "transformation": FEATURE_TRANSFORMATIONS.get(feature, "No transformation note defined."),
                "missing_rate": missing_rate,
                "unique_count_before_grouping": unique_count,
                "train_schema_levels": kept_levels,
                "example_values": example,
            }
        )

    return pd.DataFrame(rows)


def print_feature_summary(
    frame: pd.DataFrame,
    feature_cols: Iterable[str],
    categorical_cols: Iterable[str],
    *,
    schema: FeatureSchema | None = None,
) -> pd.DataFrame:
    """Print training feature lists and return the detailed feature report."""

    feature_cols = list(feature_cols)
    categorical_cols = list(categorical_cols)
    numeric_cols = [col for col in feature_cols if col not in set(categorical_cols)]
    print(f"Training rows: {len(frame):,}")
    print(f"Training features: {len(feature_cols):,}")
    print(f"Categorical features ({len(categorical_cols):,}): {categorical_cols}")
    print(f"Numeric features ({len(numeric_cols):,}): {numeric_cols}")
    report = build_feature_report(frame, feature_cols, categorical_cols, schema=schema)
    return report.sort_values(["role", "feature"]).reset_index(drop=True)


def price_bucket(values: pd.Series) -> pd.Categorical:
    """Bucket actual prices for bias analysis."""

    bins = [0, 1_000, 5_000, 25_000, 100_000, 500_000, math.inf]
    labels = ["<$1k", "$1k-$5k", "$5k-$25k", "$25k-$100k", "$100k-$500k", "$500k+"]
    return pd.cut(values, bins=bins, labels=labels, include_lowest=True)


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(f"[features] {message}")


def _load_artwork_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            records = json.load(f)
        return pd.DataFrame(records)
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


def _clean_category(values) -> pd.Series:
    if values is None:
        return pd.Series(dtype=object)
    series = pd.Series(values).astype("string")
    series = series.str.strip()
    series = series.replace({"": pd.NA, "N/A": pd.NA, "None": pd.NA, "nan": pd.NA, "<NA>": pd.NA})
    return series.fillna("__missing__").astype(str)


def _clean_material_text(values) -> pd.Series:
    """Normalize material strings while preserving detail for modeling."""

    if values is None:
        return pd.Series(dtype=object)
    series = pd.Series(values).astype("string").str.lower()
    series = series.fillna("")
    replacements = {
        r"&amp;": " and ",
        r"\bcolours\b": " colors ",
        r"\bcolour\b": " color ",
        r"\bwove\b": " wove ",
        r"\blinen\b": " linen ",
        r"\bpanel\b": " panel ",
        r"\bpaper\b": " paper ",
        r"\bcanvas\b": " canvas ",
    }
    for pattern, replacement in replacements.items():
        series = series.str.replace(pattern, replacement, regex=True)
    series = series.str.replace(r"[^a-z0-9]+", " ", regex=True)
    series = series.str.replace(r"\b(on|in|with|and|the|a|an|of|mounted|laid|down)\b", " ", regex=True)
    series = series.str.replace(r"\s+", " ", regex=True).str.strip()
    series = series.replace({"": "__missing__", "n a": "__missing__", "none": "__missing__"})
    return series.astype(str)


def _fit_material_clusterer(values, *, n_clusters: int):
    material = _clean_material_text(values)
    material = material[material.ne("__missing__")]
    unique_count = material.nunique()
    if unique_count < 2 or n_clusters < 2:
        return None, None

    from sklearn.cluster import MiniBatchKMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    cluster_count = min(n_clusters, unique_count)
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=2000,
    )
    matrix = vectorizer.fit_transform(material)
    if matrix.shape[1] == 0:
        return None, None

    cluster_model = MiniBatchKMeans(
        n_clusters=cluster_count,
        random_state=42,
        batch_size=2048,
        n_init="auto",
    )
    cluster_model.fit(matrix)
    return vectorizer, cluster_model


def _predict_material_cluster(frame: pd.DataFrame, schema: FeatureSchema) -> pd.Series:
    if schema.material_vectorizer is None or schema.material_cluster_model is None:
        return pd.Series("__missing__", index=frame.index, dtype=object)

    if "material_clean" in frame.columns:
        material = _clean_material_text(frame["material_clean"])
    else:
        material = _clean_material_text(frame.get("material"))

    out = pd.Series("__missing__", index=frame.index, dtype=object)
    valid = material.ne("__missing__")
    if valid.any():
        matrix = schema.material_vectorizer.transform(material[valid])
        labels = schema.material_cluster_model.predict(matrix)
        out.loc[valid] = [f"material_cluster_{idx:02d}" for idx in labels]
    return out


def _dimension_to_cm(values, units) -> pd.Series:
    values = _to_numeric(values)
    units = pd.Series(units if units is not None else pd.Series(index=values.index, dtype=object)).astype("string").str.lower().str.strip()
    out = values.copy().astype(float)
    is_inches = units.eq("in").fillna(False)
    is_mm = units.eq("mm").fillna(False)
    out = np.where(is_inches, out * 2.54, out)
    out = np.where(is_mm, out / 10.0, out)
    out = pd.Series(out, index=values.index)
    out = out.where(out > 0, np.nan)
    return out


def _yes_no_flag(values) -> pd.Series:
    if values is None:
        return pd.Series(dtype=float)
    series = pd.Series(values).astype("string").str.lower().str.strip()
    return series.isin(["yes", "true", "1", "y"]).astype(float)


def _has_text(values) -> pd.Series:
    if values is None:
        return pd.Series(dtype=bool)
    series = pd.Series(values).astype("string").str.strip()
    return ~(series.isna() | series.eq("") | series.str.lower().isin(["none", "n/a", "nan", "<na>"]))


def _text_length(values) -> pd.Series:
    if values is None:
        return pd.Series(dtype=float)
    series = pd.Series(values).astype("string")
    valid = _has_text(series)
    return series.str.len().where(valid, 0).astype(float)


def _term_count(values) -> pd.Series:
    if values is None:
        return pd.Series(dtype=float)
    series = pd.Series(values).astype("string")
    valid = _has_text(series)
    return series.str.split(",").str.len().where(valid, 0).astype(float)


def _material_group(values) -> pd.Series:
    series = _clean_category(values).str.lower()
    groups = pd.Series("__other__", index=series.index, dtype=object)
    patterns = {
        "oil": r"\boil\b",
        "acrylic": r"\bacrylic\b",
        "watercolor": r"watercolou?r|gouache",
        "ink": r"\bink\b",
        "bronze": r"\bbronze\b",
        "lithograph": r"lithograph",
        "screenprint": r"screen ?print|silkscreen",
        "paper": r"\bpaper\b|wove",
        "canvas": r"\bcanvas\b",
        "wood": r"\bwood\b|panel",
        "photograph": r"photo|gelatin|chromogenic",
        "mixed_media": r"mixed media|collage",
    }
    for label, pattern in patterns.items():
        match = series.str.contains(pattern, regex=True, na=False) & groups.eq("__other__")
        groups = groups.where(~match, label)
    groups = groups.where(series.ne("__missing__"), "__missing__")
    return groups


def _auction_title_type(values) -> pd.Series:
    series = _clean_category(values).str.lower()
    out = pd.Series("__other__", index=series.index, dtype=object)
    patterns = {
        "evening_sale": r"evening",
        "day_sale": r"\bday\b",
        "prints": r"print|multiple",
        "modern": r"modern|impressionist",
        "contemporary": r"contemporary|post-war|21st",
        "design": r"design",
        "online": r"online",
    }
    for label, pattern in patterns.items():
        out = out.where(~series.str.contains(pattern, regex=True, na=False), label)
    out = out.where(series.ne("__missing__"), "__missing__")
    return out
