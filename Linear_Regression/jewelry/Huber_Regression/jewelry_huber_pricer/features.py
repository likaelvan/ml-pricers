from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
JEWELRY_DATA_DIR = ROOT / "Data" / "sampled_sets_jewelry"
TRAIN_FILE = JEWELRY_DATA_DIR / "Jewelry_2025_high_midpoint_error_remainder_training.csv"
TARGET_COL = "ah_price_usd"
MARKET_COLS = ["GSPC", "FTSE", "IXIC", "DJI", "VIX", "BZ_F", "FVX"]

CATEGORICAL_FEATURES = [
    "auction_house",
    "location",
    "product_name_clean",
    "brand_clean",
    "collection_clean",
    "model_clean",
    "reference_clean",
    "maker_signatures_clean",
    "metal_type_clean",
    "metal_purity_clean",
    "metal_color_clean",
    "jewelry_category_clean",
    "pearl_type_clean",
    "pearl_color_clean",
    "pearl_luster_clean",
    "pearl_surface_quality_clean",
    "gemstone_type_clean",
    "gemstone_shape_clean",
    "gemstone_color_clean",
    "gemstone_clarity_clean",
    "gemstone_origin_clean",
    "gemstone_treatment_clean",
    "diamond_color_clean",
    "diamond_clarity_clean",
    "diamond_cut_clean",
    "design_style_clean",
    "enamel_color_clean",
    "certification_lab_clean",
    "country_of_origin_clean",
]

NUMERIC_FEATURES = [
    "sale_date_ordinal",
    "sale_year_num",
    "sale_month_num",
    "total_item_weight_num",
    "ring_size_num",
    "necklace_length_num",
    "bracelet_length_num",
    "earring_length_num",
    "pendant_length_num",
    "chain_length_num",
    "gemstone_count_num",
    "diamond_count_num",
    "pearl_count_num",
    "item_count_num",
    "primary_gemstone_carat_num",
    "diamond_total_carat_num",
    "diamond_count_from_spec_num",
    "pearl_diameter_num",
    "pearl_count_from_spec_num",
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
    "product_name_clean": "Cleaned categorical from product_name; rare product names grouped.",
    "brand_clean": "Flattened from brand_and_model_identifiers.brand and cleaned.",
    "collection_clean": "Flattened from brand_and_model_identifiers.collection and cleaned.",
    "model_clean": "Flattened from brand_and_model_identifiers.model and cleaned.",
    "reference_clean": "Flattened from brand_and_model_identifiers.reference and cleaned.",
    "maker_signatures_clean": "Cleaned categorical from maker_signatures_and_hallmarks.",
    "metal_type_clean": "Flattened from primary_metal_specification.metal_type and cleaned.",
    "metal_purity_clean": "Flattened from primary_metal_specification.purity and cleaned.",
    "metal_color_clean": "Flattened from primary_metal_specification.color and cleaned.",
    "jewelry_category_clean": "Cleaned categorical from jewelry_category_and_subtypes.",
    "pearl_type_clean": "Flattened from pearl_specification.type and cleaned.",
    "pearl_color_clean": "Flattened from pearl_specification.color and cleaned.",
    "pearl_luster_clean": "Flattened from pearl_specification.luster and cleaned.",
    "pearl_surface_quality_clean": "Flattened from pearl_specification.surface_quality and cleaned.",
    "gemstone_type_clean": "Flattened from primary_gemstone_specification.type and cleaned.",
    "gemstone_shape_clean": "Flattened from primary_gemstone_specification.shape and cleaned.",
    "gemstone_color_clean": "Flattened from primary_gemstone_specification.color and cleaned.",
    "gemstone_clarity_clean": "Flattened from primary_gemstone_specification.clarity and cleaned.",
    "gemstone_origin_clean": "Flattened from primary_gemstone_specification.origin and cleaned when available.",
    "gemstone_treatment_clean": "Flattened from primary_gemstone_specification.treatment and cleaned when available.",
    "diamond_color_clean": "Flattened from diamond_set_specification.color and cleaned.",
    "diamond_clarity_clean": "Flattened from diamond_set_specification.clarity and cleaned.",
    "diamond_cut_clean": "Flattened from diamond_set_specification.cut and cleaned.",
    "design_style_clean": "Cleaned categorical from design_style_and_motif.",
    "enamel_color_clean": "Cleaned categorical from enamel_color.",
    "certification_lab_clean": "Flattened from inclusions_and_certification.lab and cleaned.",
    "country_of_origin_clean": "Cleaned categorical from country_of_origin.",
    "sale_date_ordinal": "Numeric ordinal day derived from sale_date.",
    "sale_year_num": "Numeric sale year derived from sale_date.",
    "sale_month_num": "Numeric sale month derived from sale_date.",
    "total_item_weight_num": "First numeric token extracted from total_item_weight_measurement.",
    "ring_size_num": "First numeric token extracted from ring_size_measurement.",
    "necklace_length_num": "First numeric token extracted from necklace_length_measurement.",
    "bracelet_length_num": "First numeric token extracted from bracelet_length_measurement.",
    "earring_length_num": "First numeric token extracted from earring_length_measurement.",
    "pendant_length_num": "First numeric token extracted from pendant_length_measurement.",
    "chain_length_num": "First numeric token extracted from chain_length_measurement.",
    "gemstone_count_num": "Numeric gemstone_count.",
    "diamond_count_num": "Numeric diamond_count.",
    "pearl_count_num": "Numeric pearl_count.",
    "item_count_num": "Numeric item_count.",
    "primary_gemstone_carat_num": "Mean numeric value from primary_gemstone_specification.carat.",
    "diamond_total_carat_num": "Mean numeric value from diamond_set_specification.total_carat.",
    "diamond_count_from_spec_num": "Mean numeric value from diamond_set_specification.count.",
    "pearl_diameter_num": "Mean numeric value from pearl_specification.diameter.",
    "pearl_count_from_spec_num": "Mean numeric value from pearl_specification.count.",
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


def load_jewelry_data(
    train_file: str | Path = TRAIN_FILE,
    test_dir: str | Path = JEWELRY_DATA_DIR,
    *,
    train_before: str | None = None,
    quarters: Iterable[str] = ("2025Q1", "2025Q2", "2025Q3", "2025Q4"),
    verbose: bool = False,
) -> pd.DataFrame:
    """Load jewelry training data and explicit 2025 quarterly holdout files."""

    train_file = Path(train_file)
    test_dir = Path(test_dir)
    if not train_file.exists():
        raise FileNotFoundError(f"Training file not found: {train_file.resolve()}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir.resolve()}")

    train = _load_jewelry_file(train_file)
    train["source_file"] = train_file.name
    train["source_quarter"] = None
    train["is_quarter_test_file"] = False
    train["sale_date"] = pd.to_datetime(train["sale_date"], errors="coerce")
    if train_before:
        before_filter = len(train)
        train = train[train["sale_date"].lt(pd.Timestamp(train_before))].copy()
        _log(verbose, f"Filtered train sale_date < {train_before}: {before_filter:,}->{len(train):,}")

    frames = [train]
    for quarter in quarters:
        path = test_dir / f"Jewelry_2025_high_midpoint_error_q{quarter}_sample1000.csv"
        if not path.exists():
            raise FileNotFoundError(f"Quarter test file not found: {path.resolve()}")
        frame = _load_jewelry_file(path)
        frame["source_file"] = path.name
        frame["source_quarter"] = quarter
        frame["is_quarter_test_file"] = True
        frame["sale_date"] = pd.to_datetime(frame["sale_date"], errors="coerce")
        frames.append(frame)
        _log(verbose, f"Loaded {path.name}: {len(frame):,} rows, {len(frame.columns):,} columns")

    df = pd.concat(frames, ignore_index=True, sort=False)
    _log(verbose, f"Loaded jewelry frame: {len(df):,} rows, {len(df.columns):,} columns")
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
    """Split train/test samples and remove any test-overlapping lot_url from train."""

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
    """Create the constrained jewelry regression feature frame."""

    frame = _add_flattened_jewelry_columns(df.copy())
    _log(
        verbose,
        f"Building jewelry model frame from {len(frame):,} rows using "
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

    frame["total_item_weight_num"] = _mean_number(frame.get("total_item_weight_measurement"))
    frame["ring_size_num"] = _mean_number(frame.get("ring_size_measurement"))
    frame["necklace_length_num"] = _mean_number(frame.get("necklace_length_measurement"))
    frame["bracelet_length_num"] = _mean_number(frame.get("bracelet_length_measurement"))
    frame["earring_length_num"] = _mean_number(frame.get("earring_length_measurement"))
    frame["pendant_length_num"] = _mean_number(frame.get("pendant_length_measurement"))
    frame["chain_length_num"] = _mean_number(frame.get("chain_length_measurement"))
    frame["gemstone_count_num"] = _mean_number(frame.get("gemstone_count"))
    frame["diamond_count_num"] = _mean_number(frame.get("diamond_count"))
    frame["pearl_count_num"] = _mean_number(frame.get("pearl_count"))
    frame["item_count_num"] = _mean_number(frame.get("item_count"))
    frame["primary_gemstone_carat_num"] = _mean_number(frame.get("gemstone_carat_raw"))
    frame["diamond_total_carat_num"] = _mean_number(frame.get("diamond_total_carat_raw"))
    frame["diamond_count_from_spec_num"] = _mean_number(frame.get("diamond_count_raw"))
    frame["pearl_diameter_num"] = _mean_number(frame.get("pearl_diameter_raw"))
    frame["pearl_count_from_spec_num"] = _mean_number(frame.get("pearl_count_raw"))

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
    frame["product_name_clean"] = _clean_category(frame.get("product_name"))
    frame["brand_clean"] = _clean_category(frame.get("brand_raw"))
    frame["collection_clean"] = _clean_category(frame.get("collection_raw"))
    frame["model_clean"] = _clean_category(frame.get("model_raw"))
    frame["reference_clean"] = _clean_category(frame.get("reference_raw"))
    frame["maker_signatures_clean"] = _clean_category(frame.get("maker_signatures_and_hallmarks"))
    frame["metal_type_clean"] = _clean_category(frame.get("metal_type_raw"))
    frame["metal_purity_clean"] = _clean_category(frame.get("metal_purity_raw"))
    frame["metal_color_clean"] = _clean_category(frame.get("metal_color_raw"))
    frame["jewelry_category_clean"] = _clean_category(frame.get("jewelry_category_and_subtypes"))
    frame["pearl_type_clean"] = _clean_category(frame.get("pearl_type_raw"))
    frame["pearl_color_clean"] = _clean_category(frame.get("pearl_color_raw"))
    frame["pearl_luster_clean"] = _clean_category(frame.get("pearl_luster_raw"))
    frame["pearl_surface_quality_clean"] = _clean_category(frame.get("pearl_surface_quality_raw"))
    frame["gemstone_type_clean"] = _clean_category(frame.get("gemstone_type_raw"))
    frame["gemstone_shape_clean"] = _clean_category(frame.get("gemstone_shape_raw"))
    frame["gemstone_color_clean"] = _clean_category(frame.get("gemstone_color_raw"))
    frame["gemstone_clarity_clean"] = _clean_category(frame.get("gemstone_clarity_raw"))
    frame["gemstone_origin_clean"] = _clean_category(frame.get("gemstone_origin_raw"))
    frame["gemstone_treatment_clean"] = _clean_category(frame.get("gemstone_treatment_raw"))
    frame["diamond_color_clean"] = _clean_category(frame.get("diamond_color_raw"))
    frame["diamond_clarity_clean"] = _clean_category(frame.get("diamond_clarity_raw"))
    frame["diamond_cut_clean"] = _clean_category(frame.get("diamond_cut_raw"))
    frame["design_style_clean"] = _clean_category(frame.get("design_style_and_motif"))
    frame["enamel_color_clean"] = _clean_category(frame.get("enamel_color"))
    frame["certification_lab_clean"] = _clean_category(frame.get("certification_lab_raw"))
    frame["country_of_origin_clean"] = _clean_category(frame.get("country_of_origin"))

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
    _log(verbose, f"Fitting jewelry feature schema with min_category_count={min_category_count}")

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
    _log(verbose, f"Applying jewelry feature schema to {len(frame):,} rows and {len(feature_cols):,} features")

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


def _add_flattened_jewelry_columns(frame: pd.DataFrame) -> pd.DataFrame:
    brand = _flatten_dict_column(frame.get("brand_and_model_identifiers"))
    metal = _flatten_dict_column(frame.get("primary_metal_specification"))
    pearl = _flatten_dict_column(frame.get("pearl_specification"))
    gemstone = _flatten_dict_column(frame.get("primary_gemstone_specification"))
    diamond = _flatten_dict_column(frame.get("diamond_set_specification"))
    cert = _flatten_dict_column(frame.get("inclusions_and_certification"))

    frame["brand_raw"] = brand.get("brand")
    frame["collection_raw"] = brand.get("collection")
    frame["model_raw"] = brand.get("model")
    frame["reference_raw"] = brand.get("reference")
    frame["metal_type_raw"] = metal.get("metal_type")
    frame["metal_purity_raw"] = metal.get("purity")
    frame["metal_color_raw"] = metal.get("color")
    frame["pearl_type_raw"] = pearl.get("type")
    frame["pearl_diameter_raw"] = pearl.get("diameter")
    frame["pearl_count_raw"] = pearl.get("count")
    frame["pearl_color_raw"] = pearl.get("color")
    frame["pearl_luster_raw"] = pearl.get("luster")
    frame["pearl_surface_quality_raw"] = pearl.get("surface_quality")
    frame["gemstone_type_raw"] = gemstone.get("type")
    frame["gemstone_carat_raw"] = gemstone.get("carat")
    frame["gemstone_shape_raw"] = gemstone.get("shape")
    frame["gemstone_color_raw"] = gemstone.get("color")
    frame["gemstone_clarity_raw"] = gemstone.get("clarity")
    frame["gemstone_origin_raw"] = gemstone.get("origin")
    frame["gemstone_treatment_raw"] = gemstone.get("treatment")
    frame["diamond_total_carat_raw"] = diamond.get("total_carat")
    frame["diamond_count_raw"] = diamond.get("count")
    frame["diamond_color_raw"] = diamond.get("color")
    frame["diamond_clarity_raw"] = diamond.get("clarity")
    frame["diamond_cut_raw"] = diamond.get("cut")
    frame["certification_lab_raw"] = cert.get("lab")
    return frame


def _flatten_dict_column(values) -> pd.DataFrame:
    if values is None:
        return pd.DataFrame()
    parsed = [_parse_dict_like(value) for value in pd.Series(values)]
    return pd.DataFrame(parsed)


def _parse_dict_like(value) -> dict:
    if isinstance(value, dict):
        return value
    if value is None or pd.isna(value):
        return {}
    text = str(value).strip()
    if not (text.startswith("{") and text.endswith("}")):
        return {}
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _load_jewelry_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    if path.suffix.lower() == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported jewelry file type: {path}")


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


def _mean_number(values) -> pd.Series:
    if values is None:
        return pd.Series(dtype=float)
    out = []
    for value in pd.Series(values).astype("string"):
        if pd.isna(value):
            out.append(np.nan)
            continue
        nums = pd.Series(str(value)).str.findall(r"-?\d+(?:\.\d+)?").iloc[0]
        if not nums:
            out.append(np.nan)
            continue
        vals = [float(num) for num in nums]
        out.append(float(np.mean(vals)))
    return pd.Series(out, index=pd.Series(values).index, dtype=float)


def _clean_category(values) -> pd.Series:
    if values is None:
        return pd.Series(dtype=object)
    series = pd.Series(values).astype("string").str.strip()
    series = series.replace(
        {
            "": pd.NA,
            "N/A": pd.NA,
            "n/a": pd.NA,
            "NA": pd.NA,
            "None": pd.NA,
            "none": pd.NA,
            "nan": pd.NA,
            "<NA>": pd.NA,
        }
    )
    return series.fillna("__missing__").astype(str)


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(f"[jewelry features] {message}")
