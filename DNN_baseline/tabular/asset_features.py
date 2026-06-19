import ast
import re

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, RobustScaler


def parse_dict_like(value) -> dict:
    if isinstance(value, dict):
        return value
    if pd.isna(value):
        return {}
    if not isinstance(value, str):
        return {}
    text = value.strip()
    if not text or text.lower() in {"none", "nan", "n/a", "null"}:
        return {}
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def clean_na_string(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text.lower() in {"", "none", "nan", "n/a", "null"}:
        return np.nan
    return text


def parse_numeric_value(value):
    text = clean_na_string(value)
    if pd.isna(text):
        return np.nan
    match = re.search(r"-?\d+(?:\.\d+)?", str(text).replace(",", ""))
    if not match:
        return np.nan
    return float(match.group(0))


def parse_numeric_range_mid(value):
    text = clean_na_string(value)
    if pd.isna(text):
        return np.nan
    nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", str(text).replace(",", ""))]
    if not nums:
        return np.nan
    if len(nums) >= 2:
        return float((nums[0] + nums[1]) / 2.0)
    return float(nums[0])


def expand_dict_column(df: pd.DataFrame, column: str, prefix: str, keys: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    if column not in df.columns:
        for key in keys:
            out[f"{prefix}_{key}"] = np.nan
        return out
    parsed = df[column].map(parse_dict_like)
    for key in keys:
        out[f"{prefix}_{key}"] = parsed.map(lambda d: clean_na_string(d.get(key)))
    return out


def parse_year_series(series: pd.Series) -> pd.Series:
    year = series.astype(str).str.extract(r"(\d{4})", expand=False)
    return pd.to_numeric(year, errors="coerce")


def encode_feature_frames(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    numeric_cols: list[str],
    cat_cols: list[str],
):
    numeric_cols = [c for c in numeric_cols if c in x_train.columns]
    cat_cols = [c for c in cat_cols if c in x_train.columns]

    for c in numeric_cols:
        if c == "sale_date":
            tr_dt = pd.to_datetime(x_train[c], errors="coerce")
            te_dt = pd.to_datetime(x_test[c], errors="coerce")
            tr_ns = tr_dt.astype("int64")
            te_ns = te_dt.astype("int64")
            nat = np.iinfo(np.int64).min
            tr_ns = tr_ns.where(tr_ns != nat, np.nan)
            te_ns = te_ns.where(te_ns != nat, np.nan)
            x_train[c] = (tr_ns / 86_400_000_000_000).astype("float64")
            x_test[c] = (te_ns / 86_400_000_000_000).astype("float64")
        elif c in {"year_of_creation", "sale_year"}:
            x_train[c] = parse_year_series(x_train[c])
            x_test[c] = parse_year_series(x_test[c])
        else:
            x_train[c] = pd.to_numeric(x_train[c], errors="coerce")
            x_test[c] = pd.to_numeric(x_test[c], errors="coerce")

    x_num_train = None
    x_num_test = None
    if numeric_cols:
        numeric_cols = [c for c in numeric_cols if x_train[c].notna().any()]
    if numeric_cols:
        # Neural tabular models are sensitive to raw numeric scale and extreme
        # out-of-distribution values. Clip using train-only quantiles, then
        # robust-scale using train-only medians/IQRs.
        lower = x_train[numeric_cols].quantile(0.001)
        upper = x_train[numeric_cols].quantile(0.999)
        x_train[numeric_cols] = x_train[numeric_cols].clip(lower=lower, upper=upper, axis=1)
        x_test[numeric_cols] = x_test[numeric_cols].clip(lower=lower, upper=upper, axis=1)

        num_imputer = SimpleImputer(strategy="median")
        tr_num = num_imputer.fit_transform(x_train[numeric_cols])
        te_num = num_imputer.transform(x_test[numeric_cols])

        scaler = RobustScaler()
        x_num_train = scaler.fit_transform(tr_num).astype(np.float32)
        x_num_test = scaler.transform(te_num).astype(np.float32)

    x_cat_train = None
    x_cat_test = None
    cat_cardinalities = None
    if cat_cols:
        keep_cat_cols = [c for c in cat_cols if x_train[c].notna().any()]
        if keep_cat_cols:
            x_train_cat = x_train[keep_cat_cols].copy()
            x_test_cat = x_test[keep_cat_cols].copy()

            cat_imputer = SimpleImputer(strategy="most_frequent")
            tr_cat = cat_imputer.fit_transform(x_train_cat)
            te_cat = cat_imputer.transform(x_test_cat)

            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            tr_enc = enc.fit_transform(tr_cat)
            te_enc = enc.transform(te_cat)

            x_cat_train = (tr_enc + 1).astype(np.int64)
            x_cat_test = (te_enc + 1).astype(np.int64)
            cat_cardinalities = [int(len(cats) + 1) for cats in enc.categories_]

    return {
        "x_num_train": x_num_train,
        "x_num_test": x_num_test,
        "x_cat_train": x_cat_train,
        "x_cat_test": x_cat_test,
        "cat_cardinalities": cat_cardinalities,
        "n_num_features": 0 if x_num_train is None else x_num_train.shape[1],
    }


def preprocess_artwork_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    use_estimates: bool,
):
    base_cols = [
        "auction_house",
        "location",
        "sale_date",
        "material",
        "artwork_title",
        "medium",
        "length",
        "height",
        "width",
        "unit",
        "year_of_creation",
        "signed",
        "titled",
        "dated",
        "artist_name",
        "edition",
        "GSPC",
        "FTSE",
        "IXIC",
        "DJI",
        "VIX",
        "BZ_F",
        "FVX",
        "provenance",
    ]
    if use_estimates:
        base_cols += ["ah_low_estimate_usd", "ah_high_estimate_usd"]

    cols = [c for c in base_cols if c in train_df.columns]
    x_train = train_df[cols].copy()
    x_test = test_df[cols].copy()

    if {"unit", "length", "height", "width"} & set(x_train.columns):
        unit = x_train.get("unit", pd.Series("", index=x_train.index)).astype(str).str.strip().str.lower()
        te_unit = x_test.get("unit", pd.Series("", index=x_test.index)).astype(str).str.strip().str.lower()
        factor = np.where(unit.eq("in"), 2.54, np.where(unit.eq("mm"), 0.1, np.where(unit.eq("ft"), 30.48, 1.0)))
        te_factor = np.where(te_unit.eq("in"), 2.54, np.where(te_unit.eq("mm"), 0.1, np.where(te_unit.eq("ft"), 30.48, 1.0)))
        for c in ["length", "height", "width"]:
            if c in x_train.columns:
                x_train[c] = pd.to_numeric(x_train[c], errors="coerce") * factor
                x_test[c] = pd.to_numeric(x_test[c], errors="coerce") * te_factor

    if "provenance" in train_df.columns:
        x_train["has_provenance"] = (
            train_df["provenance"].notna()
            & (train_df["provenance"].astype(str).str.strip().str.lower() != "none")
        ).astype(np.int64)
        x_test["has_provenance"] = (
            test_df["provenance"].notna()
            & (test_df["provenance"].astype(str).str.strip().str.lower() != "none")
        ).astype(np.int64)
    if "edition" in train_df.columns:
        x_train["is_editioned"] = train_df["edition"].notna().astype(np.int64)
        x_test["is_editioned"] = test_df["edition"].notna().astype(np.int64)

    for c in ["provenance", "unit"]:
        if c in x_train.columns:
            x_train = x_train.drop(columns=[c])
            x_test = x_test.drop(columns=[c])

    cat_cols = [
        "auction_house",
        "location",
        "material",
        "artwork_title",
        "medium",
        "signed",
        "titled",
        "dated",
        "artist_name",
        "edition",
    ]
    numeric_cols = [
        "year_of_creation",
        "sale_date",
        "length",
        "height",
        "width",
        "GSPC",
        "FTSE",
        "IXIC",
        "DJI",
        "VIX",
        "BZ_F",
        "FVX",
        "has_provenance",
        "is_editioned",
    ]
    if use_estimates:
        numeric_cols += ["ah_low_estimate_usd", "ah_high_estimate_usd"]

    return encode_feature_frames(x_train, x_test, numeric_cols, cat_cols)


def build_jewelry_frame(df: pd.DataFrame, use_estimates: bool) -> pd.DataFrame:
    base_cols = [
        "auction_house",
        "location",
        "sale_date",
        "product_name",
        "maker_signatures_and_hallmarks",
        "jewelry_category_and_subtypes",
        "design_style_and_motif",
        "enamel_color",
        "country_of_origin",
        "total_item_weight_measurement",
        "ring_size_measurement",
        "necklace_length_measurement",
        "bracelet_length_measurement",
        "earring_length_measurement",
        "pendant_length_measurement",
        "chain_length_measurement",
        "gemstone_count",
        "diamond_count",
        "pearl_count",
        "item_count",
        "sale_year",
        "GSPC",
        "FTSE",
        "IXIC",
        "DJI",
        "VIX",
        "BZ_F",
        "FVX",
    ]
    if use_estimates:
        base_cols += ["ah_low_estimate_usd", "ah_high_estimate_usd"]
    x = df[[c for c in base_cols if c in df.columns]].copy()

    dict_specs = {
        "brand_and_model_identifiers": ("brand_model", ["brand", "collection", "model", "reference"]),
        "primary_metal_specification": ("metal", ["metal_type", "purity", "color"]),
        "pearl_specification": ("pearl_spec", ["type", "diameter", "count", "color", "luster", "surface_quality"]),
        "primary_gemstone_specification": (
            "primary_gemstone",
            ["type", "carat", "shape", "color", "clarity", "measurements"],
        ),
        "diamond_set_specification": ("diamond_set", ["total_carat", "count", "color", "clarity", "cut"]),
        "inclusions_and_certification": ("cert", ["lab", "report_number", "inclusions_note"]),
    }
    for source_col, (prefix, keys) in dict_specs.items():
        expanded = expand_dict_column(df, source_col, prefix, keys)
        x = pd.concat([x, expanded], axis=1)

    range_cols = [
        "primary_gemstone_carat",
        "diamond_set_total_carat",
        "diamond_set_count",
        "pearl_spec_diameter",
        "pearl_spec_count",
        "pearl_count",
    ]
    for col in range_cols:
        if col in x.columns:
            x[f"{col}_num"] = x[col].map(parse_numeric_range_mid)

    bool_like_cols = []
    for col in ["cert_report_number", "maker_signatures_and_hallmarks"]:
        if col in x.columns:
            x[f"has_{col}"] = x[col].notna().astype(np.int64)
            bool_like_cols.append(f"has_{col}")

    return x


def preprocess_jewelry_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    use_estimates: bool,
):
    x_train = build_jewelry_frame(train_df, use_estimates)
    x_test = build_jewelry_frame(test_df, use_estimates)

    numeric_cols = [
        "sale_date",
        "sale_year",
        "total_item_weight_measurement",
        "ring_size_measurement",
        "necklace_length_measurement",
        "bracelet_length_measurement",
        "earring_length_measurement",
        "pendant_length_measurement",
        "chain_length_measurement",
        "gemstone_count",
        "diamond_count",
        "pearl_count",
        "item_count",
        "primary_gemstone_carat_num",
        "diamond_set_total_carat_num",
        "diamond_set_count_num",
        "pearl_spec_diameter_num",
        "pearl_spec_count_num",
        "pearl_count_num",
        "has_cert_report_number",
        "has_maker_signatures_and_hallmarks",
        "GSPC",
        "FTSE",
        "IXIC",
        "DJI",
        "VIX",
        "BZ_F",
        "FVX",
    ]
    if use_estimates:
        numeric_cols += ["ah_low_estimate_usd", "ah_high_estimate_usd"]

    cat_cols = [
        "auction_house",
        "location",
        "product_name",
        "maker_signatures_and_hallmarks",
        "jewelry_category_and_subtypes",
        "design_style_and_motif",
        "enamel_color",
        "country_of_origin",
        "brand_model_brand",
        "brand_model_collection",
        "brand_model_model",
        "brand_model_reference",
        "metal_metal_type",
        "metal_purity",
        "metal_color",
        "pearl_spec_type",
        "pearl_spec_color",
        "pearl_spec_luster",
        "pearl_spec_surface_quality",
        "primary_gemstone_type",
        "primary_gemstone_shape",
        "primary_gemstone_color",
        "primary_gemstone_clarity",
        "primary_gemstone_measurements",
        "diamond_set_color",
        "diamond_set_clarity",
        "diamond_set_cut",
        "cert_lab",
        "cert_report_number",
        "cert_inclusions_note",
    ]

    return encode_feature_frames(x_train, x_test, numeric_cols, cat_cols)


def build_watch_frame(df: pd.DataFrame, use_estimates: bool) -> pd.DataFrame:
    base_cols = [
        "auction_house",
        "location",
        "auction_title",
        "sale_date",
        "sale_year",
        #"currency",
        "brand",
        "model",
        "ref_no",
        "year_manufactured",
        "case_size",
        "box",
        "papers",
        "condition",
        "case_material",
        "band_material",
        "karats",
        "gold_present",
        "gold_type",
        "diamonds",
        "diamonds_quantity",
        "dial_color",
        "GSPC",
        "FTSE",
        "IXIC",
        "DJI",
        "VIX",
        "BZ_F",
        "FVX",
    ]
    if use_estimates:
        base_cols += ["ah_low_estimate_usd", "ah_high_estimate_usd"]

    x = df[[c for c in base_cols if c in df.columns]].copy()

    if "case_size" in x.columns:
        x["case_size_num"] = x["case_size"].map(parse_numeric_range_mid)

    for col in [
        "box",
        "papers",
        "karats",
        "gold_present",
        "diamonds",
        "diamonds_quantity",
        "year_manufactured",
    ]:
        if col in x.columns:
            x[col] = x[col].map(parse_numeric_value)

    return x


def preprocess_watch_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    use_estimates: bool,
):
    x_train = build_watch_frame(train_df, use_estimates)
    x_test = build_watch_frame(test_df, use_estimates)

    numeric_cols = [
        "sale_date",
        "sale_year",
        "year_manufactured",
        "case_size_num",
        "box",
        "papers",
        "karats",
        "gold_present",
        "diamonds",
        "diamonds_quantity",
        "GSPC",
        "FTSE",
        "IXIC",
        "DJI",
        "VIX",
        "BZ_F",
        "FVX",
    ]
    if use_estimates:
        numeric_cols += ["ah_low_estimate_usd", "ah_high_estimate_usd"]

    cat_cols = [
        "auction_house",
        "location",
        "auction_title",
        #"currency",
        "brand",
        "model",
        "ref_no",
        "condition",
        "case_material",
        "band_material",
        "gold_type",
        "dial_color",
    ]

    return encode_feature_frames(x_train, x_test, numeric_cols, cat_cols)


def preprocess_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    drop_cols: list[str],
    use_estimates: bool,
    asset_type: str,
):
    _ = {
        target_col,
        "ah_bp_price",
        "ah_price",
        "ah_price_usd",
        "ah_low_estimate_usd",
        "ah_high_estimate_usd",
        "__quarter__",
    }
    _ = set(drop_cols)
    if asset_type == "jewelry":
        return preprocess_jewelry_features(train_df, test_df, use_estimates)
    if asset_type == "watch":
        return preprocess_watch_features(train_df, test_df, use_estimates)
    return preprocess_artwork_features(train_df, test_df, use_estimates)
