from __future__ import annotations

import ast
import math

import pandas as pd

TARGET_AND_LEAKAGE_COLS = {
    "ah_price",
    "ah_price_usd",
    "ah_bp_price",
    "ah_bp_price_usd",
    "log_price",
    "price_regime",
    "price_bin",
    "midpoint_abs_pct_error_vs_ah_price_usd",
    "jewelry-Auction-2025-gemma3-4B-text-pricer:latest",
}

MARKET_COLS = {"GSPC", "FTSE", "IXIC", "DJI", "VIX", "BZ_F", "FVX"}
ESTIMATE_COLS = {"ah_low_estimate_usd", "ah_high_estimate_usd", "ah_low_estimate", "ah_high_estimate", "estimate_midpoint_usd"}
DICT_LIKE_COLS = {
    "brand_and_model_identifiers",
    "primary_metal_specification",
    "pearl_specification",
    "primary_gemstone_specification",
    "diamond_set_specification",
    "inclusions_and_certification",
}

PROMPT_FIELD_ORDER = [
    "asset_type",
    "product_name",
    "brand_and_model_identifiers",
    "maker_signatures_and_hallmarks",
    "primary_metal_specification",
    "total_item_weight_measurement",
    "ring_size_measurement",
    "jewelry_category_and_subtypes",
    "necklace_length_measurement",
    "bracelet_length_measurement",
    "earring_length_measurement",
    "pendant_length_measurement",
    "chain_length_measurement",
    "gemstone_count",
    "diamond_count",
    "pearl_count",
    "pearl_specification",
    "primary_gemstone_specification",
    "diamond_set_specification",
    "design_style_and_motif",
    "enamel_color",
    "inclusions_and_certification",
    "country_of_origin",
    "item_count",
    "description_block",
    "auction_house",
    "auction_title",
    "location",
    "sale_date",
    "lot_number",
    "currency",
    "ah_low_estimate_usd",
    "ah_high_estimate_usd",
    "GSPC",
    "FTSE",
    "IXIC",
    "DJI",
    "VIX",
    "BZ_F",
    "FVX",
    "lot_url",
]

NA_VALUES = {"", "n/a", "na", "none", "nan", "null", "missing", "not available", "undefined"}


def build_system_prompt(include_auction_estimates: bool = True) -> str:
    estimate_guidance = (
        "Pre-sale auction estimates may be provided. Treat them as professional directional inputs, not as the final answer. "
        "Use the low and high estimates to understand expected market range, but synthesize them with maker, metal, gemstones, diamonds, certifications, condition clues, and auction context. "
        if include_auction_estimates
        else "Do not use auction-house estimate fields for valuation even if they appear in metadata. "
    )
    return (
        "You are an expert jewelry appraiser. Price the jewelry lot's expected sale value in USD under normal auction-market conditions. "
        "Use only the jewelry and auction metadata provided in the user prompt. "
        "Do not use realized sale price, buyer-premium price, or other model prediction columns as inputs. "
        f"{estimate_guidance}"
        "Consider maker, category, metal, weight, ring or chain measurements, gemstone and diamond specifications, pearl details, signatures, certification, origin, sale venue, location, sale date, and available market indicators. "
        "Return a realistic nonzero USD valuation. Respond only as JSON matching the requested schema."
    )


def build_prompt(row: dict, include_auction_estimates: bool, include_market_indicators: bool, max_text_chars: int = 1800) -> str:
    record = _clean_record(row)
    skipped = set(TARGET_AND_LEAKAGE_COLS)
    if not include_auction_estimates:
        skipped.update(ESTIMATE_COLS)
    if not include_market_indicators:
        skipped.update(MARKET_COLS)

    fields = []
    for key in PROMPT_FIELD_ORDER:
        if key in skipped:
            continue
        value = record.get(key)
        if value is None:
            continue
        label = _format_label(key)
        if key in DICT_LIKE_COLS:
            value = _format_dict_like(value)
        if key == "description_block":
            text = " ".join(str(value).split())[:max_text_chars]
            fields.append(f'{label}: """{text}"""')
        else:
            fields.append(f"{label}: {value}")

    return (
        "Price this jewelry lot using the available fields below. "
        "Do not use any realized sale price or model prediction as an input.\n\n"
        "Jewelry data:\n" + "\n".join(fields)
    )


def _format_dict_like(value) -> str:
    if isinstance(value, dict):
        data = value
    elif isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            data = parsed if isinstance(parsed, dict) else value
        except (ValueError, SyntaxError):
            data = value
    else:
        data = value
    if not isinstance(data, dict):
        return str(data)
    parts = []
    for key, val in data.items():
        if val in (None, ""):
            continue
        parts.append(f"{_format_label(str(key))}: {val}")
    return "; ".join(parts) if parts else str(data)


def _clean_record(record: dict) -> dict:
    out = {}
    for key, value in dict(record).items():
        if isinstance(value, str):
            stripped = value.strip()
            out[key] = None if stripped.lower() in NA_VALUES else stripped
        elif value is None:
            out[key] = None
        elif isinstance(value, float) and math.isnan(value):
            out[key] = None
        elif pd.isna(value):
            out[key] = None
        else:
            out[key] = value
    return out


def _format_label(key: str) -> str:
    return key.replace("_", " ").strip().title()
