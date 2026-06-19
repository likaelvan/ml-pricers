from __future__ import annotations

import math

import pandas as pd

TARGET_AND_LEAKAGE_COLS = {
    "ah_price",
    "ah_price_usd",
    "ah_bp_price",
    "ah_bp_price_usd",
    "hammer_price_bp",
    "hammer_price_original",
    "watch-Auction-new-loss-gemma3-4B-text-pricer:latest",
    "midpoint",
    "relative_diff",
    "row_based_outlier",
}

MARKET_COLS = {"GSPC", "FTSE", "IXIC", "DJI", "VIX", "BZ_F", "FVX"}
ESTIMATE_COLS = {"ah_low_estimate_usd", "ah_high_estimate_usd", "ah_low_estimate", "ah_high_estimate"}

PROMPT_FIELD_ORDER = [
    "asset_type",
    "brand",
    "model",
    "ref_no",
    "year_manufactured",
    "case_size",
    "case_material",
    "band_material",
    "dial_color",
    "condition",
    "box",
    "papers",
    "karats",
    "gold_present",
    "gold_type",
    "diamonds",
    "diamonds_quantity",
    "description_block",
    "auction_house",
    "auction_title",
    "location",
    "sale_date",
    "lot_number",
    "currency",
    "ah_low_estimate_usd",
    "ah_high_estimate_usd",
    "lot_status",
    "including_premium",
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
        "Use the low and high estimates to understand expected market range, but synthesize them with brand, model, reference, condition, materials, box/papers status, and auction context. "
        if include_auction_estimates
        else "Do not use auction-house estimate fields for valuation even if they appear in metadata. "
    )
    return (
        "You are an expert luxury watch appraiser. Price the watch's expected sale value in USD under normal auction-market conditions. "
        "Use only the watch and auction metadata provided in the user prompt. "
        "Do not use realized sale price, buyer-premium price, or other model prediction columns as inputs. "
        f"{estimate_guidance}"
        "Consider brand, model, reference number, production period, case size, case and band material, condition, box and papers, precious metals, diamonds, sale venue, location, sale date, and available market indicators. "
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
        if key == "description_block":
            text = " ".join(str(value).split())[:max_text_chars]
            fields.append(f'{label}: """{text}"""')
        else:
            fields.append(f"{label}: {value}")

    return (
        "Price this luxury watch using the available fields below. "
        "Do not use any realized sale price or model prediction as an input.\n\n"
        "Watch data:\n" + "\n".join(fields)
    )


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
