from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


QUARTER_PATTERN = re.compile(r"q?(20\d{2}Q[1-4])", re.IGNORECASE)

TARGET_AND_LEAKAGE_COLS = {
    "ah_price",
    "ah_price_usd",
    "ah_bp_price",
    "ah_bp_price_usd",
    "barker_price_premium_usd",
    "barker_price_hammer_usd",
    "pricing_model",
    "llama3.2:3b",
    "artwork-Auction-2026-llama3.2-3B-text-pricer:latest",
    "gemma3:4b",
    "artwork-Auction-new-loss-gemma3-4B-text-pricer:latest",
    "artwork-Auction-2026-gemma3-4B-text-pricer:latest",
    "artwork-gemma3-4B-text-pricer:latest",
    "log_price",
    "price_regime",
    "price_bin",
}

UNAVAILABLE_CATALOG_BLOCKS = {
    "appraisals",
    "comparables",
    "artist_sale_report",
    "sale_report",
    "macro_report",
    "loan_amount",
    "ltv",
    "pledge_number",
}

PROMPT_FIELD_ORDER = [
    "asset_type",
    "artist_name",
    "artist_birth",
    "artist_death",
    "artwork_title",
    "year_of_creation",
    "medium",
    "material",
    "subject_matter",
    "edition",
    "length",
    "height",
    "width",
    "unit",
    "signed",
    "titled",
    "dated",
    "provenance",
    "description_block",
    "auction_house",
    "auction_title",
    "location",
    "sale_date",
    "lot_number",
    "currency",
    "ah_low_estimate_usd",
    "ah_high_estimate_usd",
    "source_sale",
    "house_name",
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

NA_VALUES = {
    "n/a",
    "na",
    "none",
    "<n/a>",
    "<na>",
    "nan",
    "null",
    "<null>",
    "missing",
    "undef",
    "undefined",
    "not available",
    "",
}


def load_quarter_files(
    data_dir: str | Path,
    *,
    quarters: Iterable[str] | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Load q2025 quarter CSV/JSON files and attach source_quarter/source_file."""

    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir.resolve()}")

    requested_quarters = set(quarters) if quarters else None
    paths = sorted([*data_dir.glob("*.csv"), *data_dir.glob("*.json")])
    frames = []
    for path in paths:
        quarter = _quarter_from_filename(path.name)
        if quarter is None:
            continue
        if requested_quarters and quarter not in requested_quarters:
            continue

        frame = _load_file(path)
        frame["source_file"] = path.name
        frame["source_quarter"] = quarter
        frames.append(frame)
        if verbose:
            print(f"[openai data] loaded {path.name}: {len(frame):,} rows")

    if not frames:
        raise FileNotFoundError(f"No quarter CSV/JSON files found in {data_dir.resolve()}")

    df = pd.concat(frames, ignore_index=True)
    if "sale_date" in df.columns:
        df["sale_date"] = pd.to_datetime(df["sale_date"], errors="coerce").dt.date.astype(str)
    if verbose:
        print(f"[openai data] combined quarter rows: {len(df):,}")
        print(f"[openai data] quarters: {df['source_quarter'].value_counts().sort_index().to_dict()}")
    return df


def load_explicit_quarter_files(
    files: dict[str, str | Path],
    *,
    verbose: bool = False,
) -> pd.DataFrame:
    """Load an explicit quarter->file mapping and attach source_quarter/source_file."""

    frames = []
    for quarter, path_value in files.items():
        path = Path(path_value)
        if not path.exists():
            raise FileNotFoundError(f"Quarter file not found for {quarter}: {path.resolve()}")
        frame = _load_file(path)
        frame["source_file"] = path.name
        frame["source_quarter"] = quarter
        frames.append(frame)
        if verbose:
            print(f"[openai data] loaded {quarter} {path.name}: {len(frame):,} rows")

    if not frames:
        raise FileNotFoundError("No explicit quarter files were provided.")

    df = pd.concat(frames, ignore_index=True)
    if "sale_date" in df.columns:
        df["sale_date"] = pd.to_datetime(df["sale_date"], errors="coerce").dt.date.astype(str)
    if verbose:
        print(f"[openai data] combined quarter rows: {len(df):,}")
        print(f"[openai data] quarters: {df['source_quarter'].value_counts().sort_index().to_dict()}")
    return df


def prepare_pricing_frame(
    df: pd.DataFrame,
    *,
    rows_per_quarter: int | None = None,
    max_rows: int | None = None,
    random_state: int = 42,
    dedupe_key_col: str = "lot_url",
    verbose: bool = False,
) -> pd.DataFrame:
    """Deduplicate and optionally sample rows to price."""

    frame = df.copy()
    if dedupe_key_col in frame.columns:
        before = len(frame)
        frame = _drop_duplicate_nonmissing_keys(frame, dedupe_key_col)
        if verbose:
            print(f"[openai data] deduplicated {dedupe_key_col}: {before:,}->{len(frame):,}")

    if rows_per_quarter is not None:
        sampled = []
        for quarter, group in frame.groupby("source_quarter", dropna=False):
            n = min(rows_per_quarter, len(group))
            if verbose and n < rows_per_quarter:
                print(f"[openai data] requested {rows_per_quarter:,} for {quarter}; using {n:,}")
            sampled.append(group.sample(n=n, random_state=random_state))
        frame = pd.concat(sampled, ignore_index=True)
        if verbose:
            print(f"[openai data] after per-quarter sample: {len(frame):,}")

    if max_rows is not None and len(frame) > max_rows:
        frame = frame.sample(n=max_rows, random_state=random_state).reset_index(drop=True)
        if verbose:
            print(f"[openai data] after max_rows sample: {len(frame):,}")

    return frame.reset_index(drop=True)


def build_system_prompt(*, include_auction_estimates: bool = True) -> str:
    """Trimmed artwork valuation system prompt based on the local Barker catalog."""

    estimate_guidance = (
        "Pre-sale auction estimates may be provided. Treat them as professional directional inputs, not as the final answer. "
        "Use the low and high estimates to understand expected market range, but synthesize them with artist, medium, size, provenance, sale venue, and description. "
        "You may price outside the estimate range if the artwork facts justify it. "
        if include_auction_estimates
        else "Do not use auction-house estimate fields for valuation even if they appear in metadata. "
    )

    return (
        "You are an expert art appraiser. Price the artwork's expected sale value in USD under normal auction-market conditions. "
        "Use only the artwork and auction metadata provided in the user prompt. "
        "Ignore unavailable catalog concepts such as third-party appraisals, comparables, artist sale reports, sale reports, loan metadata, LTV, and macroeconomic narrative reports unless explicitly present in the prompt. "
        "Do not use realized sale price, buyer-premium price, Barker model prices, or other model prediction columns as inputs. "
        f"{estimate_guidance}"
        "Consider artist, title, creation year, medium/material, dimensions, edition, annotations, provenance, subject matter, sale venue, location, sale date, and available market indicators. "
        "Return a realistic nonzero USD valuation. "
        "Respond only as JSON matching the requested schema."
    )


def build_artwork_prompt(
    row: pd.Series | dict,
    *,
    include_auction_estimates: bool = True,
    include_market_indicators: bool = True,
    max_text_chars: int = 1800,
) -> str:
    """Build a compact prompt for one artwork row."""

    record = _clean_record(dict(row))
    fields = []
    skipped = set(TARGET_AND_LEAKAGE_COLS) | set(UNAVAILABLE_CATALOG_BLOCKS)
    if not include_auction_estimates:
        skipped.update({"ah_low_estimate_usd", "ah_high_estimate_usd", "ah_low_estimate", "ah_high_estimate"})
    if not include_market_indicators:
        skipped.update({"GSPC", "FTSE", "IXIC", "DJI", "VIX", "BZ_F", "FVX"})

    for key in PROMPT_FIELD_ORDER:
        if key in skipped:
            continue
        value = record.get(key)
        if value is None:
            continue
        label = _format_label(key)
        if key in {"description_block", "provenance"}:
            text = " ".join(str(value).split())[:max_text_chars]
            fields.append(f'{label}: """{text}"""')
        else:
            fields.append(f"{label}: {value}")

    return (
        "Price this artwork using the available fields below. "
        "Do not use any realized sale price or model prediction as an input.\n\n"
        "Artwork data:\n"
        + "\n".join(fields)
    )


def _load_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    if path.suffix.lower() == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported file type: {path}")


def _quarter_from_filename(filename: str) -> str | None:
    match = QUARTER_PATTERN.search(filename)
    return match.group(1).upper() if match else None


def _clean_record(record: dict) -> dict:
    out = {}
    for key, value in record.items():
        if isinstance(value, str):
            stripped = value.strip()
            out[key] = None if stripped.lower() in NA_VALUES else stripped
        elif value is None:
            out[key] = None
        elif isinstance(value, float) and np.isnan(value):
            out[key] = None
        elif pd.isna(value):
            out[key] = None
        else:
            out[key] = value
    return out


def _format_label(key: str) -> str:
    return key.replace("_", " ").strip().title()


def _drop_duplicate_nonmissing_keys(df: pd.DataFrame, key_col: str) -> pd.DataFrame:
    key = df[key_col]
    missing = key.isna()
    deduped = df[~missing].drop_duplicates(subset=[key_col], keep="first")
    if missing.any():
        deduped = pd.concat([deduped, df[missing]], ignore_index=False)
    return deduped.sort_index().copy()
