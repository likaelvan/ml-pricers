from __future__ import annotations

from openai_pricer.prompt_builder import build_artwork_prompt, build_system_prompt as _build_artwork_system_prompt


def build_system_prompt(include_auction_estimates: bool = True) -> str:
    return _build_artwork_system_prompt(include_auction_estimates=include_auction_estimates)


def build_prompt(row: dict, include_auction_estimates: bool, include_market_indicators: bool) -> str:
    return build_artwork_prompt(
        row,
        include_auction_estimates=include_auction_estimates,
        include_market_indicators=include_market_indicators,
    )


__all__ = ["build_prompt", "build_system_prompt"]
