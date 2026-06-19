from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any


PRICE_RESPONSE_SCHEMA = {
    "type": "json_schema",
    "name": "asset_price_response",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "predicted_price_usd": {
                "type": "number",
                "description": "Predicted asset value in USD. Must be greater than zero.",
            },
            "confidence": {
                "type": "string",
                "enum": ["low", "medium", "high"],
            },
            "pricing_basis": {
                "type": "string",
                "enum": ["estimate_anchored", "metadata_only", "limited_data"],
            },
            "notes": {
                "type": "string",
                "description": "Short reason, 240 characters or fewer.",
            },
        },
        "required": ["predicted_price_usd", "confidence", "pricing_basis", "notes"],
    },
}


@dataclass
class PriceResult:
    predicted_price_usd: float | None
    confidence: str | None
    pricing_basis: str | None
    notes: str | None
    raw_output: str | None
    error: str | None
    latency_seconds: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "openai_predicted_price_usd": self.predicted_price_usd,
            "openai_confidence": self.confidence,
            "openai_pricing_basis": self.pricing_basis,
            "openai_notes": self.notes,
            "openai_raw_output": self.raw_output,
            "openai_error": self.error,
            "openai_latency_seconds": self.latency_seconds,
            "openai_input_tokens": self.input_tokens,
            "openai_output_tokens": self.output_tokens,
            "openai_total_tokens": self.total_tokens,
        }


class OpenAIPricer:
    """Small wrapper around the OpenAI Responses API for asset pricing."""

    def __init__(
        self,
        *,
        model: str = "gpt-4.1-mini",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_output_tokens: int = 300,
        max_retries: int = 3,
        retry_sleep_seconds: float = 2.0,
        use_structured_output: bool = True,
    ) -> None:
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "The openai package is not installed. Install it before running API pricing cells, e.g. pip install openai"
            ) from exc

        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
        self.retry_sleep_seconds = retry_sleep_seconds
        self.use_structured_output = use_structured_output
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def price(self, *, system_prompt: str, user_prompt: str) -> PriceResult:
        started = time.perf_counter()
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                kwargs = {
                    "model": self.model,
                    "instructions": system_prompt,
                    "input": user_prompt,
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_output_tokens,
                }
                if self.use_structured_output:
                    kwargs["text"] = {"format": PRICE_RESPONSE_SCHEMA}

                response = self.client.responses.create(**kwargs)
                latency = time.perf_counter() - started
                raw_output = getattr(response, "output_text", None) or _extract_output_text(response)
                parsed = parse_price_response(raw_output)
                usage = _extract_usage(response)
                return PriceResult(
                    predicted_price_usd=parsed.get("predicted_price_usd"),
                    confidence=parsed.get("confidence"),
                    pricing_basis=parsed.get("pricing_basis"),
                    notes=parsed.get("notes"),
                    raw_output=raw_output,
                    error=None,
                    latency_seconds=latency,
                    **usage,
                )
            except Exception as exc:  # noqa: BLE001 - API wrappers should capture and checkpoint failures.
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < self.max_retries:
                    time.sleep(self.retry_sleep_seconds * attempt)

        return PriceResult(
            predicted_price_usd=None,
            confidence=None,
            pricing_basis=None,
            notes=None,
            raw_output=None,
            error=last_error,
            latency_seconds=time.perf_counter() - started,
        )


def parse_price_response(raw_output: str | None) -> dict[str, Any]:
    """Parse model JSON output and enforce a positive price when possible."""

    if raw_output is None:
        raise ValueError("Empty model output")

    text = raw_output.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Last-resort fallback for accidental plain numeric output.
        cleaned = text.replace("$", "").replace(",", "").strip()
        parsed = {"predicted_price_usd": float(cleaned), "confidence": None, "pricing_basis": None, "notes": None}

    price = parsed.get("predicted_price_usd")
    if price is None:
        raise ValueError(f"Missing predicted_price_usd in output: {raw_output}")
    price = float(price)
    if price <= 0:
        raise ValueError(f"Model returned non-positive price: {price}")
    parsed["predicted_price_usd"] = price
    return parsed


def _extract_output_text(response) -> str | None:
    try:
        chunks = []
        for item in response.output:
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", None)
                if text:
                    chunks.append(text)
        return "\n".join(chunks) if chunks else None
    except Exception:
        return None


def _extract_usage(response) -> dict[str, int | None]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"input_tokens": None, "output_tokens": None, "total_tokens": None}
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }
