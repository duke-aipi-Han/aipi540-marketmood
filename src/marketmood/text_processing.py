"""Text formatting helpers for StockTwits-style posts."""

from __future__ import annotations

import re


CASHTAG_PATTERN = re.compile(r"\$[A-Za-z][A-Za-z0-9._-]*")


def build_text_input(original: str, ticker: str, text_format: str = "raw") -> str:
    """Build one of the planned text input formats without using future data."""
    if text_format == "raw":
        return original
    if text_format == "ticker_aware":
        return f"Target ticker: {ticker}. Post: {original}"
    if text_format == "ticker_masked":
        masked = CASHTAG_PATTERN.sub("$TICKER", original)
        return f"Target ticker: $TICKER. Post: {masked}"
    raise ValueError(f"Unknown text format: {text_format}")
