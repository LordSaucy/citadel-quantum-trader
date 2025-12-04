# src/utils/asset_classifier.py
from __future__ import annotations
import re
from typing import Literal

AssetClass = Literal["forex", "metal", "index", "stock", "future", "crypto"]

# Simple regex‑based mapping – extend as you add new symbols
_FOREX_RE   = re.compile(r"^(EUR|GBP|USD|JPY|CHF|AUD|NZD)[A-Z]{3}$")
_METAL_RE   = re.compile(r"^XAU|XAG|XPT$")
_INDEX_RE   = re.compile(r"^(SPX|NDX|DJI|DAX|FTSE|N225)$")
_STOCK_RE   = re.compile(r"^[A-Z]{1,5}$")          # plain ticker (AAPL, TSLA, etc.)
_FUTURE_RE  = re.compile(r".+_FUT$")              # e.g. ES_FUT, CL_FUT
_CRYPTO_RE  = re.compile(r"^(BTC|ETH|SOL|ADA|DOT)(USDT)?$")

def classify_asset(symbol: str) -> AssetClass:
    """Return the canonical asset class for a given symbol."""
    sym = symbol.upper()
    if _FOREX_RE.fullmatch(sym):
        return "forex"
    if _METAL_RE.fullmatch(sym):
        return "metal"
    if _INDEX_RE.fullmatch(sym):
        return "index"
    if _FUTURE_RE.fullmatch(sym):
        return "future"
    if _CRYPTO_RE.fullmatch(sym):
        return "crypto"
    # Anything that looks like a plain ticker is assumed to be a stock.
    # You can refine this by checking a whitelist of supported equities.
    if _STOCK_RE.fullmatch(sym):
        return "stock"
    raise ValueError(f"Unable to classify asset class for symbol '{symbol}'")
