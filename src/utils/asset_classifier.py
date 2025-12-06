#!/usr/bin/env python3
"""
asset_classifier.py – Simple regex‑based asset class mapping.

Maps symbols (e.g. "EURUSD", "AAPL", "BTC") to asset classes
(forex, metal, stock, index, future, crypto).

✅ FIXED: Grouped regex alternatives with parentheses for clarity
Before: r"^XAU|XAG|XPT$"  (ambiguous – matches ^XAU OR XAG OR XPT$)
After:  r"^(XAU|XAG|XPT)$" (explicit – matches exactly XAU, XAG, or XPT)
"""

from __future__ import annotations

import re
from typing import Literal

AssetClass = Literal["forex", "metal", "index", "stock", "future", "crypto"]

# =====================================================================
# Simple regex‑based mapping – extend as you add new symbols
# =====================================================================

# Forex: 6‑letter pairs (EUR+USD, GBP+USD, etc.)
_FOREX_RE = re.compile(r"^(EUR|GBP|USD|JPY|CHF|AUD|NZD)[A-Z]{3}$")

# Metals: Precious metals (Gold, Silver, Platinum)
# ✅ FIXED: Grouped alternatives for clarity
# Before: r"^XAU|XAG|XPT$"  (incorrect precedence)
# After:  r"^(XAU|XAG|XPT)$" (explicit grouping)
_METAL_RE = re.compile(r"^(XAU|XAG|XPT)$")

# Indices: Major stock indices
_INDEX_RE = re.compile(r"^(SPX|NDX|DJI|DAX|FTSE|N225)$")

# Stocks: Plain ticker symbols (1-5 uppercase letters)
_STOCK_RE = re.compile(r"^[A-Z]{1,5}$")

# Futures: Symbols ending with _FUT (e.g. ES_FUT, CL_FUT, NQ_FUT)
_FUTURE_RE = re.compile(r".+_FUT$")

# Crypto: Major cryptocurrencies with optional USD pairing
_CRYPTO_RE = re.compile(r"^(BTC|ETH|SOL|ADA|DOT)(USDT)?$")


def classify_asset(symbol: str) -> AssetClass:
    """
    Return the canonical asset class for a given symbol.

    Matches symbols against regex patterns in order of specificity:
    1. Forex (6-letter pairs like EURUSD)
    2. Metals (XAU, XAG, XPT)
    3. Indices (SPX, NDX, etc.)
    4. Futures (ES_FUT, CL_FUT, etc.)
    5. Crypto (BTC, ETH, etc.)
    6. Stocks (any 1-5 letter uppercase ticker)

    Parameters
    ----------
    symbol : str
        The trading symbol (case-insensitive internally).

    Returns
    -------
    AssetClass
        One of: "forex", "metal", "index", "stock", "future", "crypto"

    Raises
    ------
    ValueError
        If the symbol doesn't match any known pattern.

    Examples
    --------
    >>> classify_asset("EURUSD")
    'forex'
    >>> classify_asset("AAPL")
    'stock'
    >>> classify_asset("XAU")
    'metal'
    >>> classify_asset("BTC")
    'crypto'
    """
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


# =====================================================================
# Optional: Export for convenience
# =====================================================================
__all__ = ["classify_asset", "AssetClass"]
