#!/usr/bin/env python3
"""
depth_guard.py

Utility that checks whether the market depth at a prospective entry price
is sufficient to satisfy the required dollar risk (stake).  It is used as a
*hard stop* before an order is sent – the trade is simply rejected if the
available volume is too low, but the risk‑fraction for the bucket is still
deducted (so win‑rate statistics stay honest).

The function is deliberately pure (no side‑effects) so it can be unit‑tested
easily.
"""

from __future__ import annotations

import logging
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Helper – convert a USD risk amount into a lot size for a given symbol.
# -------------------------------------------------------------------------
def price_to_lots(entry_price: float, usd_amount: float, lot_size_units: float = 100_000) -> float:
    """
    Convert a dollar risk (usd_amount) into a lot quantity.

    Parameters
    ----------
    entry_price: float
        The price at which the trade would be entered.
    usd_amount: float
        Dollar amount you are willing to risk on the trade.
    lot_size_units: float, optional
        Number of base‑currency units per 1.0 lot.  Default = 100 000 (standard
        FX lot).  For mini‑lots you can pass 10 000, etc.

    Returns
    -------
    float
        Lot size (positive for long, negative for short).  The sign is left
        to the caller – this helper only returns the absolute magnitude.
    """
    # For FX the risk per pip = (lot_size_units * pip_value) / entry_price.
    # A simple approximation: 1 pip ≈ 0.0001 for most majors.
    # We invert the formula to get the lot size that would lose `usd_amount`
    # if the price moved 1 pip against us.
    pip = 0.0001
    lot = usd_amount / (pip * lot_size_units / entry_price)
    return lot


# -------------------------------------------------------------------------
# Depth‑guard implementation
# -------------------------------------------------------------------------
def depth_guard(
    entry_price: float,
    dom_df: pd.DataFrame,
    required_volume_usd: float,
    *,
    tick: float = 0.0001,
    safety_multiplier: float = 2.0,
) -> bool:
    """
    Return ``True`` if the market depth at ``entry_price`` is sufficient.

    Parameters
    ----------
    entry_price: float
        Desired entry price for the trade.
    dom_df: pd.DataFrame
        Full depth‑of‑market snapshot.  Expected columns:
        ``price``, ``bid_volume`` and ``ask_volume`` (numeric).
    required_volume_usd: float
        Dollar amount you intend to risk (the *stake*).
    tick: float, optional
        Minimum price increment for the instrument (default 0.0001 for most
        FX pairs).  Override for indices, commodities, etc.
    safety_multiplier: float, optional
        How many times the needed volume we require as a cushion.
        ``2.0`` means “at least twice the volume we need”.

    Returns
    -------
    bool
        ``True`` → enough depth, safe to send the order.
        ``False`` → reject the trade (insufficient liquidity).
    """
    # -----------------------------------------------------------------
    # 1️⃣  Snap to the nearest price level that exists in the DOM.
    # -----------------------------------------------------------------
    nearest_price = round(entry_price / tick) * tick

    # Find the row whose price is closest to ``nearest_price``.
    # ``abs(...).argsort()[:1]`` returns the index of the nearest row.
    idx = (dom_df["price"] - nearest_price).abs().argsort()[:1]
    if idx.empty:
        logger.debug("Depth guard: no price level found for %.5f", nearest_price)
        return False

    row = dom_df.iloc[idx]

    # -----------------------------------------------------------------
    # 2️⃣  Convert the USD risk into a lot size.
    # -----------------------------------------------------------------
    lots_needed = price_to_lots(entry_price, required_volume_usd)

    # -----------------------------------------------------------------
    # 3️⃣  Determine which side of the book we need.
    # -----------------------------------------------------------------
    if lots_needed > 0:                     # LONG → need ASK volume
        available_vol = row["ask_volume"].iloc[0]
        side = "ASK"
    else:                                   # SHORT → need BID volume
        available_vol = row["bid_volume"].iloc[0]
        side = "BID"

    # -----------------------------------------------------------------
    # 4️⃣  Apply the safety multiplier.
    # -----------------------------------------------------------------
    needed = safety_multiplier * abs(lots_needed)

    ok = available_vol >= needed
    logger.debug(
        "Depth guard @ %.5f – side=%s, needed=%.2f, available=%.2f → %s",
        nearest_price,
        side,
        needed,
        available_vol,
        "PASS" if ok else "FAIL",
    )
    return ok
