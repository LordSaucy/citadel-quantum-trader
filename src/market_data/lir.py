#!/usr/bin/env python3
"""
lir.py – Liquidity‑Imbalance Ratio utilities
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)

def compute_lir(dom_df: pd.DataFrame) -> float:
    """
    LIR = (Σ bid_vol – Σ ask_vol) / (Σ bid_vol + Σ ask_vol)

    Returns a value in [-1, 1].
    If the DataFrame is empty (no depth) we return 0.0 (neutral).
    """
    if dom_df.empty:
        return 0.0

    bid = dom_df["bid_volume"].sum()
    ask = dom_df["ask_volume"].sum()
    total = bid + ask
    if total == 0:
        return 0.0
    lir = (bid - ask) / total
    logger.debug("LIR computed: bid=%.0f ask=%.0f → %.4f", bid, ask, lir)
    return lir
