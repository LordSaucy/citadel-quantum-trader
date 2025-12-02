# src/guard_helpers.py
import time
import numpy as np
from datetime import datetime, timedelta
from .metrics import (
    trade_skipped_total,
    guard_latency_seconds,
    guard_spread_pips,
    guard_atr,
    guard_lir,
)

# -----------------------------------------------------------------
# 2.1 Depth / LIR guard
# -----------------------------------------------------------------
def check_depth(broker, symbol: str, required_volume: float, min_lir: float = 0.5) -> bool:
    """
    Returns True if depth is sufficient, False otherwise.
    LIR = (bid_vol - ask_vol) / (bid_vol + ask_vol)
    """
    depth = broker.get_market_depth(symbol, depth=20)  # list of dicts
    if not depth:
        trade_skipped_total.labels(reason="depth").inc()
        return False

    # Aggregate bid/ask volumes
    bid_vol = sum(d["bid_volume"] for d in depth if d["side"] == "bid")
    ask_vol = sum(d["ask_volume"] for d in depth if d["side"] == "ask")

    # Simple LIR calculation (same as the one used in the research docs)
    total = bid_vol + ask_vol
    lir = (bid_vol - ask_vol) / total if total else 0.0
    guard_lir.set(lir)

    # Require at least `min_lir` * required_volume on the side we need
    if lir < min_lir:
        trade_skipped_total.labels(reason="depth").inc()
        return False

    # Also ensure the side we want has enough absolute volume
    side_needed = "bid" if required_volume > 0 else "ask"
    side_vol = bid_vol if side_needed == "bid" else ask_vol
    if side_vol < abs(required_volume):
        trade_skipped_total.labels(reason="depth").inc()
        return False

    return True


# -----------------------------------------------------------------
# 2.2 Latency guard
# -----------------------------------------------------------------
def check_latency(broker, max_latency_sec: float = 0.15) -> bool:
    """
    Measures the round‑trip time of a cheap “heartbeat” RPC.
    Returns True if latency <= max_latency_sec.
    """
    start = time.time()
    # Most broker adapters expose a cheap ping / time‑sync call
    broker.ping()                     # raise if fails
    elapsed = time.time() - start
    guard_latency_seconds.set(elapsed)

    if elapsed > max_latency_sec:
        trade_skipped_total.labels(reason="latency").inc()
        return False
    return True


# -----------------------------------------------------------------
# 2.3 Spread / slippage guard
# -----------------------------------------------------------------
def check_spread(broker, symbol: str, max_spread_pips: float = 0.5) -> bool:
    """
    Uses the broker’s market data to compute the current spread.
    Spread = ask – bid (in pips).  Returns False if spread > max.
    """
    quote = broker.get_quote(symbol)   # {'bid': 1.2345, 'ask': 1.2350}
    if not quote:
        trade_skipped_total.labels(reason="spread").inc()
        return False

    spread = (quote["ask"] - quote["bid"]) * 10_000   # 1 pip = 0.0001 for FX
    guard_spread_pips.set(spread)

    if spread > max_spread_pips:
        trade_skipped_total.labels(reason="spread").inc()
        return False
    return True


# -----------------------------------------------------------------
# 2.4 Volatility‑spike guard (ATR‑based)
# -----------------------------------------------------------------
def check_volatility(atc, symbol: str,
                     atr_multiplier: float = 2.0,
                     max_atr_pct: float = 0.20) -> bool:
    """
    `atc` = AdvancedTechnicalCalculator (or any helper that can compute ATR).
    Returns False if the latest ATR is > max_atr_pct * recent average.
    """
    # Get the most recent ATR (could be a method on the bot)
    recent_atr = atc.get_atr(symbol, lookback=14)          # absolute price units
    avg_atr = atc.get_atr_average(symbol, lookback=100)   # same units
    if recent_atr is None or avg_atr is None:
        trade_skipped_total.labels(reason="volatility").inc()
        return False

    # Normalise to a percentage of the price (optional)
    price = atc.get_price(symbol)  # current mid price
    recent_pct = recent_atr / price
    avg_pct = avg_atr / price
    guard_atr.set(recent_pct)

    # If the recent ATR spikes more than `max_atr_pct` above the average → block
    if recent_pct > (avg_pct * (1.0 + max_atr_pct)):
        trade_skipped_total.labels(reason="volatility").inc()
        return False
    return True
