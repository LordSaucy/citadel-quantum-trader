#!/usr/bin/env python3
"""
SHOCK DETECTOR – Production‑grade guard against adverse market conditions.

Monitors 5 independent shock vectors:
  1. News sentiment (score outside bounds)
  2. Macro‑calendar impact (high‑impact events)
  3. Volatility spikes (ATR relative to EMA)
  4. Sudden price jumps (% move in short window)
  5. Edge decay (win‑rate floor breach)

Returns (bool, str) where bool = blocked, str = reason for blocking.

✅ FIXED: Removed unused symbol parameter from _check_news_sentiment and _check_winrate_floor
"""

import time
import redis
import math
from datetime import datetime, timedelta
from config_loader import Config
from prometheus_client import Counter

cfg = Config().settings
r = redis.StrictRedis(
    host=cfg.get("redis_host", "redis"),
    port=cfg.get("redis_port", 6379),
    db=cfg.get("redis_db", 0),
    decode_responses=True
)

# =====================================================================
# Prometheus counters for audit / alerting
# =====================================================================
trade_blocked_total = Counter(
    "trade_blocked_total",
    "Number of trades blocked by shock‑detector",
    ["reason"]
)


# =====================================================================
# ✅ FIXED: Removed unused parameters from check functions
# =====================================================================

def _check_news_sentiment() -> tuple[bool, str]:
    """
    ✅ FIXED: Removed unused symbol parameter.
    
    Check if sentiment is within acceptable bounds.
    (Symbol not used – sentiment is global market sentiment, not symbol‑specific)
    """
    sentiment = r.get("sentiment:latest")
    if not sentiment:
        return False, ""

    try:
        s = float(sentiment)
        low = cfg["risk_shocks"]["sentiment"]["low"]
        high = cfg["risk_shocks"]["sentiment"]["high"]
        
        if s < low or s > high:
            trade_blocked_total.labels(reason="news_sentiment").inc()
            return True, "news_sentiment"
    except ValueError:
        pass  # Malformed value – ignore

    return False, ""


def _check_macro_calendar(symbol: str) -> tuple[bool, str]:
    """Check for macro calendar events blocking trades (symbol‑specific)."""
    macro_flag = r.get(f"macro:block:{symbol}")
    if macro_flag == "1":
        trade_blocked_total.labels(reason="macro_event").inc()
        return True, "macro_event"
    
    return False, ""


def _check_volatility_spike(symbol: str) -> tuple[bool, str]:
    """Check for volatility spikes (ATR vs EMA_ATR) (symbol‑specific)."""
    try:
        atr = float(r.get(f"atr:{symbol}") or 0)
        ema_atr = float(r.get(f"atr_ema30:{symbol}") or 0)
        mult = cfg["risk_shocks"]["atr_spike_multiplier"]
        
        if ema_atr > 0 and atr > mult * ema_atr:
            trade_blocked_total.labels(reason="vol_spike").inc()
            return True, "vol_spike"
    except (ValueError, TypeError):
        pass

    return False, ""


def _check_price_jump(symbol: str) -> tuple[bool, str]:
    """Check for sudden price jumps within short time window (symbol‑specific)."""
    try:
        now = time.time()
        win_secs = cfg["risk_shocks"]["price_jump"]["window_minutes"] * 60
        hist_key = f"price_hist:{symbol}"
        
        # Get oldest price in the window
        old = r.zrangebyscore(
            hist_key,
            now - win_secs,
            now - win_secs,
            start=0,
            num=1,
            withscores=False
        )
        latest = r.get(f"price:{symbol}")
        
        if old and latest:
            old_price = float(old[0])
            latest_price = float(latest)
            pct_move = abs(latest_price - old_price) / old_price * 100
            
            if pct_move > cfg["risk_shocks"]["price_jump"]["threshold_pct"]:
                trade_blocked_total.labels(reason="price_jump").inc()
                return True, "price_jump"
    except Exception:
        pass  # Any parsing issue → ignore

    return False, ""


def _check_winrate_floor() -> tuple[bool, str]:
    """
    ✅ FIXED: Removed unused symbol parameter.
    
    Check if win‑rate has fallen below minimum floor.
    (Symbol not used – win‑rate is global portfolio metric, not symbol‑specific)
    """
    winrate = r.get("edge:winrate")
    if winrate and float(winrate) < cfg["risk_shocks"]["winrate_floor"]:
        trade_blocked_total.labels(reason="winrate_floor").inc()
        return True, "winrate_floor"
    
    return False, ""


# =====================================================================
# Core guard – returns (blocked:bool, reason:str)
# =====================================================================

def should_block_trade(symbol: str) -> tuple[bool, str]:
    """
    Run all shock‑detector checks. Return True + reason if trade must be aborted.

    ✅ FIXED: Updated function calls to match new signatures.
    """
    # Run all checks in sequence
    checks = [
        _check_news_sentiment(),           # ✅ FIXED: No symbol argument
        _check_macro_calendar(symbol),
        _check_volatility_spike(symbol),
        _check_price_jump(symbol),
        _check_winrate_floor(),            # ✅ FIXED: No symbol argument
    ]

    # Return on first blocking condition
    for blocked, reason in checks:
        if blocked:
            return blocked, reason

    # Nothing triggered → allow the trade
    return False, ""


# =====================================================================
# Optional: Helper to run all checks and return detailed report
# =====================================================================

def analyze_trade_shocks(symbol: str) -> dict:
    """
    Run all shock checks and return a detailed dict showing all results.
    
    Useful for debugging or logging what was checked.
    """
    return {
        "news_sentiment": _check_news_sentiment(),
        "macro_calendar": _check_macro_calendar(symbol),
        "volatility_spike": _check_volatility_spike(symbol),
        "price_jump": _check_price_jump(symbol),
        "winrate_floor": _check_winrate_floor(),
    }
