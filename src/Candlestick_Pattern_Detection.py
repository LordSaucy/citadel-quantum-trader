#!/usr/bin/env python3
"""
Candlestick Pattern Detection with User Preferences

Prioritises Morning‑Star, removes Three Soldiers / Crows and other
patterns that you consider “waste of time”.

Author: Lawful Banker
Created: 2024‑11‑26
Version: 2.0 – Production‑grade implementation
"""

# ----------------------------------------------------------------------
# Standard library
# ----------------------------------------------------------------------
import logging
from typing import Dict, List, Tuple, Optional

# ----------------------------------------------------------------------
# Third‑party
# ----------------------------------------------------------------------
import MetaTrader5 as mt5
import numpy as np
from datetime import datetime

# ----------------------------------------------------------------------
# Logging – separate logger so you can control its level independently
# ----------------------------------------------------------------------
candlestick_logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# USER‑DEFINED WEIGHTS (these can be overridden at runtime via the
# ConfluenceController – see comments in the code)
# ----------------------------------------------------------------------
PATTERN_WEIGHTS: Dict[str, float] = {
    # ⭐⭐⭐⭐⭐ Your favourites
    "morning_star": 0.20,          # Most preferred
    "evening_star": 0.15,

    # ⭐⭐⭐⭐ Very good
    "morning_doji_star": 0.15,
    "evening_doji_star": 0.15,
    "bullish_engulfing": 0.15,     # Must engulf ≥2 candles
    "bearish_engulfing": 0.15,

    # ⭐⭐⭐ Good
    "hammer": 0.10,
    "inverted_hammer": 0.10,
    "hanging_man": 0.10,
    "shooting_star": 0.10,

    # ❌ Removed (complete waste of time)
    # "three_white_soldiers": 0.00,
    # "three_black_crows": 0.00,
    # "piercing_line": 0.00,
    # "dark_cloud_cover": 0.00,
}


# ----------------------------------------------------------------------
# Helper – fetch recent OHLCV bars from MT5
# ----------------------------------------------------------------------
def _fetch_rates(
    symbol: str,
    timeframe: int = mt5.TIMEFRAME_H1,
    count: int = 250,
) -> Optional[np.ndarray]:
    """
    Pull ``count`` bars for ``symbol`` on ``timeframe``.
    Returns a NumPy structured array or ``None`` on failure.
    """
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None or len(rates) == 0:
        candlestick_logger.error(
            f"Failed to fetch rates for {symbol} (tf={timeframe}, count={count})"
        )
        return None
    return rates


# ----------------------------------------------------------------------
# Individual pattern detectors (pure functions – easier to unit‑test)
# ----------------------------------------------------------------------
def detect_morning_star(c1: Dict, c2: Dict, c3: Dict) -> bool:
    """
    Morning Star (your favourite)

    1️⃣ Large bearish candle
    2️⃣ Small‑body “star” (can be bullish or bearish)
    3️⃣ Large bullish candle closing above the midpoint of candle 1
    """
    # Candle 1 – large bearish
    bearish_1 = c1["close"] < c1["open"]
    large_1 = abs(c1["close"] - c1["open"]) > (c1["high"] - c1["low"]) * 0.6

    # Candle 2 – small body
    small_body = abs(c2["close"] - c2["open"]) < (c2["high"] - c2["low"]) * 0.3

    # Candle 3 – large bullish
    bullish_3 = c3["close"] > c3["open"]
    large_3 = abs(c3["close"] - c3["open"]) > (c3["high"] - c3["low"]) * 0.6

    # Close above midpoint of candle 1
    closes_above_mid = c3["close"] > (c1["open"] + c1["close"]) / 2

    return (
        bearish_1
        and large_1
        and small_body
        and bullish_3
        and large_3
        and closes_above_mid
    )


def detect_evening_star(c1: Dict, c2: Dict, c3: Dict) -> bool:
    """
    Evening Star – mirror image of Morning Star (bearish reversal)
    """
    # Candle 1 – large bullish
    bullish_1 = c1["close"] > c1["open"]
    large_1 = abs(c1["close"] - c1["open"]) > (c1["high"] - c1["low"]) * 0.6

    # Candle 2 – small body
    small_body = abs(c2["close"] - c2["open"]) < (c2["high"] - c2["low"]) * 0.3

    # Candle 3 – large bearish
    bearish_3 = c3["close"] < c3["open"]
    large_3 = abs(c3["close"] - c3["open"]) > (c3["high"] - c3["low"]) * 0.6

    # Close below midpoint of candle 1
    closes_below_mid = c3["close"] < (c1["open"] + c1["close"]) / 2

    return (
        bullish_1
        and large_1
        and small_body
        and bearish_3
        and large_3
        and closes_below_mid
    )


def detect_morning_doji_star(c1: Dict, c2: Dict, c3: Dict) -> bool:
    """
    Morning Doji Star – same structure as Morning Star but the middle
    candle is a Doji (very small body).
    """
    # Candle 1 – large bearish
    bearish_1 = c1["close"] < c1["open"]
    large_1 = abs(c1["close"] - c1["open"]) > (c1["high"] - c1["low"]) * 0.6

    # Candle 2 – Doji (body < 10 % of range)
    body = abs(c2["close"] - c2["open"])
    rng = c2["high"] - c2["low"]
    is_doji = body < rng * 0.10

    # Candle 3 – large bullish
    bullish_3 = c3["close"] > c3["open"]
    large_3 = abs(c3["close"] - c3["open"]) > (c3["high"] - c3["low"]) * 0.6

    # Close above midpoint of candle 1
    closes_above_mid = c3["close"] > (c1["open"] + c1["close"]) / 2

    return (
        bearish_1
        and large_1
        and is_doji
        and bullish_3
        and large_3
        and closes_above_mid
    )


def detect_evening_doji_star(c1: Dict, c2: Dict, c3: Dict) -> bool:
    """
    Evening Doji Star – mirror image of Morning Doji Star.
    """
    # Candle 1 – large bullish
    bullish_1 = c1["close"] > c1["open"]
    large_1 = abs(c1["close"] - c1["open"]) > (c1["high"] - c1["low"]) * 0.6

    # Candle 2 – Doji
    body = abs(c2["close"] - c2["open"])
    rng = c2["high"] - c2["low"]
    is_doji = body < rng * 0.10

    # Candle 3 – large bearish
    bearish_3 = c3["close"] < c3["open"]
    large_3 = abs(c3["close"] - c3["open"]) > (c3["high"] - c3["low"]) * 0.6

    # Close below midpoint of candle 1
    closes_below_mid = c3["close"] < (c1["open"] + c1["close"]) / 2

    return (
        bullish_1
        and large_1
        and is_doji
        and bearish_3
        and large_3
        and closes_below_mid
    )


def detect_engulfing_multi_candle(
    rates: np.ndarray,
    idx: int,
    direction: str = "BULLISH",
    min_candles: int = 2,
) -> bool:
    """
    Engulfing pattern that must engulf at least ``min_candles`` previous
    candles (your explicit requirement).

    Returns ``True`` if the condition is satisfied.
    """
    cur = rates[idx]

    if direction.upper() == "BULLISH":
        # Current candle must be bullish
        if cur["close"] <= cur["open"]:
            return False

        engulfed = 0
        for i in range(1, 5):          # look back up to 4 candles
            if idx - i < 0:
                break
            prev = rates[idx - i]
            if cur["open"] <= prev["close"] and cur["close"] >= prev["open"]:
                engulfed += 1
            else:
                break
        return engulfed >= min_candles

    else:  # BEARISH
        if cur["close"] >= cur["open"]:
            return False

        engulfed = 0
        for i in range(1, 5):
            if idx - i < 0:
                break
            prev = rates[idx - i]
            if cur["open"] >= prev["close"] and cur["close"] <= prev["open"]:
                engulfed += 1
            else:
                break
        return engulfed >= min_candles


def detect_hammer(candle: Dict) -> bool:
    """
    Hammer (bullish reversal) – small body, long lower shadow,
    upper shadow ≤ 0.1 × body.
    """
    body = abs(candle["close"] - candle["open"])
    lower_shadow = candle["open"] - candle["low"] if candle["close"] >= candle["open"] else candle["close"] - candle["low"]
    upper_shadow = candle["high"] - candle["close"] if candle["close"] >= candle["open"] else candle["high"] - candle["open"]

    if body == 0:
        return False
    return lower_shadow >= 2 * body and upper_shadow <= 0.1 * body


def detect_inverted_hammer(candle: Dict) -> bool:
    """
    Inverted Hammer – small body, long upper shadow,
    lower shadow ≤ 0.1 × body.
    """
    body = abs(candle["close"] - candle["open"])
    upper_shadow = candle["high"] - candle["close"] if candle["close"] >= candle["open"] else candle["high"] - candle["open"]
    lower_shadow = candle["open"] - candle["low"] if candle["close"] >= candle["open"] else candle["close"] - candle["low"]

    if body == 0:
        return False
    return upper_shadow >= 2 * body and lower_shadow <= 0.1 * body


def detect_hanging_man(candle: Dict) -> bool:
    """
    Hanging Man – same geometry as Hammer but appears in an up‑trend.
    (We only check geometry here; trend detection is left to the caller.)
    """
    return detect_hammer(candle)


def detect_shooting_star(candle: Dict) -> bool:
    """
    Shooting Star – same geometry as Inverted Hammer but appears in a down‑trend.
    (Again, only geometry is checked here.)
    """
    return detect_inverted_hammer(candle)


# ----------------------------------------------------------------------
# Main orchestrator class
# ----------------------------------------------------------------------
class CandlestickPatternDetector:
    """
    Detects candlestick patterns according to the user‑defined
    preferences (weights, removed patterns, multi‑candle engulfing, …).

    The public method ``detect_patterns(symbol, direction)`` returns a
    dictionary that the rest of the system can consume:

    {
        "pattern_found": bool,
        "pattern_name": str | None,
        "pattern_score": float,          # 0‑100 weighted score
        "pattern_description": str | None,
    }
    """

    def __init__(self, timeframe: int = mt5.TIMEFRAME_H1, lookback: int = 250):
        """
        Args:
            timeframe: MT5 timeframe to analyse (default H1)
            lookback: Number of candles to fetch (must be ≥ 3)
        """
        self.timeframe = timeframe
        self.lookback = lookback
        candlestick_logger.info(
            f"CandlestickPatternDetector initialised (tf={timeframe}, lookback={lookback})"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect_patterns(self, symbol: str, direction: str) -> Dict:
        """
        Scan the most recent bars for the *best* pattern according to the
        weighted scheme.

        Args:
            symbol: Trading symbol (e.g. "EURUSD")
            direction: "BUY" or "SELL" – used for engulfing orientation

        Returns:
            dict with keys:
                - pattern_found (bool)
                - pattern_name (str | None)
                - pattern_score (float, 0‑100 weighted)
                - pattern_description (str | None)
        """
        rates = _fetch_rates(symbol, self.timeframe, self.lookback)
        if rates is None:
            return {
                "pattern_found": False,
                "pattern_name": None,
                "pattern_score": 0.0,
                "pattern_description": "Failed to fetch market data",
            }

        # ------------------------------------------------------------------
        # 1️⃣  Gather raw detections (True/False) for every pattern
        # ------------------------------------------------------------------
        detections: List[Tuple[str, bool, str]] = []   # (name, found?, description)

        # --- Morning / Evening Star (3‑candle patterns) ---
        if len(rates) >= 3:
            c1, c2, c3 = rates[-3], rates[-2], rates[-1]
            detections.append(
                ("morning_star", detect_morning_star(c1, c2, c3), "Morning Star")
            )
            detections.append(
                ("evening_star", detect_evening_star(c1, c2, c3), "Evening Star")
            )
            detections.append(
                ("morning_doji_star", detect_morning_doji_star(c1, c2, c3), "Morning Doji Star")
            )
            detections.append(
                ("evening_doji_star", detect_evening_doji_star(c1, c2, c3), "Evening Doji Star")
            )

        # --- Multi‑candle engulfing (requires at least 2 previous candles) ---
        for idx in range(len(rates)):
            if detect_engulfing_multi_candle(rates, idx, direction="BULLISH"):
                detections.append(("bullish_engulfing", True, "Bullish Engulfing (≥2 candles)"))
            if detect_engulfing_multi_candle(rates, idx, direction="BEARISH"):
                detections.append(("bearish_engulfing", True, "Bearish Engulfing (≥2 candles)"))

        # --- Single‑candle patterns (last candle only) ---
        last = rates[-1]
        detections.append(("hammer", detect_hammer(last), "Hammer"))
        detections.append(("inverted_hammer", detect_inverted_hammer(last), "Inverted Hammer"))
        detections.append(("hanging_man", detect_hanging_man(last), "Hanging Man"))
        detections.append(("shooting_star", detect_shooting_star(last), "Shooting Star"))

            # ------------------------------------------------------------------
        # 2️⃣  Compute weighted score for each *found* pattern
        # ------------------------------------------------------------------
        for name, found, description in detections:
            if not found:
                continue

            # The weight dictionary expresses the *relative importance* of each
            # pattern (0 … 1).  A weight of 0 means the pattern is deliberately
            # ignored (e.g. “three white soldiers”).  The final contribution
            # to the score is therefore:  weight × 100.
            raw_weight = PATTERN_WEIGHTS.get(name, 0.0)

            # Guard against accidental zero‑weight patterns – they simply do not
            # affect the score.
            if raw_weight <= 0.0:
                continue

            weighted_score = raw_weight * 100.0

            # Keep the pattern with the *highest* weighted contribution.
            if best_pattern is None or weighted_score > best_pattern[1]:
                best_pattern = (name, weighted_score, description)

        # ------------------------------------------------------------------
        # 3️⃣  Build the final result dictionary
        # ------------------------------------------------------------------
        if best_pattern is None:
            # No pattern survived the weight filter – treat as “no signal”.
            return {
                "pattern_found": False,
                "pattern_name": None,
                "pattern_score": 0.0,
                "pattern_description": "No qualifying candlestick pattern detected",
            }

        pattern_name, pattern_score, pattern_desc = best_pattern

        return {
            "pattern_found": True,
            "pattern_name": pattern_name,
            "pattern_score": round(pattern_score, 2),   # 0‑100 scale
            "pattern_description": pattern_desc,
        }

# ----------------------------------------------------------------------
# Global singleton – import this from ``src/main.py`` and use it directly
# ----------------------------------------------------------------------
candlestick_pattern_detector = CandlestickPatternDetector()
