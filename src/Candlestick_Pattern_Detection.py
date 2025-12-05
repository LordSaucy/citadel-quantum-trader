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

# ----------------------------------------------------------------------
# Local utilities
# ----------------------------------------------------------------------
from src.utils.common import utc_now  # centralised UTC helper

# ----------------------------------------------------------------------
# Module logger (independent from the global app logger)
# ----------------------------------------------------------------------
candlestick_logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# USER‑DEFINED WEIGHTS (overrideable at runtime via ConfluenceController)
# ----------------------------------------------------------------------
PATTERN_WEIGHTS: Dict[str, float] = {
    # ⭐⭐⭐⭐⭐ Your favourites
    "morning_star": 0.20,
    "evening_star": 0.15,

    # ⭐⭐⭐⭐ Very good
    "morning_doji_star": 0.15,
    "evening_doji_star": 0.15,
    "bullish_engulfing": 0.15,
    "bearish_engulfing": 0.15,

    # ⭐⭐⭐ Good
    "hammer": 0.10,
    "inverted_hammer": 0.10,
    "hanging_man": 0.10,
    "shooting_star": 0.10,
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
    if not rates:
        candlestick_logger.error(
            f"Failed to fetch rates for {symbol} (tf={timeframe}, count={count})"
        )
        return None
    return rates


# ----------------------------------------------------------------------
# Geometry helpers (pure functions – easy to unit‑test)
# ----------------------------------------------------------------------
def _body(candle: Dict) -> float:
    """Absolute body size."""
    return abs(candle["close"] - candle["open"])


def _range(candle: Dict) -> float:
    """Full high‑low range."""
    return candle["high"] - candle["low"]


def _midpoint(candle: Dict) -> float:
    """Mid‑point of the candle (average of open & close)."""
    return (candle["open"] + candle["close"]) / 2


# ----------------------------------------------------------------------
# Individual pattern detectors (pure functions – easier to unit‑test)
# ----------------------------------------------------------------------
def detect_morning_star(c1: Dict, c2: Dict, c3: Dict) -> bool:
    """Morning Star – bullish reversal (3‑candle pattern)."""
    # Candle 1 – large bearish
    bearish_1 = c1["close"] < c1["open"]
    large_1 = _body(c1) > 0.6 * _range(c1)

    # Candle 2 – small body (doji‑like)
    small_body = _body(c2) < 0.3 * _range(c2)

    # Candle 3 – large bullish
    bullish_3 = c3["close"] > c3["open"]
    large_3 = _body(c3) > 0.6 * _range(c3)

    # Close above midpoint of candle 1
    closes_above_mid = c3["close"] > _midpoint(c1)

    return all([bearish_1, large_1, small_body, bullish_3, large_3, closes_above_mid])


def detect_evening_star(c1: Dict, c2: Dict, c3: Dict) -> bool:
    """Evening Star – bearish reversal (mirror of Morning Star)."""
    bullish_1 = c1["close"] > c1["open"]
    large_1 = _body(c1) > 0.6 * _range(c1)

    small_body = _body(c2) < 0.3 * _range(c2)

    bearish_3 = c3["close"] < c3["open"]
    large_3 = _body(c3) > 0.6 * _range(c3)

    closes_below_mid = c3["close"] < _midpoint(c1)

    return all([bullish_1, large_1, small_body, bearish_3, large_3, closes_below_mid])


def detect_morning_doji_star(c1: Dict, c2: Dict, c3: Dict) -> bool:
    """Morning Doji Star – same as Morning Star but middle candle is a Doji."""
    bearish_1 = c1["close"] < c1["open"]
    large_1 = _body(c1) > 0.6 * _range(c1)

    # Doji: body < 10 % of range
    is_doji = _body(c2) < 0.10 * _range(c2)

    bullish_3 = c3["close"] > c3["open"]
    large_3 = _body(c3) > 0.6 * _range(c3)

    closes_above_mid = c3["close"] > _midpoint(c1)

    return all([bearish_1, large_1, is_doji, bullish_3, large_3, closes_above_mid])


def detect_evening_doji_star(c1: Dict, c2: Dict, c3: Dict) -> bool:
    """Evening Doji Star – mirror of Morning Doji Star."""
    bullish_1 = c1["close"] > c1["open"]
    large_1 = _body(c1) > 0.6 * _range(c1)

    is_doji = _body(c2) < 0.10 * _range(c2)

    bearish_3 = c3["close"] < c3["open"]
    large_3 = _body(c3) > 0.6 * _range(c3)

    closes_below_mid = c3["close"] < _midpoint(c1)

    return all([bullish_1, large_1, is_doji, bearish_3, large_3, closes_below_mid])


def detect_engulfing_multi_candle(
    rates: np.ndarray,
    idx: int,
    direction: str = "BULLISH",
    min_candles: int = 2,
) -> bool:
    """
    Engulfing pattern that must engulf at least ``min_candles`` previous candles.

    Args:
        rates: NumPy structured array of OHLCV bars (chronological order).
        idx: Index of the *current* candle to evaluate.
        direction: ``"BULLISH"`` or ``"BEARISH"``.
        min_candles: Minimum number of previous candles that must be engulfed.

    Returns:
        ``True`` if the condition is satisfied.
    """
    cur = rates[idx]

    if direction.upper() == "BULLISH":
        # Current candle must be bullish
        if cur["close"] <= cur["open"]:
            return False
        # Check how many previous candles are fully engulfed
        engulfed = sum(
            1
            for i in range(1, min_candles + 1)
            if idx - i >= 0
            and cur["open"] <= rates[idx - i]["close"]
            and cur["close"] >= rates[idx - i]["open"]
        )
        return engulfed >= min_candles

    # ---- BEARISH branch -------------------------------------------------
    if cur["close"] >= cur["open"]:
        return False
    engulfed = sum(
        1
        for i in range(1, min_candles + 1)
        if idx - i >= 0
        and cur["open"] >= rates[idx - i]["close"]
        and cur["close"] <= rates[idx - i]["open"]
    )
    return engulfed >= min_candles


def detect_hammer(candle: Dict) -> bool:
    """Hammer – small body, long lower shadow, short upper shadow."""
    body = _body(candle)
    if body == 0:
        return False
    lower_shadow = candle["open"] - candle["low"] if candle["close"] >= candle["open"] else candle["close"] - candle["low"]
    upper_shadow = candle["high"] - candle["close"] if candle["close"] >= candle["open"] else candle["high"] - candle["open"]
    return lower_shadow >= 2 * body and upper_shadow <= 0.1 * body


def detect_inverted_hammer(candle: Dict) -> bool:
    """Inverted Hammer – small body, long upper shadow, short lower shadow."""
    body = _body(candle)
    if body == 0:
        return False
    upper_shadow = candle["high"] - candle["close"] if candle["close"] >= candle["open"] else candle["high"] - candle["open"]
    lower_shadow = candle["open"] - candle["low"] if candle["close"] >= candle["open"] else candle["close"] - candle["low"]
    return upper_shadow >= 2 * body and lower_shadow <= 0.1 * body


def detect_hanging_man(candle: Dict) -> bool:
    """Hanging Man – geometrically identical to Hammer (trend check is external)."""
    return detect_hammer(candle)


def detect_shooting_star(candle: Dict) -> bool:
    """Shooting Star – geometrically identical to Inverted Hammer (trend check is external)."""
    return detect_inverted_hammer(candle)


# ----------------------------------------------------------------------
# Main orchestrator class
# ----------------------------------------------------------------------
class CandlestickPatternDetector:
    """
    Detects candlestick patterns according to the user‑defined preferences
    (weights, removed patterns, multi‑candle engulfing, …).

    Public method ``detect_patterns(symbol)`` returns a dictionary consumable by
    the rest of the system:

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
            timeframe: MT5 timeframe to analyse (default H1).
            lookback: Number of candles to fetch (must be ≥ 3).
        """
        self.timeframe = timeframe
        self.lookback = lookback
        candlestick_logger.info(
            f"CandlestickPatternDetector initialised (tf={timeframe}, lookback={lookback})"
        )

    # ------------------------------------------------------------------
    # Private helpers – keep the public API tiny and testable
    # ------------------------------------------------------------------
    def _collect_raw_detections(self, rates: np.ndarray) -> List[Tuple[str, bool, str]]:
        """
        Run every low‑level detector and return a list of tuples:

        (pattern_name, is_detected, human_readable_description)
        """
        detections: List[Tuple[str, bool, str]] = []

        # ---- 3‑candle patterns (need at least 3 bars) -----------------
        if len(rates) >= 3:
            c1, c2, c3 = rates[-3], rates[-2], rates[-1]
            detections.extend(
                [
                    ("morning_star", detect_morning_star(c1, c2, c3), "Morning Star"),
                    ("evening_star", detect_evening_star(c1, c2, c3), "Evening Star"),
                    ("morning_doji_star", detect_morning_doji_star(c1, c2, c3), "Morning Doji Star"),
                    ("evening_doji_star", detect_evening_doji_star(c1, c2, c3), "Evening Doji Star"),
                ]
            )

        # ---- Multi‑candle engulfing (must engulf ≥2 previous candles) --
        for idx in range(len(rates)):
            if detect_engulfing_multi_candle(rates, idx, direction="BULLISH"):
                detections.append(("bullish_engulfing", True, "Bullish Engulfing (≥2 candles)"))
                        # ---- Bearish engulfing (same loop, opposite direction) ----
                if detect_engulfing_multi_candle(rates, idx, direction="BEARISH"):
                    detections.append(
                        ("bearish_engulfing", True, "Bearish Engulfing (≥2 candles)")
                    )

        # ---- Single‑candle patterns (evaluate only the most recent bar) ----
        last = rates[-1]
        detections.extend(
            [
                ("hammer", detect_hammer(last), "Hammer"),
                ("inverted_hammer", detect_inverted_hammer(last), "Inverted Hammer"),
                ("hanging_man", detect_hanging_man(last), "Hanging Man"),
                ("shooting_star", detect_shooting_star(last), "Shooting Star"),
            ]
        )
        return detections

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect_patterns(self, symbol: str) -> Dict:
        """
        Scan the most recent bars for the *best* pattern according to the
        weighted scheme.

        Args:
            symbol: Trading symbol (e.g. "EURUSD").

        Returns:
            dict with keys:
                - pattern_found (bool)
                - pattern_name (str | None)
                - pattern_score (float, 0‑100)
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
        # 1️⃣  Gather raw detections
        # ------------------------------------------------------------------
        raw_detections = self._collect_raw_detections(rates)

        # ------------------------------------------------------------------
        # 2️⃣  Compute weighted score for each *detected* pattern
        # ------------------------------------------------------------------
        best: Optional[Tuple[str, float, str]] = None  # (name, weighted_score, description)

        for name, found, description in raw_detections:
            if not found:
                continue

            # Weight expressed as a fraction (0 … 1).  Zero weight means the pattern
            # is intentionally ignored (e.g. three‑soldiers).
            weight = PATTERN_WEIGHTS.get(name, 0.0)
            if weight <= 0.0:
                continue

            # Convert to a 0‑100 scale for downstream consumption.
            weighted_score = weight * 100.0

            # Keep the pattern with the highest weighted contribution.
            if best is None or weighted_score > best[1]:
                best = (name, weighted_score, description)

        # ------------------------------------------------------------------
        # 3️⃣  Build the final result dictionary
        # ------------------------------------------------------------------
        if best is None:
            return {
                "pattern_found": False,
                "pattern_name": None,
                "pattern_score": 0.0,
                "pattern_description": "No qualifying candlestick pattern detected",
            }

        pattern_name, pattern_score, pattern_desc = best
        return {
            "pattern_found": True,
            "pattern_name": pattern_name,
            "pattern_score": round(pattern_score, 2),  # keep two decimals
            "pattern_description": pattern_desc,
        }


# ----------------------------------------------------------------------
# Global singleton – import this from `src/main.py` and use it directly
# ----------------------------------------------------------------------
candlestick_pattern_detector = CandlestickPatternDetector()
