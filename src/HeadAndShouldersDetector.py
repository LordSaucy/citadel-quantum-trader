#!/usr/bin/env python3
"""
Head and Shoulders Pattern Detection

Rules (from the original specification):

1️⃣  Reversal pattern only (not continuation)  
2️⃣  Must break neckline to be valid  
3️⃣  Must wait for retest of neckline before entry  
4️⃣  Never trade before neckline break  
5️⃣  Stop‑loss placed below/above neckline

Pattern Structure:

- Left Shoulder : approximately equal high (or slightly higher) to previous high
- Head          : highest point (new higher high)
- Right Shoulder: approximately equal high to left shoulder (or slightly lower)
- Neckline      : drawn between the lows after the left shoulder and after the head
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
# head_and_shoulders_detector.py  (add at the top)
from src.ultimate_confluence_system import confluence_controller   # <-- singleton

import MetaTrader5 as mt5
import numpy as np

# ----------------------------------------------------------------------
# Logging – the main application configures the root logger;
# we just obtain a child logger here.
# ----------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass
class ShoulderPoint:
    """A shoulder or head point in the pattern."""
    price: float
    time: datetime
    index: int
    type: str  # "LEFT_SHOULDER", "HEAD", "RIGHT_SHOULDER"


@dataclass
class Neckline:
    """Neckline of a head‑and‑shoulders pattern."""
    price: float          # Average price of the neckline (used for SL)
    left_point: float     # Low (or high) after the left shoulder
    right_point: float    # Low (or high) after the head
    slope: float          # 0 for flat, otherwise the gradient


@dataclass
class HeadAndShouldersPattern:
    """Complete H&S pattern description."""
    pattern_type: str                     # "BEARISH_HS" or "BULLISH_INV_HS"
    left_shoulder: ShoulderPoint
    head: ShoulderPoint
    right_shoulder: ShoulderPoint
    neckline: Neckline
    neckline_broken: bool                 # Has price broken the neckline?
    neckline_retested: bool               # Has a retest occurred after the break?
    valid_for_entry: bool                 # Can we safely enter now?
    strength: float                       # 0‑1 based on symmetry & extension
    formation_time: datetime              # Time of the right‑shoulder candle


# ----------------------------------------------------------------------
# Detector class
# ----------------------------------------------------------------------
class HeadAndShouldersDetector:
    """
    Detects and validates Head‑and‑Shoulders patterns.

    Critical business rules (enforced by the code):
        • Pattern is only considered **after** the neckline has been broken.
        • Entry is allowed **only after** a successful neckline retest.
        • The pattern is a reversal (not a continuation).
    """

    # ----- Tunable thresholds ------------------------------------------------
# ------------------------------------------------------------------
# Inside the class – replace the class‑level constants with properties
# ------------------------------------------------------------------
@property
def SHOULDER_SYMMETRY_TOLERANCE(self) -> float:
    return confluence_controller.get("hs_shoulder_symmetry_tolerance")

@property
def HEAD_MINIMUM_EXTENSION(self) -> float:
    return confluence_controller.get("hs_head_minimum_extension")

@property
def NECKLINE_BREAK_CONFIRMATION(self) -> float:
    return confluence_controller.get("hs_neckline_break_confirmation")

@property
def RETEST_TOLERANCE(self) -> float:
    return confluence_controller.get("hs_retest_tolerance")

    # ------------------------------------------------------------------
    def __init__(self) -> None:
        """Create a fresh detector instance."""
        self.detected_patterns: List[HeadAndShouldersPattern] = []
        logger.info("Head‑and‑Shoulders Detector initialised")

    # ------------------------------------------------------------------
    # Public entry point ----------------------------------------------------
    def detect_pattern(
        self,
        symbol: str,
        direction: str,
        timeframe: int = mt5.TIMEFRAME_H4,
    ) -> Optional[HeadAndShouldersPattern]:
        """
        Detect a head‑and‑shoulders pattern for ``symbol`` in ``direction``.

        Args:
            symbol: MT5 symbol (e.g. "EURUSD").
            direction: "BUY"  → look for **inverted** H&S (bullish reversal)  
                       "SELL" → look for **regular** H&S (bearish reversal)
            timeframe: MT5 timeframe to analyse (default H4).

        Returns:
            A populated ``HeadAndShouldersPattern`` if a valid pattern is found,
            otherwise ``None``.
        """
        logger.info(f"Scanning for {direction} H&S pattern on {symbol}")

        # --------------------------------------------------------------
        # 1️⃣  Pull price data
        # --------------------------------------------------------------
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 500)
        if rates is None or len(rates) < 100:
            logger.warning("Insufficient data for H&S detection")
            return None

        # --------------------------------------------------------------
        # 2️⃣  Dispatch to the correct sub‑routine
        # --------------------------------------------------------------
        if direction.upper() == "SELL":
            pattern = self._detect_bearish_hs(rates, symbol, timeframe)
        elif direction.upper() == "BUY":
            pattern = self._detect_inverted_hs(rates, symbol, timeframe)
        else:
            logger.error(f"Invalid direction '{direction}' – must be BUY or SELL")
            return None

        # --------------------------------------------------------------
        # 3️⃣  Log details (if a pattern was found)
        # --------------------------------------------------------------
        if pattern:
            logger.info(f"✅ {pattern.pattern_type} detected!")
            logger.info(f"   Left  : {pattern.left_shoulder.price:.5f}")
            logger.info(f"   Head  : {pattern.head.price:.5f}")
            logger.info(f"   Right : {pattern.right_shoulder.price:.5f}")
            logger.info(f"   Neckline price : {pattern.neckline.price:.5f}")
            logger.info(f"   Neckline broken: {pattern.neckline_broken}")
            logger.info(f"   Neckline retested: {pattern.neckline_retested}")
            logger.info(f"   Valid for entry : {pattern.valid_for_entry}")

        return pattern

    # ------------------------------------------------------------------
    # 4️⃣  Bearish (regular) H&S – reversal at market tops
    # ------------------------------------------------------------------
    def _detect_bearish_hs(
        self,
        rates: np.ndarray,
        symbol: str,
        timeframe: int,
    ) -> Optional[HeadAndShouldersPattern]:
        """
        Detect a **bearish** head‑and‑shoulders pattern (sell signal).

        The algorithm:
            1. Locate swing highs (potential shoulders / head).
            2. Scan the most recent triples for a valid geometry.
            3. Verify head‑extension & shoulder symmetry.
            4. Derive the neckline from the lows after the left shoulder & head.
            5. Confirm neckline break, then look for a retest.
            6. Compute a strength metric (symmetry + extension).
        """
        swing_highs = self._find_swing_highs(rates, lookback=10)

        if len(swing_highs) < 3:
            return None

        # Examine the newest triples (right‑most first)
        for i in range(len(swing_highs) - 3, max(0, len(swing_highs) - 10), -1):
            left_idx = swing_highs[i]
            head_idx = swing_highs[i + 1]
            right_idx = swing_highs[i + 2]

            left_price = rates[left_idx]["high"]
            head_price = rates[head_idx]["high"]
            right_price = rates[right_idx]["high"]

            # ----- Structural rules -------------------------------------------------
            # 1️⃣ Head must be higher than both shoulders
            if not (head_price > left_price and head_price > right_price):
                continue

            # 2️⃣ Head must be at least 1 % above the shoulders
            if head_price < left_price * (1 + self.HEAD_MINIMUM_EXTENSION):
                continue

            # 3️⃣ Shoulders must be roughly equal (≤2 % diff)
            shoulder_diff = abs(left_price - right_price) / left_price
            if shoulder_diff > self.SHOULDER_SYMMETRY_TOLERANCE:
                continue

            # ----- Neckline construction -------------------------------------------
            neckline = self._calculate_neckline(
                rates, left_idx, head_idx, right_idx, pattern_type="BEARISH"
            )
            if neckline is None:
                continue

            # ----- Build the three shoulder objects ---------------------------------
            left_shoulder = ShoulderPoint(
                price=left_price,
                time=datetime.fromtimestamp(rates[left_idx]["time"]),
                index=left_idx,
                type="LEFT_SHOULDER",
            )
            head = ShoulderPoint(
                price=head_price,
                time=datetime.fromtimestamp(rates[head_idx]["time"]),
                index=head_idx,
                type="HEAD",
            )
            right_shoulder = ShoulderPoint(
                price=right_price,
                time=datetime.fromtimestamp(rates[right_idx]["time"]),
                index=right_idx,
                type="RIGHT_SHOULDER",
            )

            # ----- Neckline break detection ----------------------------------------
            current_price = rates[-1]["close"]
            neckline_broken = current_price < (
                neckline.price * (1 - self.NECKLINE_BREAK_CONFIRMATION)
            )

            # ----- Neckline retest & entry validation -------------------------------
            neckline_retested = False
            valid_for_entry = False
            if neckline_broken:
                neckline_retested, valid_for_entry = self._check_neckline_retest(
                    rates, right_idx, neckline.price, pattern_type="BEARISH"
                )

            # ----- Pattern strength -------------------------------------------------
            strength = self._calculate_pattern_strength(
                left_price, head_price, right_price, shoulder_diff
            )

            # ----- Assemble the final pattern object -------------------------------
            pattern = HeadAndShouldersPattern(
                pattern_type="BEARISH_HS",
                left_shoulder=left_shoulder,
                head=head,
                right_shoulder=right_shoulder,
                neckline=neckline,
                neckline_broken=neckline_broken,
                neckline_retested=neckline_retested,
                valid_for_entry=valid_for_entry,
                strength=strength,
                formation_time=datetime.fromtimestamp(rates[right_idx]["time"]),
            )
            return pattern

        return None

    # ------------------------------------------------------------------
    # 5️⃣  Inverted (bullish) H&S – reversal at market bottoms
    # ------------------------------------------------------------------
    def _detect_inverted_hs(
        self,
        rates: np.ndarray,
        symbol: str,
        timeframe: int,
    ) -> Optional[HeadAndShouldersPattern]:
        """
        Detect an **inverted** head‑and‑shoulders pattern (buy signal).

        The logic mirrors the bearish version but works on swing lows
        and expects the head to be *lower* than the shoulders.
        """
        swing_lows = self._find_swing_lows(rates, lookback=10)

        if len(swing_lows) < 3:
            return None

        for i in range(len(swing_lows) - 3, max(0, len(swing_lows) - 10), -1):
            left_idx = swing_lows[i]
            head_idx = swing_lows[i + 1]
            right_idx = swing_lows[i + 2]

            left_price = rates[left_idx]["low"]
            head_price = rates[head_idx]["low"]
            right_price = rates[right_idx]["low"]

            # ----- Structural rules -------------------------------------------------
            # 1️⃣ Head must be lower than both shoulders
            if not (head_price < left_price and head_price < right_price):
                continue

            # 2️⃣ Head must be at least 1 % lower than the shoulders
            if head_price > left_price * (1 - self.HEAD_MINIMUM_EXTENSION):
                continue

            # 3️⃣ Shoulders must be roughly equal (≤2 % diff)
            shoulder_diff = abs(left_price - right_price) / left_price
            if shoulder_diff > self.SHOULDER_SYMMETRY_TOLERANCE:
                continue

            # ----- Neckline construction -------------------------------------------
            neckline = self._calculate_neckline(
                rates, left_idx, head_idx, right_idx, pattern_type="BULLISH"
            )
            if neckline is None:
                continue

            # ----- Build shoulder objects -------------------------------------------
            left_shoulder = ShoulderPoint(
                price=left_price,
                time=datetime.fromtimestamp(rates[left_idx]["time"]),
                index=left_idx,
                type="LEFT_SHOULDER",
            )
            head = ShoulderPoint(
                price=head_price,
                time=datetime.fromtimestamp(rates[head_idx]["time"]),
                index=head_idx,
                type="HEAD",
            )
            right_shoulder = ShoulderPoint(
                price=right_price,
                time=datetime.fromtimestamp(rates[right_idx]["time"]),
                index=right_idx,
                type="RIGHT_SHOULDER",
            )

            # ----- Neckline break detection ----------------------------------------
            current_price = rates[-1]["close"]
            neckline_broken = current_price > (
                neckline.price * (1 + self.NECKLINE_BREAK_CONFIRMATION)
            )

            # ----- Neckline retest & entry validation -------------------------------
            neckline_retested = False
            valid_for_entry = False
            if neckline_broken:
                neckline_retested, valid_for_entry = self._check_neckline_retest(
                    rates, right_idx, neckline.price, pattern_type="BULLISH"
                )

            # ----- Pattern strength -------------------------------------------------
            strength = self._calculate_pattern_strength(
                left_price, head_price, right_price, shoulder_diff
            )

            # ----- Assemble the final pattern object -------------------------------
            pattern = HeadAndShouldersPattern(
                pattern_type="BULLISH_INV_HS",
                left_shoulder=left_shoulder,
                head=head,
                right_shoulder=right_shoulder,
                neckline=neckline,
                neckline_broken=neckline_broken,
                neckline_retested=neckline_retested,
                valid_for_entry=valid_for_entry,
                strength=strength,
                formation_time=datetime.fromtimestamp(rates[right_idx]["time"]),
            )
            return pattern

        return None

    # ------------------------------------------------------------------
    # 6️⃣  Helper – swing high / low detection
    # ------------------------------------------------------------------
    def _find_swing_highs(self, rates: np.ndarray, lookback: int = 10) -> List[int]:
        """Return indices of swing‑high candles."""
        swing_highs = []
        for i in range(lookback, len(rates) - lookback):
            window = rates[i - lookback : i + lookback + 1]
            if rates[i]["high"] == np.max(window["high"]):
                swing_highs.append(i)
        return swing_highs

    def _find_swing_lows(self, rates: np.ndarray, lookback: int = 10) -> List[int]:
        """Return indices of swing‑low candles."""
        swing_lows = []
        for i in range(lookback, len(rates) - lookback):
            window = rates[i - lookback : i + lookback + 1]
            if rates[i]["low"] == np.min(window["low"]):
                swing_lows.append(i)
        return swing_lows

    # ------------------------------------------------------------------
    # 7️⃣  Neckline calculation
    # ------------------------------------------------------------------
    def _calculate_neckline(
        self,
        rates: np.ndarray,
        left_idx: int,
        head_idx: int,
        right_idx: int,
        pattern_type: str,
    ) -> Optional[Neckline]:
        """
        Build the neckline.

        *Bearish*  → connect the **lowest** price after the left shoulder
                     and the lowest price after the head.

        *Bullish*  → connect the **highest** price after the left shoulder
                     and the highest price after the head.
        """
        if pattern_type == "BEARISH":
            left_low = self._find_low_between(rates, left_idx, head_idx)
            right_low = self._find_low_between(rates, head_idx, right_idx)
            if left_low is None or right_low is None:
                return None
            left_point = rates[left_low]["low"]
            right_point = rates[right_low]["low"]
        else:  # BULLISH
            left_high = self._find_high_between(rates, left_idx, head_idx)
            right_high = self._find_high_between(rates, head_idx, right_idx)
            if left_high is None or right_high is None:
                return None
            left_point = rates[left_high]["high"]
            right_point = rates[right_high]["high"]

        avg_price = (left_point + right_point) / 2.0
        slope = (right_point - left_point) / (right_idx - left_idx) if right_idx != left_idx else 0.0

        return Neckline(
            price=avg_price,
            left_point=left_point,
            right_point=right_point,
            slope=slope,
        )

    def _find_low_between(self, rates: np.ndarray, start_idx: int, end_idx: int) -> Optional[int]:
        """Lowest price between two indices (inclusive)."""
        if start_idx >= end_idx or end_idx >= len(rates):
            return None
        segment = rates[start_idx : end_idx + 1]
        if len(segment) == 0:
            return None
        min_idx = np.argmax(segment[“low”])
Return start_idx + max_idx

   def _find_high_between(self, rates: np.ndarray, start_idx: int, end_idx: int) -> Optional[int]:
        """Highest price between two indices (inclusive)."""
        if start_idx >= end_idx or end_idx >= len(rates):
            return None
        segment = rates[start_idx : end_idx + 1]
        if len(segment) == 0:
            return None
        max_idx = np.argmax(segment["high"])
        return start_idx + max_idx

    # ------------------------------------------------------------------
    # 8️⃣  Neckline retest detection
    # ------------------------------------------------------------------
    def _check_neckline_retest(
        self,
        rates: np.ndarray,
        right_shoulder_idx: int,
        neckline_price: float,
        pattern_type: str,
    ) -> Tuple[bool, bool]:
        """
        After the neckline has been broken, verify that price **re‑tests**
        the neckline and then rejects (closes on the opposite side).

        Returns:
            (retested, valid_for_entry)
        """
        # Look at candles *after* the right‑shoulder candle
        post_candles = rates[right_shoulder_idx + 1 :]

        if len(post_candles) < 3:
            return False, False

        retested = False
        valid_for_entry = False

        if pattern_type == "BEARISH":
            # Expect price to rise back up to the neckline from below
            for i, c in enumerate(post_candles):
                # Does the candle touch the neckline (within tolerance)?
                if abs(c["high"] - neckline_price) / neckline_price <= self.RETEST_TOLERANCE:
                    retested = True
                    # Did it close *below* the neckline afterwards? (rejection)
                    if c["close"] < neckline_price:
                        valid_for_entry = True
                    break
        else:  # BULLISH
            # Expect price to fall back down to the neckline from above
            for i, c in enumerate(post_candles):
                if abs(c["low"] - neckline_price) / neckline_price <= self.RETEST_TOLERANCE:
                    retested = True
                    if c["close"] > neckline_price:
                        valid_for_entry = True
                    break

        return retested, valid_for_entry

    # ------------------------------------------------------------------
    # 9️⃣  Pattern strength calculation (symmetry + head extension)
    # ------------------------------------------------------------------
    def _calculate_pattern_strength(
        self,
        left_price: float,
        head_price: float,
        right_price: float,
        shoulder_diff: float,
    ) -> float:
        """
        Produce a 0‑1 strength metric.

        *Symmetry*  – the closer the two shoulders, the higher the score.
        *Extension* – the farther the head is above (or below) the shoulders,
                      the higher the score, capped at 5 % for a perfect 1.0.
        """
        # ---- symmetry (0‑1) -------------------------------------------------
        symmetry = 1.0 - (shoulder_diff / self.SHOULDER_SYMMETRY_TOLERANCE)
        symmetry = max(0.0, min(1.0, symmetry))

        # ---- head extension -------------------------------------------------
        avg_shoulder = (left_price + right_price) / 2.0
        extension = abs(head_price - avg_shoulder) / avg_shoulder
        extension_score = min(1.0, extension / 0.05)   # 5 % => 1.0

        # ---- weighted blend -------------------------------------------------
        strength = symmetry * 0.6 + extension_score * 0.4
        return strength

    # ------------------------------------------------------------------
    # 10️⃣  Public helper – convert a detected pattern into a 0‑100 score
    # ------------------------------------------------------------------
    def get_hs_score(self, pattern: Optional[HeadAndShouldersPattern]) -> float:
        """
        Translate a detected pattern into a 0‑100 score.

        Scoring rules:
            * 0   – no pattern / neckline not broken / no retest
            * 80+ – valid for entry (neckline broken & retested)
            * +strength × 20 (strength is 0‑1) → max 100
        """
        if pattern is None:
            return 0.0

        if not pattern.neckline_broken:
            logger.info("H&S: Neckline not broken yet – score 0")
            return 0.0

        if not pattern.neckline_retested:
            logger.info("H&S: Waiting for neckline retest – score 0")
            return 0.0

        if not pattern.valid_for_entry:
            logger.info("H&S: Retest occurred but no rejection – score 0")
            return 0.0

        base = 80.0
        bonus = pattern.strength * 20.0
        total = base + bonus
        logger.info(f"H&S: Valid for entry – score {total:.0f}")
        return total

# ----------------------------------------------------------------------
# Global singleton – import this from ``src/main.py`` and use it directly
# ----------------------------------------------------------------------

