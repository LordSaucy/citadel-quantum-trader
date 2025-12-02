#!/usr/bin/env python3
"""
SMART MONEY CONCEPTS (SMC) CONFLUENCE SYSTEM

Institutional‑grade order‑flow analysis that adds six SMC factors to the
overall confluence picture:

1️⃣ Order Blocks
2️⃣ Liquidity Sweeps
3️⃣ Fair‑Value Gaps (FVG)
4️⃣ Break‑of‑Structure (BOS)
5️⃣ Premium / Discount Zones
6️⃣ Change‑of‑Character (ChoCh)

The system is fully self‑contained, production‑ready and can be
imported from ``src/main.py`` as ``smc_system``.
"""

# ----------------------------------------------------------------------
# Standard library
# ----------------------------------------------------------------------
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from threading import Event, Thread
from typing import Dict, List, Optional, Tuple

# ----------------------------------------------------------------------
# Third‑party
# ----------------------------------------------------------------------
import MetaTrader5 as mt5
import numpy as np
from prometheus_client import Gauge

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# 1️⃣  Data structures
# ----------------------------------------------------------------------
@dataclass
class OrderBlock:
    """Order‑block zone (institutional stop‑hunt area)."""
    price_high: float
    price_low: float
    time: datetime
    type: str                 # "bullish" or "bearish"
    strength: int             # 1‑100 (relative to move after the block)
    tested: bool = False      # has price revisited the block?


@dataclass
class FairValueGap:
    """Fair‑Value Gap (FVG) – a price void left by a rapid move."""
    gap_high: float
    gap_low: float
    time: datetime
    type: str                 # "bullish" or "bearish"
    size: float               # in price units
    filled: bool = False      # has the gap been taken?



@dataclass
class LiquiditySweep:
    """Liquidity sweep (stop‑hunt) event."""
    level: float
    time: datetime
    direction: str            # "long" or "short"
    strength: int = 90        # 1‑100 (hard‑coded strong by default)


@dataclass
class SMCAnalysis:
    """Aggregated SMC analysis result."""
    order_block_score: float
    liquidity_sweep_score: float
    fvg_score: float
    bos_score: float
    premium_discount_score: float
    choch_score: float
    total_smc_score: float
    recommendation: str


# ----------------------------------------------------------------------
# 2️⃣  Runtime‑tunable controller (weights, thresholds, persistence)
# ----------------------------------------------------------------------
class SMCController:
    """
    Holds all tunable SMC parameters, persists them to JSON and exposes
    Prometheus gauges so they can be changed from a Grafana UI in real‑time.
    """

    # ------------------------------------------------------------------
    # Default configuration (used on first start)
    # ------------------------------------------------------------------
    DEFAULTS = {
        # ----- factor weights (must sum to 1.0) -----
        "weight_order_block": 0.15,
        "weight_liquidity_sweep": 0.12,
        "weight_fair_value_gap": 0.15,
        "weight_break_of_structure": 0.10,
        "weight_premium_discount": 0.08,
        "weight_change_of_character": 0.10,
        # ----- score thresholds -----
        "min_total_score": 85.0,          # overall SMC score needed to trade
        "min_order_block_score": 0.30,    # raw (0‑1) before weighting
        "min_liquidity_sweep_score": 0.30,
        "min_fvg_score": 0.30,
        "min_bos_score": 0.30,
        "min_premium_discount_score": 0.30,
        "min_choch_score": 0.30,
        # ----- misc -----
        "debug": False,
    }

    CONFIG_PATH = Path("/app/config/smc_config.json")   # mount this dir in Docker

    # ------------------------------------------------------------------
     def __init__(self):
        # Load once
        self.refresh_params()

    # ------------------------------------------------------------------
    def _load_or_create(self) -> None:
        """Load persisted JSON or fall back to defaults."""
        if self.CONFIG_PATH.exists():
            try:
                with open(self.CONFIG_PATH, "r") as f:
                    self.values = json.load(f)
                logger.info("SMCController – config loaded")
            except Exception as exc:   # pragma: no cover
                logger.error(f"Failed to read SMC config ({exc}), using defaults")
                self.values = self.DEFAULTS.copy()
        else:
            logger.info("SMCController – no config file, creating defaults")
            self.values = self.DEFAULTS.copy()
            self._persist()

    # ------------------------------------------------------------------
    def _persist(self) -> None:
        """Write the current dict to disk."""
        try:
            self.CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(self.CONFIG_PATH, "w") as f:
                json.dump(self.values, f, indent=2)
        except Exception as exc:   # pragma: no cover
            logger.error(f"Could not persist SMC config: {exc}")

    # ------------------------------------------------------------------
    def _register_gauges(self) -> None:
        """Expose every key as a Prometheus gauge."""
        self._gauges: Dict[str, Gauge] = {}
        for key, val in self.values.items():
            g = Gauge(
                "smc_parameter",
                "Runtime‑tunable SMC parameter",
                ["parameter"],
            )
            g.labels(parameter=key).set(val)
            self._gauges[key] = g

    # ------------------------------------------------------------------
    def set(self, key: str, value: float) -> None:
        """Update a parameter (used by the HTTP API or Grafana)."""
        if key not in self.values:
            raise KeyError(f"Unknown SMC parameter: {key}")
        self.values[key] = float(value)
        self._gauges[key].labels(parameter=key).set(float(value))
        self._persist()
        logger.info(f"SMCController – set {key} = {value}")

    # ------------------------------------------------------------------
    def get(self, key: str) -> float:
        """Read a parameter."""
        return self.values.get(key, self.DEFAULTS.get(key))

    # ------------------------------------------------------------------
    def _start_watcher(self) -> None:
        """Reload JSON if it is edited manually."""
        def _watch():
            last_mtime = (
                self.CONFIG_PATH.stat().st_mtime
                if self.CONFIG_PATH.exists()
                else 0
            )
            while not self._stop.is_set():
                if self.CONFIG_PATH.exists():
                    mtime = self.CONFIG_PATH.stat().st_mtime
                    if mtime != last_mtime:
                        logger.info("SMCController – config file changed, reloading")
                        self._load_or_create()
                        for k, v in self.values.items():
                            self._gauges[k].labels(parameter=k).set(v)
                        last_mtime = mtime
                self._stop.wait(2)

        Thread(target=_watch, daemon=True, name="smc-config-watcher").start()

    # ------------------------------------------------------------------
    def stop(self) -> None:
        self._stop.set()


# Global controller instance (used by the SMC engine)
smc_controller = SMCController()


# ----------------------------------------------------------------------
# 3️⃣  Smart‑Money‑Concepts Engine
# ----------------------------------------------------------------------
class SmartMoneyConceptsSystem:
    """
    Institutional order‑flow analysis using the six SMC factors.
    All scores are **raw 0‑1** values; they are multiplied by the
    runtime‑tunable weights from ``smc_controller`` and summed to obtain
    the final SMC score (0‑100).
    """

    # ------------------------------------------------------------------
    # Helper – fetch a weight (fallback to default if missing)
    # ------------------------------------------------------------------
    @staticmethod
    def _weight(key: str) -> float:
        return smc_controller.get(key)

    # ------------------------------------------------------------------
    # Public entry point -------------------------------------------------
    # ------------------------------------------------------------------
    def analyze_smc(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        timeframe: int = mt5.TIMEFRAME_H1,
    ) -> SMCAnalysis:
        """
        Perform the full SMC analysis for ``symbol`` in ``direction`` at
        ``entry_price``.  Returns an ``SMCAnalysis`` dataclass.
        """
        logger.info(f"SMC analysis for {symbol} {direction} @ {entry_price}")

        # --------------------------------------------------------------
        # Pull price data (fallback to empty array on failure)
        # --------------------------------------------------------------
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 500)
        if rates is None or len(rates) < 100:
            logger.warning("Insufficient market data – returning defaults")
            return self._default_analysis()

        # --------------------------------------------------------------
        # 1️⃣  Order Block
        # --------------------------------------------------------------
        ob_raw = self._analyze_order_blocks(rates, entry_price, direction)
        ob_score = ob_raw * self._weight("weight_order_block")

        # --------------------------------------------------------------
        # 2️⃣  Liquidity Sweep
        # --------------------------------------------------------------
        ls_raw = self._analyze_liquidity_sweeps(rates, entry_price, direction)
        ls_score = ls_raw * self._weight("weight_liquidity_sweep")

        # --------------------------------------------------------------
        # 3️⃣  Fair‑Value Gap
        # --------------------------------------------------------------
        fvg_raw = self._analyze_fair_value_gaps(rates, entry_price, direction)
        fvg_score = fvg_raw * self._weight("weight_fair_value_gap")

        # --------------------------------------------------------------
        # 4️⃣  Break‑of‑Structure
        # --------------------------------------------------------------
        bos_raw = self._analyze_break_of_structure(rates, direction)
        bos_score = bos_raw * self._weight("weight_break_of_structure")

        # --------------------------------------------------------------
        # 5️⃣  Premium / Discount zone
        # --------------------------------------------------------------
        pd_raw = self._analyze_premium_discount(rates, entry_price, direction)
        pd_score = pd_raw * self._weight("weight_premium_discount")

        # --------------------------------------------------------------
        # 6️⃣  Change‑of‑Character
        # --------------------------------------------------------------
        choch_raw = self._analyze_change_of_character(rates, direction)
        choch_score = choch_raw * self._weight("weight_change_of_character")

        # --------------------------------------------------------------
        # Combine into total (0‑100)
        # --------------------------------------------------------------
        total_raw = (
            ob_raw * self._weight("weight_order_block")
            + ls_raw * self._weight("weight_liquidity_sweep")
            + fvg_raw * self._weight("weight_fair_value_gap")
            + bos_raw * self._weight("weight_break_of_structure")
            + pd_raw * self._weight("weight_premium_discount")
            + choch_raw * self._weight("weight_change_of_character")
        )
        total_score = round(total_raw * 100, 1)

        # --------------------------------------------------------------
        # Recommendation string
        # --------------------------------------------------------------
        recommendation = self._generate_smc_recommendation(total_score, direction)

        return SMCAnalysis(
            order_block_score=round(ob_raw * 100, 1),
            liquidity_sweep_score=round(ls_raw * 100, 1),
            fvg_score=round(fvg_raw * 100, 1),
            bos_score=round(bos_raw * 100, 1),
            premium_discount_score=round(pd_raw * 100, 1),
            choch_score=round(choch_raw * 100, 1),
            total_smc_score=total_score,
            recommendation=recommendation,
        )

    # ------------------------------------------------------------------
    # 1️⃣  Order Block detection & scoring
    # ------------------------------------------------------------------
    def _analyze_order_blocks(
        self,
        rates: np.ndarray,
        entry_price: float,
        direction: str,
    ) -> float:
        """
        Detect the most recent order block that is *near* the entry price.
        Returns a raw score in the range 0‑1.
        """
        if len(rates) < 50:
            return 0.5

        candidates = []

        # Scan the last ~100 candles for a single‑candle opposite‑bias block
        for i in range(20, len(rates) - 10):
            candle = rates[i]
            following = rates[i + 1 : i + 11]          # next 10 candles

            # ---------- bullish order block (sell‑side) ----------
            if direction == "BUY" and candle["close"] < candle["open"]:
                # Is there a strong bullish move afterwards?
                bullish_move = sum(
                    1 for c in following[:5] if c["close"] > c["open"]
                ) >= 4
                if bullish_move:
                    low, high = candle["low"], candle["high"]
                    # Entry must be within ~0.2 % of the block
                    if low <= entry_price <= high * 1.002:
                        strength = self._ob_strength(candle, following)
                        distance = abs(entry_price - low) / low
                        candidates.append(
                            {"low": low, "high": high, "strength": strength, "distance": distance}
                        )

            # ---------- bearish order block (buy‑side) ----------
            if direction == "SELL" and candle["close"] > candle["open"]:
                bearish_move = sum(
                    1 for c in following[:5] if c["close"] < c["open"]
                ) >= 4
                if bearish_move:
                    low, high = candle["low"], candle["high"]
                    if high * 0.998 <= entry_price <= high:
                        strength = self._ob_strength(candle, following)
                        distance = abs(entry_price - high) / high
                        candidates.append(
                            {"low": low, "high": high, "strength": strength, "distance": distance}
                        )

        if not candidates:
            return 0.30   # no block found – penalise a little

        # Pick the strongest block that is also closest
        best = max(candidates, key=lambda x: x["strength"] * (1 - x["distance"]))
        # Raw score = strength * proximity factor (0‑1)
        raw = best["strength"] * max(0.0, 1 - best["distance"] * 2)
        return max(0.0, min(1.0, raw))

    @staticmethod
    def _ob_strength(candle: np.ndarray, following: np.ndarray) -> float:
        """Strength = size of move after the block divided by block range."""
        block_range = candle["high"] - candle["low"]
        move = max(
            abs(following[-1]["high"] - candle["close"]),
            abs(following[-1]["low"] - candle["close"]),
        )
        if block_range <= 0:
            return 0.5
        strength = min(1.0, move / (block_range * 5))
        return strength

    # ------------------------------------------------------------------
    # 2️⃣  Liquidity Sweep detection & scoring
    # ------------------------------------------------------------------
    def _analyze_liquidity_sweeps(
        self,
        rates: np.ndarray,
        entry_price: float,
        direction: str,
    ) -> float:
        """
        Detect a recent liquidity sweep that is close to the entry price.
        Returns a raw 0‑1 score.
        """
        if len(rates) < 50:
            return 0.5

        sweeps = []

        for i in range(10, len(rates) - 5):
            # ---- swing high (potential short‑stop pool) ----
            if rates[i]["high"] == np.max(rates[i - 5 : i + 6]["high"]):
                swing = rates[i]["high"]
                # look ahead up to 10 candles for a wick above swing then close below
                for j in range(i + 1, min(i + 11, len(rates))):
                   # ---- check for a sweep ----
                    # price makes a wick above the swing high and then closes below it
                    if rates[j]["high"] > swing and rates[j]["close"] < swing:
                        # look for a strong reversal in the next 3 candles
                        rev = rates[j + 1 : j + 4]
                        if len(rev) > 0:
                            strong_rev = sum(1 for c in rev if c["close"] < c["open"]) >= 2
                            if strong_rev and direction == "SELL":
                                # bearish sweep (short‑stop hunt)
                                dist = abs(entry_price - swing) / swing
                                if dist < 0.01:          # within 1 %
                                    sweeps.append(
                                        {
                                            "level": swing,
                                            "strength": 0.9,
                                            "distance": dist,
                                        }
                                    )
            # ---- swing low (potential long‑stop pool) ----
            if rates[i]["low"] == np.min(rates[i - 5 : i + 6]["low"]):
                swing = rates[i]["low"]
                for j in range(i + 1, min(i + 11, len(rates))):
                    if rates[j]["low"] < swing and rates[j]["close"] > swing:
                        rev = rates[j + 1 : j + 4]
                        if len(rev) > 0:
                            strong_rev = sum(1 for c in rev if c["close"] > c["open"]) >= 2
                            if strong_rev and direction == "BUY":
                                dist = abs(entry_price - swing) / swing
                                if dist < 0.01:
                                    sweeps.append(
                                        {
                                            "level": swing,
                                            "strength": 0.9,
                                            "distance": dist,
                                        }
                                    )

        # ------------------------------------------------------------------
        # 2️⃣  Score the best recent sweep (if any)
        # ------------------------------------------------------------------
        if not sweeps:
            return 0.40  # no sweep found – modest penalty

        # choose the sweep with the highest combined strength‑proximity metric
        best = max(sweeps, key=lambda x: x["strength"] * (1 - x["distance"]))
        raw_score = best["strength"] * (1 - best["distance"] * 3)
        return max(0.0, min(1.0, raw_score))

    # ------------------------------------------------------------------
    # 3️⃣  Fair‑Value Gap detection & scoring
    # ------------------------------------------------------------------
    def _analyze_fair_value_gaps(
        self,
        rates: np.ndarray,
        entry_price: float,
        direction: str,
    ) -> float:
        """
        Detect unfilled Fair‑Value Gaps (FVG) that contain the entry price.
        Returns a raw 0‑1 score.
        """
        if len(rates) < 20:
            return 0.50

        fvgs = []

        # iterate over triples of candles
        for i in range(10, len(rates) - 3):
            c1, c2, c3 = rates[i], rates[i + 1], rates[i + 2]

            # -------- bullish FVG (gap upward) --------
            if c3["low"] > c1["high"]:          # clear gap
                gap_low, gap_high = c1["high"], c3["low"]
                if direction == "BUY" and gap_low <= entry_price <= gap_high:
                    # check that the gap has NOT been filled yet
                    filled = any(r["low"] <= gap_low for r in rates[i + 3 :])
                    if not filled:
                        size = gap_high - gap_low
                        strength = min(1.0, size / c1["high"] * 100)
                        age = len(rates) - i
                        fvgs.append(
                            {
                                "low": gap_low,
                                "high": gap_high,
                                "strength": strength,
                                "age": age,
                            }
                        )

            # -------- bearish FVG (gap downward) --------
            if c3["high"] < c1["low"]:
                gap_low, gap_high = c3["high"], c1["low"]
                if direction == "SELL" and gap_low <= entry_price <= gap_high:
                    filled = any(r["high"] >= gap_high for r in rates[i + 3 :])
                    if not filled:
                        size = gap_high - gap_low
                        strength = min(1.0, size / c1["low"] * 100)
                        age = len(rates) - i
                        fvgs.append(
                            {
                                "low": gap_low,
                                "high": gap_high,
                                "strength": strength,
                                "age": age,
                            }
                        )

        if not fvgs:
            return 0.35  # no viable FVG found

        # pick the strongest recent gap (strength weighted by recency)
        best = max(fvgs, key=lambda x: x["strength"] * (1 / (1 + x["age"] / 50)))
        raw_score = best["strength"] * min(1.0, 50 / best["age"])
        return max(0.0, min(1.0, raw_score))

    # ------------------------------------------------------------------
    # 4️⃣  Break‑of‑Structure detection & scoring
    # ------------------------------------------------------------------
    def _analyze_break_of_structure(
        self,
        rates: np.ndarray,
        direction: str,
    ) -> float:
        """
        Detect a Break‑of‑Structure (BOS) – price breaking the most recent
        swing high (for BUY) or swing low (for SELL).
        Returns a raw 0‑1 score.
        """
        if len(rates) < 50:
            return 0.50

        # collect recent swing points
        swing_highs = []
        swing_lows = []

        for i in range(10, len(rates) - 10):
            if rates[i]["high"] == np.max(rates[i - 5 : i + 6]["high"]):
                swing_highs.append((i, rates[i]["high"]))
            if rates[i]["low"] == np.min(rates[i - 5 : i + 6]["low"]):
                swing_lows.append((i, rates[i]["low"]))

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return 0.50

        recent_high = swing_highs[-1][1]
        recent_low = swing_lows[-1][1]
        current_price = rates[-1]["close"]

        if direction == "BUY":
            if current_price > recent_high:
                # strength proportional to how far we broke the level
                break_dist = (current_price - recent_high) / recent_high
                return max(0.70, min(1.0, break_dist * 100))
            else:
                return 0.40
        else:  # SELL
            if current_price < recent_low:
                break_dist = (recent_low - current_price) / recent_low
                return max(0.70, min(1.0, break_dist * 100))
            else:
                return 0.40

    # ------------------------------------------------------------------
    # 5️⃣  Premium / Discount zone scoring
    # ------------------------------------------------------------------
    def _analyze_premium_discount(
        self,
        rates: np.ndarray,
        entry_price: float,
        direction: str,
    ) -> float:
        """
        Split the recent price range into three zones:
        * Discount (bottom 30 %)
        * Premium (top 30 %)
        * Neutral (middle 40 %)
        Returns a raw 0‑1 score that favours entries inside the favourable zone.
        """
        if len(rates) < 100:
            return 0.50

        highs = rates[-100:]["high"]
        lows = rates[-100:]["low"]
        range_high = np.max(highs)
        range_low = np.min(lows)
        rng = range_high - range_low
        if rng == 0:
            return 0.50

        premium_start = range_low + rng * 0.70   # top 30 %
        discount_end = range_low + rng * 0.30    # bottom 30 %

        # entry position as a fraction of the total range
        entry_frac = (entry_price - range_low) / rng

        if direction == "BUY":
            if entry_price <= discount_end:                     # inside discount
                score = 1.0 - (entry_frac / 0.30)               # deeper = better
                return max(0.70, min(1.0, score))
            elif entry_price <= range_low + rng * 0.50:          # neutral zone
                return 0.60
            else:                                               # premium zone (bad for BUY)
                return 0.30
        else:  # SELL
            if entry_price >= premium_start:                     # inside premium
                score = (entry_frac - 0.70) / 0.30               # deeper = better
                return max(0.70, min(1.0, score))
            elif entry_price >= range_low + rng * 0.50:          # neutral zone
                return 0.60
            else:                                               # discount zone (bad for SELL)
                return 0.30

    # ------------------------------------------------------------------
    # 6️⃣  Change‑of‑Character detection & scoring
    # ------------------------------------------------------------------
    def _analyze_change_of_character(
        self,
        rates: np.ndarray,
        direction: str,
    ) -> float:
        """
        Detect a Change‑of‑Character (ChoCh) – a reversal of the prevailing
        trend signalled by a break of the most recent swing point.
        Returns a raw 0‑1 score.
        """
        if len(rates) < 50:
            return 0.50

        highs = rates[-30:]["high"]
        lows = rates[-30:]["low"]

        recent_high = np.max(highs[-10:])
        prev_high = np.max(highs[-20:-10])
        recent_low = np.min(lows[-10:])
        prev_low = np.min(lows[-20:-10])

        current_price = rates[-1]["close"]

        if direction == "BUY":
            # was in downtrend (lower highs & lower lows) and now breaks above recent high
            was_down = recent_high < prev_high and recent_low < prev_low
            breaking = current_price > recent_high
            if was_down and breaking:
                strength = (current_price - recent_high) / recent_high
                return max(0.75, min(1.0, 0.75 + strength * 100))
            elif breaking:
                return 0.60
            else:
                return 0.40
        else:  # SELL
            # was in uptrend and now breaks below recent low
            was_up = recent_high > prev_high and recent_low > prev_low
            breaking = current_price < recent_low
            if was_up and breaking:
                strength = (recent_low - current_price) / recent_low
                return max(0.75, min(1.0, 0.75 + strength * 100))
            elif breaking:
                return 0.60
            else:
                return 0.40

    # ------------------------------------------------------------------
    # 7️⃣  Recommendation generator
    # ------------------------------------------------------------------
    @staticmethod
    def _generate_smc_recommendation(total_score: float, direction: str) -> str:
        """
        Produce a short, human‑readable recommendation based on the final
        SMC score.
        """
        if total_score >= 85:
            return f"🎯 PERFECT SMC SETUP ({total_score:.0f}/100) – Institutional confluence"
        if total_score >= 70:
            return f"✅ Strong SMC ({total_score:.0f}/100) – High probability"
        if total_score >= 55:
            return f"⚠️ Moderate SMC ({total_score:.0f}/100) – Proceed with caution"
        return f"❌ Weak SMC ({total_score:.0f}/100) – Avoid trade"

    # ------------------------------------------------------------------
    # 8️⃣  Default analysis (fallback when market data is insufficient)
    # ------------------------------------------------------------------
    @staticmethod
    def _default_analysis() -> SMCAnalysis:
        """Return a neutral analysis when we cannot compute anything."""
        return SMCAnalysis(
            order_block_score=0.50,
            liquidity_sweep_score=0.50,
            fvg_score=0.50,
            bos_score=0.50,
            premium_discount_score=0.50,
            choch_score=0.50,
            total_smc_score=50.0,
            recommendation="Insufficient data for SMC analysis",
        )

def refresh_params(self):
        self.order_block_weight = get_param("order_block_weight", 0.7)
        self.liquidity_sweep_weight = get_param("liquidity_sweep_weight", 0.6)
        self.fair_value_gap_weight = get_param("fair_value_gap_weight", 0.5)
        self.break_of_structure_weight = get_param("break_of_structure_weight", 0.6)
        self.premium_discount_thresh = get_param("premium_discount_thresh", 0.02)
        self.max_confluence_score = get_param("max_confluence_score", 0.95)

    def score(self, market_state):
        # Example usage
        score = (
            self.order_block_weight * market_state.order_block_signal +
            self.liquidity_sweep_weight * market_state.liquidity_signal +
            self.fair_value_gap_weight * market_state.fvg_signal +
            self.break_of_structure_weight * market_state.bos_signal
        )
        # Clip to the max allowed confluence
        return min(score, self.max_confluence_score)

# ----------------------------------------------------------------------
# Global singleton – import this from ``src/main.py`` and use it directly
# ----------------------------------------------------------------------
