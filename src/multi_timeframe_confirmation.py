#!/usr/bin/env python3
"""
CITADEL QUANTUM TRADER – MULTI‑TIMEFRAME CONFIRMATION SYSTEM

Lever #4 of the 7‑lever win‑rate optimisation system.
Ensures that a candidate trade is aligned across several time‑frames
before the bot actually sends the order.

Author: Lawful Banker
Created: 2024‑11‑26
Version: 2.0 – Production‑ready
"""

# ----------------------------------------------------------------------
# Standard library
# ----------------------------------------------------------------------
import logging
from datetime import datetime
from enum import Enum
from typing import Dict, List, Tuple

# ----------------------------------------------------------------------
# Third‑party
# ----------------------------------------------------------------------
import MetaTrader5 as mt5
import numpy as np

# ----------------------------------------------------------------------
# Logging configuration (writes to ./logs/multi_timeframe_confirmation.log)
# ----------------------------------------------------------------------
from pathlib import Path

LOG_DIR = Path("./logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "multi_timeframe_confirmation.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Enum – how many time‑frames agree
# ----------------------------------------------------------------------
class TimeframeAlignment(Enum):
    """Possible alignment states across the analysed time‑frames."""

    PERFECT = "perfect"      # All 5 TFs agree
    STRONG = "strong"        # 4 / 5 TFs agree
    MODERATE = "moderate"    # 3 / 5 TFs agree
    WEAK = "weak"            # 2 / 5 TFs agree
    CONFLICTING = "conflicting"  # Opposing signals dominate


# ----------------------------------------------------------------------
# Dataclass – the result of a full MTF analysis
# ----------------------------------------------------------------------
from dataclasses import dataclass


@dataclass
class MultiTimeframeAnalysis:
    """Container for everything the confirmation engine produces."""

    alignment: TimeframeAlignment          # Overall alignment enum
    alignment_score: float                 # 0‑100 (higher = better)
    primary_trend: str                     # BUY / SELL / NEUTRAL (from higher TFs)
    timeframe_signals: Dict[str, str]      # e.g. {"M15": "BUY", "H4": "NEUTRAL", …}
    key_levels: Dict[str, List[float]]     # support / resistance per TF
    recommendation: str                    # Human‑readable recommendation
    confidence: float                      # 0‑100 confidence metric


# ----------------------------------------------------------------------
# Main class – orchestrates the whole multi‑timeframe check
# ----------------------------------------------------------------------
class MultiTimeframeConfirmation:
    """
    Analyses five time‑frames (M15, H1, H4, D1, W1) and decides whether a
    proposed trade is sufficiently aligned to be executed.

    The class is deliberately **stateless** apart from the optional
    ``require_alignment_score`` threshold; all heavy lifting is performed
    on‑the‑fly using MT5 historical data.
    """

    # ------------------------------------------------------------------
    # Time‑frame mapping (MT5 constants)
    # ------------------------------------------------------------------
    TIMEFRAMES = {
        "M15": mt5.TIMEFRAME_M15,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
        "W1": mt5.TIMEFRAME_W1,
    }

    # ------------------------------------------------------------------
    # Relative importance of each TF when computing the overall score
    # ------------------------------------------------------------------
    WEIGHTS = {
        "M15": 0.10,   # entry‑precision – least important
        "H1": 0.20,
        "H4": 0.30,
        "D1": 0.25,
        "W1": 0.15,    # long‑term bias – smallest weight
    }

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self):
        """
        ``require_alignment_score`` – minimum weighted score (0‑100) that
        a trade must achieve to be considered “aligned”.  The default of
        70 % is a good compromise between aggressiveness and safety.
        """
        self.require_alignment_score = mtf_config.get("require_alignment_score", 70.0)
        logger.info(
            f"Multi‑timeframe confirmation initialised – "
            f"min alignment score = {self.require_alignment_score:.0f}%"
        )

 @property
    def TIMEFRAMES(self) -> dict:
        # Convert the string identifiers (e.g. "TIMEFRAME_M15") to the actual
        # mt5 constants at runtime.
        mapping = {}
        for name, const_name in mtf_config.get("timeframes", {}).items():
            mapping[name] = getattr(mt5, const_name)
        return mapping

@property
    def WEIGHTS(self) -> dict:
        return mtf_config.get("weights", {
            "M15": 0.10,
            "H1":  0.20,
            "H4":  0.30,
            "D1":  0.25,
            "W1":  0.15,
        })


@property
    def EMA_WINDOWS(self) -> Tuple[int, int]:
        cfg = mtf_config.get("ema_windows", {})
        short = cfg.get("short", 20)
        long = cfg.get("long", 50)
        return short, long

 @property
    def PROXIMITY_TOLERANCE(self) -> float:
        return mtf_config.get("proximity_tolerance_pct", 0.5)

    # --------------------------------------------------------------

@property
    def BOOST_THRESHOLDS(self) -> dict:
        return mtf_config.get(
            "alignment_boost",
            {"high_confidence_threshold": 85.0, "medium_confidence_threshold": 75.0},
        )

    # ------------------------------------------------------------------
    # Public entry point – analyse *all* TFs for a symbol / direction
    # ------------------------------------------------------------------
    def analyze_all_timeframes(
        self, symbol: str, proposed_direction: str
    ) -> MultiTimeframeAnalysis:
        """
        Run the full MTF pipeline:

        1️⃣  Gather a BUY / SELL / NEUTRAL signal from each TF.
        2️⃣  Compute a weighted alignment score.
        3️⃣  Derive a “primary trend” from the higher TFs.
        4️⃣  Extract key support / resistance levels from H4 & D1.
        5️⃣  Produce a human‑readable recommendation.
        6️⃣  Estimate a confidence percentage.

        Returns a :class:`MultiTimeframeAnalysis` instance.
        """
        logger.info(f"MTF analysis – symbol={symbol}, direction={proposed_direction}")

        # ----------------------------------------------------------------
        # 1️⃣  Individual TF signals
        # ----------------------------------------------------------------
        timeframe_signals: Dict[str, str] = {}
        for tf_name, tf_constant in self.TIMEFRAMES.items():
            signal = self._analyze_single_timeframe(symbol, tf_constant)
            timeframe_signals[tf_name] = signal
            logger.debug(f"  {tf_name}: {signal}")

        # ----------------------------------------------------------------
        # 2️⃣  Alignment score & enum
        # ----------------------------------------------------------------
        alignment, alignment_score = self._calculate_alignment(
            timeframe_signals, proposed_direction
        )

        # ----------------------------------------------------------------
        # 3️⃣  Primary trend (derived mainly from higher TFs)
        # ----------------------------------------------------------------
        primary_trend = self._determine_primary_trend(timeframe_signals)

        # ----------------------------------------------------------------
        # 4️⃣  Key support / resistance levels (H4 & D1)
        # ----------------------------------------------------------------
        key_levels = self._identify_key_levels(symbol)

        # ----------------------------------------------------------------
        # 5️⃣  Recommendation text
        # ----------------------------------------------------------------
        recommendation = self._generate_recommendation(
            alignment, alignment_score, proposed_direction, primary_trend
        )

        # ----------------------------------------------------------------
        # 6️⃣  Confidence metric
        # ----------------------------------------------------------------
        confidence = self._calculate_confidence(alignment_score, timeframe_signals)

        return MultiTimeframeAnalysis(
            alignment=alignment,
            alignment_score=alignment_score,
            primary_trend=primary_trend,
            timeframe_signals=timeframe_signals,
            key_levels=key_levels,
            recommendation=recommendation,
            confidence=confidence,
        )

    # ------------------------------------------------------------------
    # 1️⃣  Single‑timeframe analysis (EMA‑based trend detection)
    # ------------------------------------------------------------------
    def _analyze_single_timeframe(self, symbol: str, timeframe) -> str:
        short_w, long_w = self.EMA_WINDOWS          # <-- new
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, max(short_w, long_w) + 10)
        if not rates or len(rates) < long_w:
            return "NEUTRAL"

        closes = rates["close"]
        ema_short = np.mean(closes[-short_w:])
        ema_long  = np.mean(closes[-long_w:])
        price = closes[-1]

        # Same logic as before, just using the configurable windows
        if price > ema_short > ema_long:
            if (price - ema_long) / ema_long > 0.01:
                return "BUY"
            return "NEUTRAL"
        if price < ema_short < ema_long:
            if (ema_long - price) / ema_long > 0.01:
                return "SELL"
            return "NEUTRAL"
        return "NEUTRAL"

    # ------------------------------------------------------------------
    # 2️⃣  Weighted alignment calculation
    # ------------------------------------------------------------------
    def _calculate_alignment(
        self, timeframe_signals: Dict[str, str], proposed_direction: str
    ) -> Tuple[TimeframeAlignment, float]:
        """
        Counts how many TFs agree with the *proposed* direction and
        produces a weighted score (0‑100).  The enum reflects the raw
        count, while the numeric score incorporates the per‑TF weights.
        """
        matching = opposing = neutral = 0
        weighted_score = 0.0

        for tf_name, signal in timeframe_signals.items():
            weight = self.WEIGHTS[tf_name]

            if signal == proposed_direction:
                matching += 1
                weighted_score += weight * 100
            elif signal == "NEUTRAL":
                neutral += 1
                weighted_score += weight * 50   # half credit for neutral
            else:
                opposing += 1
                # no credit for opposing

        # Decide enum based on raw counts
        if matching >= 5:
            alignment = TimeframeAlignment.PERFECT
        elif matching >= 4:
            alignment = TimeframeAlignment.STRONG
        elif matching >= 3:
            alignment = TimeframeAlignment.MODERATE
        elif matching >= 2:
            alignment = TimeframeAlignment.WEAK
        else:
            alignment = TimeframeAlignment.CONFLICTING

        return alignment, weighted_score

    # ------------------------------------------------------------------
    # 3️⃣  Primary trend (bias from higher‑weight TFs)
    # ------------------------------------------------------------------
    def _determine_primary_trend(self, timeframe_signals: Dict[str, str]) -> str:
        """
        Aggregates the signals with the same weights used for alignment,
        then decides a “dominant” market direction.
        """
        buy_score = sell_score = 0.0
        for tf_name, signal in timeframe_signals.items():
            weight = self.WEIGHTS[tf_name]
            if signal == "BUY":
                buy_score += weight
            elif signal == "SELL":
                sell_score += weight

        # Require a 1.5× advantage to claim a clear trend
        if buy_score > sell_score * 1.5:
            return "BUY"
        if sell_score > buy_score * 1.5:
            return "SELL"
        return "NEUTRAL"

    # ------------------------------------------------------------------
    # 4️⃣  Key support / resistance extraction (H4 & D1)
    # ------------------------------------------------------------------
    def _identify_key_levels(self, symbol: str) -> Dict[str, List[float]]:
        """
        Scans the last 100 candles of the H4 and D1 charts,
        extracts swing highs / lows and returns the most recent 10
        unique levels for each TF.
        """
        key_levels: Dict[str, List[float]] = {}

        for tf_name in ("H4", "D1"):
            tf_const = self.TIMEFRAMES[tf_name]
            rates = mt5.copy_rates_from_pos(symbol, tf_const, 0, 100)

            if not rates:
                key_levels[tf_name] = []
                continue

            highs = rates["high"]
            lows = rates["low"]
            levels: List[float] = []

            # Simple swing‑high / swing‑low detection (5‑bar window)
            for i in range(10, len(rates) - 10):
                if highs[i] == np.max(highs[i - 5 : i + 5]):
                    levels.append(highs[i])
                if lows[i] == np.min(lows[i - 5 : i + 5]):
                    levels.append(lows[i])

            # Keep the most recent 10 distinct levels
            key_levels[tf_name] = sorted(set(levels))[-10:]

        return key_levels

    # ------------------------------------------------------------------
    # 5️⃣  Human‑readable recommendation
    # ------------------------------------------------------------------
    def _generate_recommendation(
        self,
        alignment: TimeframeAlignment,
        alignment_score: float,
        proposed_direction: str,
        primary_trend: str,
    ) -> str:
        """
        Returns a short sentence that can be shown to the trader /
        logged in the audit trail.
        """
        if alignment_score >= 85:
            return f"✅ Excellent MTF alignment ({alignment_score:.0f} %) – high‑probability setup"
        if alignment_score >= 70:
            return f"✅ Good MTF alignment ({alignment_score:.0f} %) – trade approved"
        if alignment_score >= 60:
            return f"⚠️ Moderate alignment ({alignment_score:.0f} %) – seek extra confluence"
        if alignment_score >= 50:
            return f"⚠️ Weak alignment ({alignment_score:.0f} %) – risky trade"
        return f"❌ Poor MTF alignment ({alignment_score:.0f} %) – avoid trade"

    # ------------------------------------------------------------------
    # 6️⃣  Confidence metric (adds a boost for higher‑TF agreement)
    # ------------------------------------------------------------------
    def _calculate_confidence(
        self, alignment_score: float, timeframe_signals: Dict[str, str]
    ) -> float:
        """
        Base confidence = 70 % of the alignment score.
        Add up to 30 % extra if the three highest‑weight TFs (H4, D1, W1)
        all emit a *non‑neutral* signal.
        """
        confidence = alignment_score * 0.70
        higher_tfs = ("D1", "W1", "H4")
        agreeing = sum(
            1 for tf in higher_tfs if timeframe_signals.get(tf) and timeframe_signals[tf] != "NEUTRAL"
        )
        confidence += (agreeing / len(higher_tfs)) * 30
        return min(100.0, confidence)

    # ------------------------------------------------------------------
    # 7️⃣  Proximity to a key level (used for entry‑timing checks)
    # ------------------------------------------------------------------

    def is_near_key_level(self, symbol: str, price: float) -> Tuple[bool, Optional[str], Optional[float]]:
        tol = self.PROXIMITY_TOLERANCE               # <-- new
        key_levels = self._identify_key_levels(symbol)
        for tf_name, levels in key_levels.items():
            for lvl in levels:
                if abs(price - lvl) / lvl * 100 <= tol:
                    level_type = "SUPPORT" if price > lvl else "RESISTANCE"
                    logger.info(
                        f"Price {price:.5f} near {tf_name} {level_type} at {lvl:.5f}"
                    )
                    return True, level_type, lvl
        return False, None, None


    # ------------------------------------------------------------------
    # 8️⃣  Final decision – should the bot place the trade?
    # ------------------------------------------------------------------
    def should_approve_trade(self, symbol: str, proposed_direction: str, entry_price: float) -> Tuple[bool, str, MultiTimeframeAnalysis]:
        analysis = self.analyze_all_timeframes(symbol, proposed_direction)

        # 1️⃣ alignment score
        if analysis.alignment_score < self.require_alignment_score:
            return False, f"MTF alignment too low: {analysis.alignment_score:.1f}% < {self.require_alignment_score:.0f}%", analysis

        # 2️⃣ proximity to key level
        near, lvl_type, lvl_price = self.is_near_key_level(symbol, entry_price)
        if near:
            if (
                (proposed_direction == "BUY" and lvl_type == "RESISTANCE")
                or (proposed_direction == "SELL" and lvl_type == "SUPPORT")
            ) and analysis.alignment_score < self.BOOST_THRESHOLDS["high_confidence_threshold"]:
                reason = (
                    f"Trade near {lvl_type.lower()} ({lvl_price:.5f}) "
                    f"and alignment ({analysis.alignment_score:.1f}%) is below "
                    f"{self.BOOST_THRESHOLDS['high_confidence_threshold']}%"
                )
                return False, reason, analysis

        # 3️⃣ primary trend conflict
        if (
            proposed_direction != analysis.primary_trend
            and analysis.primary_trend != "NEUTRAL"
        ):
            if analysis.alignment_score < self.BOOST_THRESHOLDS["medium_confidence_threshold"]:
                reason = (
                    f"Direction {proposed_direction} conflicts with primary trend "
                    f"{analysis.primary_trend} and alignment ({analysis.alignment_score:.1f}%) "
                    f"is below {self.BOOST_THRESHOLDS['medium_confidence_threshold']}%"
                )
                return False, reason, analysis

        # All good
        reason = f"MTF approved – {analysis.alignment.value} alignment ({analysis.alignment_score:.1f}%)"
        return True, reason, analysis

        # ----------------------------------------------------------------
        # All checks passed – trade is approved
        # ----------------------------------------------------------------
        reason = (
            f"MTF approved – {analysis.alignment.value} alignment "
           f"({analysis.alignment_score:.1f}%)"
        )
        return True, reason, analysis

    # ------------------------------------------------------------------
    # Helper: quick sanity‑check / demo when the module is executed directly
    # ------------------------------------------------------------------
    def _demo(self, symbol: str, direction: str) -> None:
        """
        Run a one‑off demonstration for a given symbol/direction.
        Prints the full analysis and the final decision to stdout.
        """
        # Grab a realistic entry price from the market
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            logger.error(f"Unable to fetch tick for {symbol}")
            return

        entry_price = tick.ask if direction.upper() == "BUY" else tick.bid

        approved, reason, analysis = self.should_approve_trade(
            symbol=symbol,
            proposed_direction=direction.upper(),
            entry_price=entry_price,
        )

        print("\n=== MULTI‑TIMEFRAME ANALYSIS ===")
        print(f"Symbol               : {symbol}")
        print(f"Proposed direction   : {direction.upper()}")
        print(f"Entry price          : {entry_price:.5f}")
        print(f"Alignment            : {analysis.alignment.value}")
        print(f"Alignment score      : {analysis.alignment_score:.2f}%")
        print(f"Primary trend        : {analysis.primary_trend}")
        print(f"Recommendation       : {analysis.recommendation}")
        print(f"Confidence           : {analysis.confidence:.1f}%")
        print("\nSignals per timeframe:")
        for tf, sig in analysis.timeframe_signals.items():
            print(f"  {tf:>3}: {sig}")

        print("\nKey levels (support / resistance):")
        for tf, levels in analysis.key_levels.items():
            lvl_str = ", ".join(f"{lvl:.5f}" for lvl in levels)
            print(f"  {tf:>3}: {lvl_str}")

        print("\nFinal decision:")
        print(f"  APPROVED?  : {'YES' if approved else 'NO'}")
        print(f"  REASON     : {reason}")

        # Clean‑up MT5 connection if this demo was the only use
        mt5.shutdown()


# ----------------------------------------------------------------------
# Global singleton – import this from ``src/main.py`` and use it directly
# ----------------------------------------------------------------------
mtf_confirmation = MultiTimeframeConfirmation(require_alignment_score=70)


# ----------------------------------------------------------------------
# Simple command‑line demo (run: ``python -m src.multi_timeframe_confirmation EURUSD BUY``)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python -m src.multi_timeframe_confirmation <SYMBOL> <BUY|SELL>")
        sys.exit(1)

    symbol_arg = sys.argv[1].upper()
    direction_arg = sys.argv[2].upper()
    if direction_arg not in ("BUY", "SELL"):
        print("Direction must be BUY or SELL")
        sys.exit(1)

    # Initialise MT5 (required before any MT5 call)
    if not mt5.initialize():
        print("Failed to initialise MetaTrader5")
        sys.exit(1)

    # Run the demo

    mtf_confirmation._demo(symbol_arg, direction_arg)

# src/multi_timeframe_confirmation.py  (only the changed parts shown)

from src.mtf_config import mtf_config   # <-- NEW import

class MultiTimeframeConfirmation:
    # ------------------------------------------------------------------
    # Constructor – now reads the threshold from the config dict
    # ------------------------------------------------------------------
    def __init__(self):
        # The config dict is *mutable*, so any reload will instantly affect
        # new instances (and the singleton we expose later).
        self.require_alignment_score = mtf_config.get("require_alignment_score", 70.0)
        logger.info(
            f"Multi‑timeframe confirmation initialised – "
            f"min alignment score = {self.require_alignment_score:.0f}%"
        )

    # ------------------------------------------------------------------
    # TIMEFRAMES – built from the config mapping
    # ------------------------------------------------------------------
    @property
    def TIMEFRAMES(self) -> dict:
        # Convert the string identifiers (e.g. "TIMEFRAME_M15") to the actual
        # mt5 constants at runtime.
        mapping = {}
        for name, const_name in mtf_config.get("timeframes", {}).items():
            mapping[name] = getattr(mt5, const_name)
        return mapping

    # ------------------------------------------------------------------
    # WEIGHTS – also read from the config
    # ------------------------------------------------------------------
    @property
    def WEIGHTS(self) -> dict:
        return mtf_config.get("weights", {
            "M15": 0.10,
            "H1":  0.20,
            "H4":  0.30,
            "D1":  0.25,
            "W1":  0.15,
        })

    # ------------------------------------------------------------------
    # EMA windows – now configurable
    # ------------------------------------------------------------------
    @property
    def EMA_WINDOWS(self) -> Tuple[int, int]:
        cfg = mtf_config.get("ema_windows", {})
        short = cfg.get("short", 20)
        long = cfg.get("long", 50)
        return short, long

    # ------------------------------------------------------------------
    # Proximity tolerance (used in is_near_key_level)
    # ------------------------------------------------------------------
    @property
    def PROXIMITY_TOLERANCE(self) -> float:
        return mtf_config.get("proximity_tolerance_pct", 0.5)

    # ------------------------------------------------------------------
    # Alignment‑boost thresholds (used in should_approve_trade)
    # ------------------------------------------------------------------
    @property
    def BOOST_THRESHOLDS(self) -> dict:
        return mtf_config.get(
            "alignment_boost",
            {"high_confidence_threshold": 85.0, "medium_confidence_threshold": 75.0},
        )
