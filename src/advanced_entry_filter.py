#!/usr/bin/env python3
"""
Advanced entry quality scoring system

Only takes top‑tier setups – part of the 7‑lever win‑rate optimisation
framework used by the Citadel Quantum Trader.

Author: Lawful Banker
Created: 2024‑11‑26
Version: 2.0 – Production‑ready
"""

# ----------------------------------------------------------------------
# Standard library
# ----------------------------------------------------------------------
import logging
from dataclasses import dataclass
from datetime import datetime, date, time as dt_time, timedelta
from typing import Optional, List, Dict, Tuple

# ----------------------------------------------------------------------
# Third‑party
# ----------------------------------------------------------------------
import MetaTrader5 as mt5
import numpy as np

# ----------------------------------------------------------------------
# Logging configuration (writes to ./logs/advanced_entry_filter.log)
# ----------------------------------------------------------------------
LOG_DIR = "./logs"
import os
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "advanced_entry_filter.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Dataclass – the result that the bot will consume
# ----------------------------------------------------------------------
@dataclass
class EntryQualityScore:
    """Result of the entry‑quality assessment."""

    total_score: float          # 0‑100
    trend_score: float          # 0‑25
    structure_score: float      # 0‑25
    momentum_score: float       # 0‑25
    risk_reward_score: float    # 0‑25
    should_trade: bool
    reasoning: str


# ----------------------------------------------------------------------
# Main filter class
# ----------------------------------------------------------------------
class EntryQualityFilter:
    """
    Filters trades based on a 0‑100 quality score.
    Only setups graded **A+** (≥ 85) or **A** (≥ 75) are allowed to trade.
    """

    # ------------------------------------------------------------------
    # Grade thresholds (expressed as total_score)
    # ------------------------------------------------------------------
    GRADE_A_PLUS = 85  # always take
    GRADE_A = 75       # take
    GRADE_B = 65       # skip (default threshold)
    GRADE_C = 55       # definitely skip

    # ------------------------------------------------------------------
    def __init__(self, min_quality_score: float = 75):
        """
        ``min_quality_score`` – the minimum total_score required for a trade.
        """
        self.min_quality_score = min_quality_score
        logger.info(f"EntryQualityFilter initialised – minimum score {min_quality_score}")

    # ------------------------------------------------------------------
    # PUBLIC API – called by the trading engine
    # ------------------------------------------------------------------
    def assess_entry_quality(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
    ) -> EntryQualityScore:
        """
        Run the full 4‑component quality assessment and return a
        ``EntryQualityScore`` instance.
        """
        # 1️⃣ Trend quality (25 pts)
        trend_score = self._assess_trend_quality(symbol, direction)

        # 2️⃣ Structure quality (25 pts)
        structure_score = self._assess_structure_quality(symbol, entry_price, direction)

        # 3️⃣ Momentum quality (25 pts)
        momentum_score = self._assess_momentum_quality(symbol, direction)

        # 4️⃣ Risk‑/Reward quality (25 pts)
        rr_score = self._assess_risk_reward_quality(symbol, entry_price, stop_loss, direction)

        total_score = trend_score + structure_score + momentum_score + rr_score
        should_trade = total_score >= self.min_quality_score

        # ------------------------------------------------------------------
        # Human‑readable reasoning (useful for logs / audit)
        # ------------------------------------------------------------------
        if total_score >= self.GRADE_A_PLUS:
            grade = "A+"
        elif total_score >= self.GRADE_A:
            grade = "A"
        elif total_score >= self.GRADE_B:
            grade = "B"
        elif total_score >= self.GRADE_C:
            grade = "C"
        else:
            grade = "F"

        reasoning = (
            f"Grade: {grade}\n"
            f"Trend: {trend_score:.1f}/25\n"
            f"Structure: {structure_score:.1f}/25\n"
            f"Momentum: {momentum_score:.1f}/25\n"
            f"Risk/Reward: {rr_score:.1f}/25"
        )

        return EntryQualityScore(
            total_score=total_score,
            trend_score=trend_score,
            structure_score=structure_score,
            momentum_score=momentum_score,
            risk_reward_score=rr_score,
            should_trade=should_trade,
            reasoning=reasoning,
        )

    # ------------------------------------------------------------------
    # COMPONENT 1 – Trend quality (25 pts)
    # ------------------------------------------------------------------
    def _assess_trend_quality(self, symbol: str, direction: str) -> float:
        """
        Trend quality is a weighted blend of:

        * EMA alignment (10 pts)
        * ADX‑derived trend strength (10 pts)
        * Recent candle consistency (5 pts)
        """
        # ------------------------------------------------------------------
        # 1️⃣ EMA alignment (0‑1 → *10)
        # ------------------------------------------------------------------
        rates_h1 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 100)
        if rates_h1 is None:
            logger.warning(f"Unable to fetch H1 rates for {symbol} – trend score defaulted")
            ema_score = 0.5
        else:
            ema_score = self._check_ema_alignment(rates_h1, direction)

        # ------------------------------------------------------------------
        # 2️⃣ ADX strength (0‑1 → *10)
        # ------------------------------------------------------------------
        adx = self._calculate_adx(symbol, mt5.TIMEFRAME_H1)
        if adx >= 40:
            adx_score = 1.0
        elif adx >= 30:
            adx_score = 0.8
        elif adx >= 25:
            adx_score = 0.5
        elif adx >= 20:
            adx_score = 0.2
        else:
            adx_score = 0.0

        # ------------------------------------------------------------------
        # 3️⃣ Candle‑trend consistency (0‑1 → *5)
        # ------------------------------------------------------------------
        consistency = self._check_trend_consistency(symbol, direction)

        # ------------------------------------------------------------------
        # Combine
        # ------------------------------------------------------------------
        trend_score = ema_score * 10 + adx_score * 10 + consistency * 5
        return min(trend_score, 25.0)

    def _check_ema_alignment(self, rates: np.ndarray, direction: str) -> float:
        """EMA‑20 vs EMA‑50 vs price alignment (0‑1)."""
        closes = rates["close"]
        ema_20 = np.mean(closes[-20:])
        ema_50 = np.mean(closes[-50:])
        price = closes[-1]

        if direction == "BUY":
            if price > ema_20 > ema_50:
                return 1.0
            if price > ema_20:
                return 0.6
            return 0.2
        else:  # SELL
            if price < ema_20 < ema_50:
                return 1.0
            if price < ema_20:
                return 0.6
            return 0.2

    def _calculate_adx(self, symbol: str, timeframe) -> float:
        """Simple ADX implementation (period = 14)."""
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 50)
        if rates is None or len(rates) < 28:
            return 20.0  # fallback moderate value

        period = 14
        highs = rates["high"]
        lows = rates["low"]
        closes = rates["close"]

        # True range
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1]),
            ),
        )

        # Directional movement
        plus_dm = np.maximum(highs[1:] - highs[:-1], 0)
        minus_dm = np.maximum(lows[:-1] - lows[1:], 0)

        atr = np.mean(tr[-period:])
        if atr == 0:
            return 0.0

        plus_di = 100.0 * np.mean(plus_dm[-period:]) / atr
        minus_di = 100.0 * np.mean(minus_dm[-period:]) / atr

        dx = (
            abs(plus_di - minus_di) / (plus_di + minus_di) * 100.0
            if (plus_di + minus_di) > 0
            else 0.0
        )
        return dx

    def _check_trend_consistency(self, symbol: str, direction: str) -> float:
        """Proportion of last 10 H1 candles that agree with the direction (0‑1)."""
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 10)
        if rates is None or len(rates) == 0:
            return 0.5

        bullish = sum(1 for r in rates if r["close"] > r["open"])
        bearish = sum(1 for r in rates if r["close"] < r["open"])

        if direction == "BUY":
            return bullish / len(rates)
        else:
            return bearish / len(rates)

    # ------------------------------------------------------------------
    # COMPONENT 2 – Structure quality (25 pts)
    # ------------------------------------------------------------------
    def _assess_structure_quality(self, symbol: str, entry_price: float, direction: str) -> float:
        """
        Structure quality = 15 pts (entry location) + 10 pts (S/R clarity)
        """
        loc_score = self._assess_structure_location(symbol, entry_price, direction)
        sr_score = self._assess_sr_quality(symbol, entry_price, direction)

        return min(loc_score * 15 + sr_score * 10, 25.0)

    def _assess_structure_location(self, symbol: str, entry_price: float, direction: str) -> float:
        """
        Returns a 0‑1 value describing how well the entry sits within the
        most recent swing range (H4 timeframe).
        """
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H4, 0, 100)
        if rates is None or len(rates) < 20:
            return 0.5

        highs = rates["high"]
        lows = rates["low"]
        recent_high = np.max(highs[-20:])
        recent_low = np.min(lows[-20:])
        swing = recent_high - recent_low
        if swing == 0:
            return 0.5

        if direction == "BUY":
            # distance from low as a fraction of swing
            pct = (entry_price - recent_low) / swing
            if pct <= 0.15:
                return 1.0
            if pct <= 0.30:
                return 0.8
            if pct <= 0.50:
                return 0.5
            return 0.2
        else:  # SELL
            pct = (recent_high - entry_price) / swing
            if pct <= 0.15:
                return 1.0
            if pct <= 0.30:
                return 0.8
            if pct <= 0.50:
                return 0.5
            return 0.2

    def _assess_sr_quality(self, symbol: str, entry_price: float, direction: str) -> float:
        """
        Checks how “clean” the nearest support/resistance level is.
        Returns 0‑1.
        """
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H4, 0, 50)
        if rates is None:
            return 0.5

        tolerance = entry_price * 0.002  # 0.2 %
        touches = 0
        for r in rates:
            if abs(r["high"] - entry_price) <= tolerance or abs(r["low"] - entry_price) <= tolerance:
                touches += 1

        if touches >= 3:
            return 1.0
        if touches == 2:
            return 0.7
        if touches == 1:
            return 0.5
        return 0.3

    # ------------------------------------------------------------------
    # COMPONENT 3 – Momentum quality (25 pts)
    # ------------------------------------------------------------------
    def _assess_momentum_quality(self, symbol: str, direction: str) -> float:
        """
        Momentum = 10 pts (RSI) + 10 pts (MACD) + 5 pts (Volume)
        """
        rsi = self._calculate_rsi(symbol, mt5.TIMEFRAME_H1)
        macd_ok = self._check_macd_alignment(symbol, direction)
        vol_ok = self._check_volume_confirmation(symbol, direction)

        rsi_score = 0.0
        if 40 <= rsi <= 60:
            rsi_score = 1.0
        elif 30 <= rsi < 40 or 60 < rsi <= 70:
            rsi_score = 0.8
        elif 20 <= rsi < 30 or 70 < rsi <= 80:
            rsi_score = 0.5
        else:
            rsi_score = 0.2

        return min(rsi_score * 10 + macd_ok * 10 + vol_ok * 5, 25.0)

    def _calculate_rsi(self, symbol: str, timeframe, period: int = 14) -> float:
        """Classic 14‑period RSI."""
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, period + 50)
        if rates is None or len(rates) < period + 1:
            return 50.0

        closes = rates["close"]
        deltas = np.diff(closes)

        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    def _check_macd_alignment(self, symbol: str, direction: str) -> float:
        """Very light MACD check – returns 1.0 if MACD sign matches direction."""
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 50)
        if rates is None or len(rates) < 35:
            return 0.5

        closes = rates["close"]
        ema_12 = np.mean(closes[-12:])
        ema_26 = np.mean(closes[-26:])
        macd = ema_12 - ema_26

        if direction == "BUY" and macd > 0:
            return 1.0
        if direction == "SELL" and macd < 0:
            return 1.0
        return 0.3

    def _check_volume_confirmation(self, symbol: str, direction: str) -> float:
        """Volume compared to its 10‑bar average (0‑1)."""
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 20)
        if rates is None or len(rates) < 10:
            return 0.5

        vols = rates["tick_volume"]
        cur_vol = vols[-1]
        avg_vol = np.mean(vols[-10:])

        if cur_vol >= avg_vol * 1.5:
            return 1.0
        if cur_vol >= avg_vol * 1.2:
            return 0.8
        if cur_vol >= avg_vol:
            return 0.6
        return 0.4

     # ------------------------------------------------------------------
    # COMPONENT 4 – Risk/Reward quality (continued)
    # ------------------------------------------------------------------
    def _assess_risk_reward_quality(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        direction: str,
    ) -> float:
        """
        R/R quality = 15 pts (RR ratio) + 10 pts (SL placement)
        """
        # ---- 1️⃣  RR ratio (0‑1 → *15) ----
        risk = abs(entry_price - stop_loss)

        if direction == "BUY":
            target = self._find_next_resistance(symbol, entry_price)
        else:
            target = self._find_next_support(symbol, entry_price)

        if target is None:
            # No clear target – assume a modest 1.5 : 1 ratio
            rr_ratio = 1.5
        else:
            reward = abs(target - entry_price)
            rr_ratio = reward / risk if risk > 0 else 0.0

        # ----- score the RR ratio (0‑1) -----
        if rr_ratio >= 3.0:
            rr_score = 1.0
        elif rr_ratio >= 2.5:
            rr_score = 0.8
        elif rr_ratio >= 2.0:
            rr_score = 0.5
        elif rr_ratio >= 1.5:
            rr_score = 0.3
        else:
            rr_score = 0.1

        # ---- 2️⃣  Stop‑loss placement quality (0‑1 → *10) ----
        sl_quality = self._assess_stop_loss_quality(
            symbol, entry_price, stop_loss, direction
        )

        return min(rr_score * 15 + sl_quality * 10, 25.0)

    # ------------------------------------------------------------------
    # Helper – find the nearest significant resistance above entry
    # ------------------------------------------------------------------
    def _find_next_resistance(self, symbol: str, entry_price: float) -> Optional[float]:
        """
        Scan the H4 timeframe for swing‑highs above ``entry_price``.
        Returns the *nearest* such level or ``None`` if none are found.
        """
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H4, 0, 200)
        if rates is None or len(rates) < 30:
            return None

        highs = rates["high"]
        # Detect swing highs (simple 2‑bar peak)
        swing_highs = [
            highs[i]
            for i in range(2, len(highs) - 2)
            if highs[i] > highs[i - 1]
            and highs[i] > highs[i - 2]
            and highs[i] > highs[i + 1]
            and highs[i] > highs[i + 2]
            and highs[i] > entry_price
        ]

        return min(swing_highs) if swing_highs else None

    # ------------------------------------------------------------------
    # Helper – find the nearest significant support below entry
    # ------------------------------------------------------------------
    def _find_next_support(self, symbol: str, entry_price: float) -> Optional[float]:
        """
        Scan the H4 timeframe for swing‑lows below ``entry_price``.
        Returns the *nearest* such level or ``None`` if none are found.
        """
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H4, 0, 200)
        if rates is None or len(rates) < 30:
            return None

        lows = rates["low"]
        swing_lows = [
            lows[i]
            for i in range(2, len(lows) - 2)
            if lows[i] < lows[i - 1]
            and lows[i] < lows[i - 2]
            and lows[i] < lows[i + 1]
            and lows[i] < lows[i + 2]
            and lows[i] < entry_price
        ]

        return max(swing_lows) if swing_lows else None

    # ------------------------------------------------------------------
    # Helper – assess stop‑loss placement quality (0‑1)
    # ------------------------------------------------------------------
    def _assess_stop_loss_quality(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        direction: str,
    ) -> float:
        """
        Compare the absolute SL distance to the recent ATR.
        Ideal SL = 1‑2 × ATR → score 1.0.
        """
        risk = abs(entry_price - stop_loss)

        # Get recent H1 bars for ATR
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 30)
        if rates is None or len(rates) < 14:
            return 0.5  # fallback neutral score

        atr = self._calculate_atr(rates, period=14)
        if atr == 0:
            return 0.5

        sl_in_atr = risk / atr

        if 1.0 <= sl_in_atr <= 2.0:
            return 1.0
        if 0.8 <= sl_in_atr < 1.0 or 2.0 < sl_in_atr <= 2.5:
            return 0.7
        if 0.5 <= sl_in_atr < 0.8 or 2.5 < sl_in_atr <= 3.0:
            return 0.5
        return 0.3

    # ------------------------------------------------------------------
    # Helper – simple ATR calculation
    # ------------------------------------------------------------------
    def _calculate_atr(self, rates: np.ndarray, period: int = 14) -> float:
        """
        Average True Range over ``period`` bars.
        """
        if len(rates) < period + 1:
            return 0.0

        highs = rates["high"]
        lows = rates["low"]
        closes = rates["close"]

        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1]),
            ),
        )
        return float(np.mean(tr[-period:]))

    # ------------------------------------------------------------------
    # OPTIONAL: a tiny demo when the file is run directly
    # ------------------------------------------------------------------
    def demo(self, symbol: str = "EURUSD"):
        """
        Run a quick self‑test on the supplied symbol and print the result.
        """
        # Example parameters – in a real bot these come from the signal generator
        direction = "BUY"
        entry_price = mt5.symbol_info_tick(symbol).ask
        stop_loss = entry_price - 0.0010  # 10 pips SL for demo

        score = self.assess_entry_quality(symbol, direction, entry_price, stop_loss)
        logger.info("\n" + "=" * 60)
        logger.info(f"Entry quality for {symbol} {direction} @ {entry_price:.5f}")
        logger.info(score.reasoning)
        logger.info(f"TOTAL SCORE: {score.total_score:.1f} – {'TAKE' if score.should_trade else 'SKIP'}")
        logger.info("=" * 60)


# ----------------------------------------------------------------------
# Global singleton – import this from ``src/main.py`` and use it directly
# ----------------------------------------------------------------------
entry_quality_filter = EntryQualityFilter(min_quality_score=95)

# ----------------------------------------------------------------------
# Simple command‑line demo (useful during development)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise MT5 (required before any MT5 call)
    if not mt5.initialize():
        raise RuntimeError(f"MT5 init failed – {mt5.last_error()}")

    # Run demo for the default symbol
    entry_quality_filter.demo("EURUSD")

    # Shut down MT5 cleanly
    mt5.shutdown()

# src/advanced_entry_filter.py  – at the bottom of the file
from src.config_loader import load_config

# Load the config once at import time (or later, if you prefer lazy init)
_cfg = load_config()
entry_quality_filter = EntryQualityFilter(_cfg)   # <-- here we pass the dict

def _assess_risk_reward_quality(
    self,
    symbol: str,
    entry_price: float,
    stop_loss: float,
    direction: str,
) -> float:
    """
    Risk/Reward = RR ratio (15 pts) + SL placement (10 pts)
    All thresholds are configurable.
    """

    # ------------------------------------------------------------------
    # Configurable RR targets & SL‑ATR ranges
    # ------------------------------------------------------------------
    rr_cfg = self.risk_reward_cfg
    rr_target = rr_cfg.get("rr_target", 3.0)          # full 15‑pt score at ≥ 3.0
    rr_good   = rr_cfg.get("rr_good", 2.5)
    rr_ok     = rr_cfg.get("rr_ok", 2.0)

    sl_atr_min = rr_cfg.get("sl_atr_min", 1.0)        # 1 × ATR
    sl_atr_max = rr_cfg.get("sl_atr_max", 2.0)        # 2 × ATR

    # ------------------------------------------------------------------
    # 1️⃣  RR ratio (0‑1)
    # ------------------------------------------------------------------
    risk = abs(entry_price - stop_loss)

    if direction == "BUY":
        target = self._find_next_resistance(symbol, entry_price)
    else:
        target = self._find_next_support(symbol, entry_price)

    if target is None:
        rr_ratio = 1.5                     # fallback modest ratio
    else:
        reward = abs(target - entry_price)
        rr_ratio = reward / risk if risk > 0 else 0.0

    if rr_ratio >= rr_target:
        rr_score = 1.0
    elif rr_ratio >= rr_good:
        rr_score = 0.8
    elif rr_ratio >= rr_ok:
        rr_score = 0.5
    else:
        rr_score = 0.2

    # ------------------------------------------------------------------
    # 2️⃣  SL‑ATR quality (0‑1)
    # ------------------------------------------------------------------
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 30)
    atr = self._calculate_atr(rates, period=14) if rates is not None else 0.0

    if atr == 0:
        sl_score = 0.5                     # neutral fallback
    else:
        sl_in_atr = risk / atr
        if sl_atr_min <= sl_in_atr <= sl_atr_max:
            sl_score = 1.0
        elif (sl_atr_min * 0.8) <= sl_in_atr < sl_atr_min or sl_atr_max < sl_in_atr <= (sl_atr_max * 1.25):
            sl_score = 0.7
        elif (sl_atr_min * 0.5) <= sl_in_atr < (sl_atr_min * 0.8) or (sl_atr_max * 1.25) < sl_in_atr <= (sl_atr_max * 1.5):
            sl_score = 0.5
        else:
            sl_score = 0.3

    # ------------------------------------------------------------------
    # 3️⃣  Weighted total (15 pts for RR, 10 pts for SL)
    # ------------------------------------------------------------------
    total = rr_score * 15 + sl_score * 10
    return min(total, 25.0)

def _assess_momentum_quality(self, symbol: str, direction: str) -> float:
    """
    Momentum = RSI (10 pts) + MACD (10 pts) + Volume (5 pts)
    All thresholds are now configurable.
    """

    # ------------------------------------------------------------------
    # Pull configurable thresholds
    # ------------------------------------------------------------------
    rsi_cfg = self.momentum_cfg.get("rsi", {})
    rsi_neutral_low  = rsi_cfg.get("neutral_low", 40)
    rsi_neutral_high = rsi_cfg.get("neutral_high", 60)
    rsi_good_low     = rsi_cfg.get("good_low", 30)
    rsi_good_high    = rsi_cfg.get("good_high", 70)

    macd_weight = self.momentum_cfg.get("macd_weight", 10) / 10.0   # normalise to 0‑1
    vol_weight  = self.momentum_cfg.get("volume_weight", 5) / 5.0

    # ------------------------------------------------------------------
    # 1️⃣  RSI score (0‑1)
    # ------------------------------------------------------------------
    rsi = self._calculate_rsi(symbol, mt5.TIMEFRAME_H1)
    if rsi_neutral_low <= rsi <= rsi_neutral_high:
        rsi_score = 1.0
    elif rsi_good_low <= rsi < rsi_neutral_low or rsi_neutral_high < rsi <= rsi_good_high:
        rsi_score = 0.8
    elif (rsi_good_low - 10) <= rsi < rsi_good_low or rsi_good_high < rsi <= (rsi_good_high + 10):
        rsi_score = 0.5
    else:
        rsi_score = 0.2

    # ------------------------------------------------------------------
    # 2️⃣  MACD alignment (still 0‑1)
    # ------------------------------------------------------------------
    macd_ok = self._check_macd_alignment(symbol, direction)

    # ------------------------------------------------------------------
    # 3️⃣  Volume confirmation (still 0‑1)
    # ------------------------------------------------------------------
    vol_ok = self._check_volume_confirmation(symbol, direction)

    # ------------------------------------------------------------------
    # 4️⃣  Weighted combination
    # ------------------------------------------------------------------
    momentum_score = (
        rsi_score * 10
        + macd_ok * macd_weight * 10
        + vol_ok  * vol_weight * 5
    )
    return min(momentum_score, 25.0) 

def _assess_structure_quality(self, symbol: str, entry_price: float, direction: str) -> float:
    """
    Structure quality = entry‑location (15 pts) + S/R clarity (10 pts)
    Both sub‑scores are multiplied by configurable weights.
    """

    # ------------------------------------------------------------------
    # Configurable weights (default 15 pts & 10 pts → normalise to 0‑1)
    # ------------------------------------------------------------------
    loc_weight = self.structure_cfg.get("location_weight", 15) / 25.0
    sr_weight  = self.structure_cfg.get("sr_weight", 10) / 25.0

    # ------------------------------------------------------------------
    # 1️⃣  Entry‑location score (still 0‑1 internally)
    # ------------------------------------------------------------------
    loc_score_raw = self._assess_structure_location(symbol, entry_price, direction)

    # ------------------------------------------------------------------
    # 2️⃣  Support/Resistance clarity score (still 0‑1)
    # ------------------------------------------------------------------
    sr_score_raw = self._assess_sr_quality(symbol, entry_price, direction)

    # ------------------------------------------------------------------
    # 3️⃣  Apply the weights
    # ------------------------------------------------------------------
    total = (
        loc_score_raw * loc_weight * 25
        + sr_score_raw  * sr_weight * 25
    )
    return min(total, 25.0) 

structure:
  swing_lookback: 20          # how many H4 bars we scan for swing highs/lows
  tolerance_pct: 0.002        # 0.2 % price band around entry

lookback = self.structure_cfg.get("swing_lookback", 20)
tolerance = self.structure_cfg.get("tolerance_pct", 0.002)

rates_h4 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H4, 0, lookback)
# … use `tolerance` when checking distance from swing high/low …

def _assess_trend_quality(self, symbol: str, direction: str) -> float:
    """
    Trend quality = EMA alignment (weight‑controlled) +
                    ADX strength (weight‑controlled) +
                    Recent‑candle consistency (weight‑controlled)
    Returns a 0‑25 score.
    """

    # ------------------------------------------------------------------
    # 1️⃣  Pull the configurable weights (they sum to 25 points)
    # ------------------------------------------------------------------
    ema_weight = self.trend_cfg.get("ema_weight", 10) / 25.0          # 0‑1
    adx_weight = self.trend_cfg.get("adx_weight", 10) / 25.0          # 0‑1
    cons_weight = self.trend_cfg.get("consistency_weight", 5) / 25.0 # 0‑1

    # ------------------------------------------------------------------
    # 2️⃣  EMA alignment (still 0‑1 internally)
    # ------------------------------------------------------------------
    rates_h1 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 100)
    if rates_h1 is None:
        ema_score = 0.5                     # neutral fallback
    else:
        ema_score = self._check_ema_alignment(rates_h1, direction)

    # ------------------------------------------------------------------
    # 3️⃣  ADX strength – thresholds are now configurable
    # ------------------------------------------------------------------
    adx = self._calculate_adx(symbol, mt5.TIMEFRAME_H1)

    th = self.trend_cfg.get("adx_thresholds", {})
    strong = th.get("strong", 40)
    medium = th.get("medium", 30)
    weak   = th.get("weak", 20)

    if adx >= strong:
        adx_score = 1.0
    elif adx >= medium:
        adx_score = 0.8
    elif adx >= weak:
        adx_score = 0.5
    else:
        adx_score = 0.2

    # ------------------------------------------------------------------
    # 4️⃣  Candle‑consistency (unchanged – still 0‑1)
    # ------------------------------------------------------------------
    consistency = self._check_trend_consistency(symbol, direction)

    # ------------------------------------------------------------------
    # 5️⃣  Combine using the **weights** we pulled above
    # ------------------------------------------------------------------
    trend_score = (
        ema_score * ema_weight * 25
        + adx_score * adx_weight * 25
        + consistency * cons_weight * 25
    )
    return min(trend_score, 25.0)
