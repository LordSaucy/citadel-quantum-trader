#!/usr/bin/env python3
"""
CITADEL QUANTUM TRADER – MARKET REGIME DETECTOR

Lever #3 of the 7‑lever win‑rate optimisation system.
Detects the prevailing market condition and returns a rich
`RegimeAnalysis` object that the bot can use to adapt strategy,
risk‑size and trade‑frequency.

Author: Lawful Banker
Created: 2024‑11‑26
Version: 2.0 – Production‑ready
"""

# ----------------------------------------------------------------------
# Standard library
# ----------------------------------------------------------------------
import logging
from datetime import datetime, date, time as dt_time, timedelta
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# ----------------------------------------------------------------------
# Third‑party
# ----------------------------------------------------------------------
import MetaTrader5 as mt5
import numpy as np

# ----------------------------------------------------------------------
# Logging configuration (writes to ./logs/market_regime.log)
# ----------------------------------------------------------------------
LOG_DIR = "./logs"
import pathlib
pathlib.Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"{LOG_DIR}/market_regime.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Market‑regime enumeration
# ----------------------------------------------------------------------
class MarketRegime(Enum):
    """Canonical market‑regime identifiers."""
    STRONG_TREND_UP   = "strong_trend_up"
    TREND_UP          = "trend_up"
    WEAK_TREND_UP     = "weak_trend_up"
    RANGING           = "ranging"
    WEAK_TREND_DOWN   = "weak_trend_down"
    TREND_DOWN        = "trend_down"
    STRONG_TREND_DOWN = "strong_trend_down"
    VOLATILE          = "volatile"   # high volatility, no clear direction
    BREAKOUT          = "breakout"   # breaking out of a range


# ----------------------------------------------------------------------
# Dataclass that bundles the analysis result
# ----------------------------------------------------------------------
@dataclass
class RegimeAnalysis:
    """All information the bot needs to adapt to the current regime."""
    regime: MarketRegime
    confidence: float                 # 0‑100 %
    trend_strength: float             # -100 → +100 (negative = down)
    volatility_state: str             # LOW / SUBDUED / NORMAL / ELEVATED / HIGH
    recommended_strategy: str
    recommended_risk_adjustment: float   # 0.5 → 1.5 multiplier
    trade_frequency_adjustment: str      # INCREASE / NORMAL / DECREASE


# ----------------------------------------------------------------------
# Core detector class
# ----------------------------------------------------------------------
class MarketRegimeDetector:
    """
    Detects market regime on‑the‑fly using a blend of classic technical
    indicators (ADX, ATR, EMA slopes, Bollinger‑Band width) and simple
    price‑structure heuristics (higher highs / higher lows, etc.).
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self) -> None:
        self.regime_history: List[Dict] = []          # for optional audit trail
        self.current_regime: Optional[MarketRegime] = None
        logger.info("MarketRegimeDetector initialised")

    # ------------------------------------------------------------------
    # Public entry point – called by the bot for every new symbol
    # ------------------------------------------------------------------
    def analyze_market_regime(self, symbol: str) -> RegimeAnalysis:
        """
        Perform a full‑stack regime analysis for ``symbol``.
        Returns a populated ``RegimeAnalysis`` instance.
        """
        # ----------------------------------------------------------------
        # 1️⃣  Pull recent multi‑time‑frame price data
        # ----------------------------------------------------------------
        h4 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H4, 0, 200)
        h1 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 200)

        if h4 is None or h1 is None or len(h4) < 50 or len(h1) < 50:
            logger.warning(f"Insufficient data for {symbol} – falling back to default")
            return self._default_regime()

        # ----------------------------------------------------------------
        # 2️⃣  Trend detection (direction & strength)
        # ----------------------------------------------------------------
        trend_dir, trend_strength = self._detect_trend(h4, h1)

        # ----------------------------------------------------------------
        # 3️⃣  Volatility assessment
        # ----------------------------------------------------------------
        volatility_state, volatility_pct = self._analyze_volatility(symbol, h4)

        # ----------------------------------------------------------------
        # 4️⃣  Market‑structure inspection (HH/HL/LH/LL)
        # ----------------------------------------------------------------
        structure = self._analyze_market_structure(h4)

        # ----------------------------------------------------------------
        # 5️⃣  Regime classification
        # ----------------------------------------------------------------
        regime = self._classify_regime(
            trend_dir, trend_strength, volatility_state, structure
        )

        # ----------------------------------------------------------------
        # 6️⃣  Confidence scoring
        # ----------------------------------------------------------------
        confidence = self._calculate_confidence(trend_strength, structure)

        # ----------------------------------------------------------------
        # 7️⃣  Strategy & risk recommendations
        # ----------------------------------------------------------------
        strategy = self._recommend_strategy(regime)
        risk_adj = self._recommend_risk_adjustment(regime, volatility_state)
        freq_adj = self._recommend_frequency(regime)

        # ----------------------------------------------------------------
        # 8️⃣  Pack everything into the dataclass
        # ----------------------------------------------------------------
        analysis = RegimeAnalysis(
            regime=regime,
            confidence=confidence,
            trend_strength=trend_strength,
            volatility_state=volatility_state,
            recommended_strategy=strategy,
            recommended_risk_adjustment=risk_adj,
            trade_frequency_adjustment=freq_adj,
        )

        # ----------------------------------------------------------------
        # 9️⃣  Store for history & expose as current regime
        # ----------------------------------------------------------------
        self.regime_history.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "symbol": symbol,
                "regime": regime.value,
                "confidence": confidence,
            }
        )
        self.current_regime = regime

        logger.info(
            f"[{symbol}] Regime={regime.value} | "
            f"Confidence={confidence:.1f}% | "
            f"Strategy={strategy} | RiskAdj={risk_adj:.2f}x"
        )
        return analysis

    # ------------------------------------------------------------------
    # 1️⃣  Trend detection (EMA alignment + ADX + EMA‑slope combo)
    # ------------------------------------------------------------------
    def _detect_trend(
        self, h4: np.ndarray, h1: np.ndarray
    ) -> Tuple[float, float]:
        """
        Returns ``(direction, strength)`` where ``direction`` ∈ {‑1, 0, +1}
        and ``strength`` ∈ [0, 100].
        """
        # ----- EMA alignment (H4) -----
        h4_close = h4["close"]
        ema20 = np.mean(h4_close[-20:])
        ema50 = np.mean(h4_close[-50:])
        ema100 = np.mean(h4_close[-100:]) if len(h4_close) >= 100 else ema50
        price = h4_close[-1]

        if price > ema20 > ema50 > ema100:
            ema_score = 100
        elif price > ema20 > ema50:
            ema_score = 70
        elif price > ema20:
            ema_score = 40
        elif price < ema20 < ema50 < ema100:
            ema_score = -100
        elif price < ema20 < ema50:
            ema_score = -70
        elif price < ema20:
            ema_score = -40
        else:
            ema_score = 0

        # ----- ADX (trend strength) -----
        adx = self._calculate_adx(h4)

        # ----- EMA‑slope analysis (H4) -----
        slope_score = self._analyze_ema_slopes(h4_close)

        # ----- Combine -----
        direction = np.sign(ema_score + slope_score)          # -1 / 0 / +1
        strength = min(100.0, (abs(ema_score) + adx) / 2.0)   # 0‑100

        return direction, strength

    # ------------------------------------------------------------------
    # 2️⃣  Volatility analysis (ATR‑based)
    # ------------------------------------------------------------------
    def _analyze_volatility(
        self, symbol: str, h4: np.ndarray
    ) -> Tuple[str, float]:
        """
        Returns ``(state, pct)`` where ``state`` ∈
        {LOW, SUBDUED, NORMAL, ELEVATED, HIGH}.
        """
        atr_14 = self._calculate_atr_from_data(h4, period=14)
        atr_50 = self._calculate_atr_from_data(h4, period=50)

        if atr_50 == 0:
            return "NORMAL", 100.0

        ratio = atr_14 / atr_50

        if ratio >= 1.5:
            state, pct = "HIGH", min(200, ratio * 100)
        elif ratio >= 1.2:
            state, pct = "ELEVATED", ratio * 100
        elif ratio >= 0.8:
            state, pct = "NORMAL", 100.0
        elif ratio >= 0.6:
            state, pct = "SUBDUED", ratio * 100
        else:
            state, pct = "LOW", ratio * 100

        return state, pct

    # ------------------------------------------------------------------
    # 3️⃣  Market‑structure inspection (higher highs / higher lows)
    # ------------------------------------------------------------------
    def _analyze_market_structure(self, h4: np.ndarray) -> Dict:
        """Detects HH/HL/LH/LL patterns and returns a quality score."""
        highs = h4["high"]
        lows = h4["low"]

        swing_highs: List[Tuple[int, float]] = []
        swing_lows: List[Tuple[int, float]] = []

        # simple 5‑bar swing detection
        for i in range(5, len(highs) - 5):
            if highs[i] == np.max(highs[i - 5 : i + 5]):
                swing_highs.append((i, highs[i]))
            if lows[i] == np.min(lows[i - 5 : i + 5]):
                swing_lows.append((i, lows[i]))

        higher_highs = higher_lows = lower_highs = lower_lows = False
        if len(swing_highs) >= 2:
            higher_highs = swing_highs[-1][1] > swing_highs[-2][1]
            lower_highs = swing_highs[-1][1] < swing_highs[-2][1]
        if len(swing_lows) >= 2:
            higher_lows = swing_lows[-1][1] > swing_lows[-2][1]
            lower_lows = swing_lows[-1][1] < swing_lows[-2][1]

        # quality heuristic
        if higher_highs and higher_lows:
            quality = 90
        elif lower_highs and lower_lows:
            quality = 90
        elif higher_highs or higher_lows or lower_highs or lower_lows:
            quality = 50
        else:
            quality = 30

        return {
            "higher_highs": higher_highs,
            "higher_lows": higher_lows,
            "lower_highs": lower_highs,
            "lower_lows": lower_lows,
            "structure_quality": quality,
        }

    # ------------------------------------------------------------------
    # 4️⃣  Regime classification – combines all signals
    # ------------------------------------------------------------------
    def _classify_regime(
        self,
        trend_dir: float,
        trend_strength: float,
        volatility_state: str,
        structure: Dict,
    ) -> MarketRegime:
        """Maps the raw signals onto a canonical regime."""
        # Ranging / volatile first
        if trend_strength < 20:
            if volatility_state in ("HIGH", "ELEVATED"):
                return MarketRegime.VOLATILE
            return MarketRegime.RANGING

        # Strong trends
        if trend_dir > 0:  # up‑trend
            if trend_strength >= 70:
                return MarketRegime.STRONG_TREND_UP
            if trend_strength >= 40:
                return MarketRegime.TREND_UP
            return MarketRegime.WEAK_TREND_UP
        else:  # down‑trend
            if trend_strength >= 70:
                return MarketRegime.STRONG_TREND_DOWN
            if trend_strength >= 40:
                return MarketRegime.TREND_DOWN
            return MarketRegime.WEAK_TREND_DOWN

    # ------------------------------------------------------------------
    # 5️⃣  Confidence scoring
    # ------------------------------------------------------------------
    def _calculate_confidence(self, trend_strength: float, structure: Dict) -> float:
        """Blend trend strength and structure quality into a 0‑100 confidence."""
        conf = trend_strength * 0.6 + structure["structure_quality"] * 0.4
        return min(100.0, conf)

    # ------------------------------------------------------------------
    # 6️⃣  Strategy recommendation
    # ------------------------------------------------------------------
    def _recommend_strategy(self, regime: MarketRegime) -> str:
        """Human‑readable strategy hint for the given regime."""
        mapping = {
            MarketRegime.STRONG_TREND_UP: "Trend‑following – BUY only, wide targets",
            MarketRegime.TREND_UP: "Trend‑following – BUY focus, pull‑back entries",
            MarketRegime.WEAK_TREND_UP: "Cautious BUY – require strong confluence",
            MarketRegime.RANGING: "Mean‑reversion – fade extremes",
            MarketRegime.WEAK_TREND_DOWN: "Cautious SELL – require strong confluence",
            MarketRegime.TREND_DOWN: "Trend‑following – SELL focus, rally entries",
            MarketRegime.STRONG_TREND_DOWN: "Trend‑following – SELL only, wide targets",
            MarketRegime.VOLATILE: "Reduce activity – only A+ setups",
            MarketRegime.BREAKOUT: "Breakout trades – tight stops, wide targets",
        }
        return mapping.get(regime, "Standard strategy")

    # ------------------------------------------------------------------
    # 7️⃣  Risk‑adjustment recommendation (multiplier)
    # ------------------------------------------------------------------
    def _recommend_risk_adjustment(
        self, regime: MarketRegime, volatility_state: str
    ) -> float:
        """Return a multiplier in the range 0.5 – 1.5."""
        base = {
            MarketRegime.STRONG_TREND_UP: 1.30,
            MarketRegime.TREND_UP: 1.20,
            MarketRegime.WEAK_TREND_UP: 0.90,
            MarketRegime.RANGING: 0.80,
            MarketRegime.WEAK_TREND_DOWN: 0.90,
            MarketRegime.TREND_DOWN: 1.20,
            MarketRegime.STRONG_TREND_DOWN: 1.30,
            MarketRegime.VOLATILE: 0.60,
            MarketRegime.BREAKOUT: 1.00,
        }.get(regime, 1.00)

        vol_adj = {
            "LOW": 1.10,
            "SUBDUED": 1.00,
            "NORMAL": 1.00,
            "ELEVATED": 0.90,
            "HIGH": 0.70,
        }.get(volatility_state, 1.00)

        final = base * vol_adj
        return max(0.5, min(1.5, final))

    # ------------------------------------------------------------------
    # 8️⃣  Trade‑frequency recommendation
    # ------------------------------------------------------------------
    def _recommend_frequency(self, regime: MarketRegime) -> str:
        """INCREASE / NORMAL / DECREASE."""
        if regime in (MarketRegime.STRONG_TREND_UP, MarketRegime.STRONG_TREND_DOWN):
            return "INCREASE"
        if regime == MarketRegime.RANGING:
            return "DECREASE"
        if regime == MarketRegime.VOLATILE:
            return "DECREASE"
        return "NORMAL"

      # ------------------------------------------------------------------
    # 9️⃣  Helper used by the bot to decide whether a signal aligns
    # ------------------------------------------------------------------
    def should_trade_in_current_regime(
        self, signal_direction: str
    ) -> Tuple[bool, str]:
        """
        Given a prospective trade direction (\"BUY\" / \"SELL\"), decide if it
        complies with the *current* regime.

        Returns ``(allowed, reason)`` where ``allowed`` is a boolean and
        ``reason`` explains the decision.
        """
        if self.current_regime is None:
            return True, "No regime information – allow trade"

        r = self.current_regime

        # ----------------------------------------------------------------
        # Strong trends – only trade *with* the trend
        # ----------------------------------------------------------------
        if r == MarketRegime.STRONG_TREND_UP and signal_direction != "BUY":
            return False, "Counter‑trend trade in strong uptrend"
        if r == MarketRegime.STRONG_TREND_DOWN and signal_direction != "SELL":
            return False, "Counter‑trend trade in strong downtrend"

        # ----------------------------------------------------------------
        # Volatile markets – be very selective; allow both sides but warn
        # ----------------------------------------------------------------
        if r == MarketRegime.VOLATILE:
            return True, "Volatile market – only high‑conviction setups"

        # ----------------------------------------------------------------
        # Ranging markets – favour mean‑reversion (sell at highs, buy at lows)
        # ----------------------------------------------------------------
        if r == MarketRegime.RANGING:
            # The bot can still trade both sides; we just note the preference
            return True, "Ranging market – prefer extremes"

        # ----------------------------------------------------------------
        # Weak trends – allow both directions but prefer the indicated trend
        # ----------------------------------------------------------------
        if r in (
            MarketRegime.TREND_UP,
            MarketRegime.WEAK_TREND_UP,
        ) and signal_direction != "BUY":
            return False, f"Signal {signal_direction} opposes up‑trend regime"
        if r in (
            MarketRegime.TREND_DOWN,
            MarketRegime.WEAK_TREND_DOWN,
        ) and signal_direction != "SELL":
            return False, f"Signal {signal_direction} opposes down‑trend regime"

        # ----------------------------------------------------------------
        # Breakout – allow any direction but expect momentum continuation
        # ----------------------------------------------------------------
        if r == MarketRegime.BREAKOUT:
            return True, "Breakout regime – any direction accepted"

        # ----------------------------------------------------------------
        # Default – allow trade
        # ----------------------------------------------------------------
        return True, "Regime permits trade"

    # ------------------------------------------------------------------
    # 10️⃣  EMA‑slope analysis (used by trend detection)
    # ------------------------------------------------------------------
    def _analyze_ema_slopes(self, closes: np.ndarray) -> float:
        """
        Compute a combined EMA‑slope score.
        Returns a value in the range ‑100 … +100 (positive = up‑trend).
        """
        if len(closes) < 50:
            return 0.0

        ema20_now = np.mean(closes[-20:])
        ema20_prev = np.mean(closes[-25:-5])

        ema50_now = np.mean(closes[-50:])
        ema50_prev = np.mean(closes[-55:-5])

        # slope in % per period
        slope20 = (ema20_now - ema20_prev) / ema20_prev * 100.0
        slope50 = (ema50_now - ema50_prev) / ema50_prev * 100.0

        # weighted combination (short‑term a bit more important)
        combined = slope20 * 0.6 + slope50 * 0.4

        # clip to -100 … +100
        return float(np.clip(combined, -100, 100))

    # ------------------------------------------------------------------
    # 11️⃣  ADX calculation (trend strength)
    # ------------------------------------------------------------------
    def _calculate_adx(self, rates: np.ndarray, period: int = 14) -> float:
        """
        Classic ADX implementation.
        Returns a value in the range 0‑100 (higher = stronger trend).
        """
        if len(rates) < period * 2:
            return 20.0  # default moderate value

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

        atr = np.mean(tr[-period:]) if np.mean(tr[-period:]) != 0 else 1.0
        plus_di = 100.0 * np.mean(plus_dm[-period:]) / atr
        minus_di = 100.0 * np.mean(minus_dm[-period:]) / atr

        if plus_di + minus_di == 0:
            return 0.0

        dx = 100.0 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = np.mean(dx)  # simple smoothing
        return float(adx)

    # ------------------------------------------------------------------
    # 12️⃣  ATR calculation (used for volatility)
    # ------------------------------------------------------------------
    def _calculate_atr_from_data(self, rates: np.ndarray, period: int = 14) -> float:
        """
        Average True Range over the last ``period`` bars.
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
    # 13️⃣  Default regime (fallback when data is missing)
    # ------------------------------------------------------------------
    def _default_regime(self) -> RegimeAnalysis:
        """Return a safe‑default (ranging) analysis."""
        return RegimeAnalysis(
            regime=MarketRegime.RANGING,
            confidence=50.0,
            trend_strength=0.0,
            volatility_state="NORMAL",
            recommended_strategy="Standard strategy",
            recommended_risk_adjustment=1.0,
            trade_frequency_adjustment="NORMAL",
        )


# ----------------------------------------------------------------------
# Global singleton – import this from ``src/main.py`` and use it directly
# ----------------------------------------------------------------------
market_regime_detector = MarketRegimeDetector()

def predict_regime(df: pd.DataFrame) -> str:
    # ... returns one of "trend", "range", "high_vol"

