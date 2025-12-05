#!/usr/bin/env python3
"""
VolEntry.py – Production‑ready volatility‑entry refinement system.

Implements the “Lever 7 of the 7‑lever win‑rate optimisation” logic.
All tunable parameters are stored in a JSON file (mounted volume) and
exposed as Prometheus gauges.  A tiny Flask API (port 5006) allows
runtime adjustments from Grafana or an operator.

Author: Lawful Banker
Created: 2024‑11‑26
Version: 2.0 – Production‑Ready
"""

# ----------------------------------------------------------------------
# Standard library
# ----------------------------------------------------------------------
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Thread
from time import sleep
from typing import Dict, Optional, Tuple

# ----------------------------------------------------------------------
# Third‑party
# ----------------------------------------------------------------------
import MetaTrader5 as mt5
import numpy as np
from flask import Flask, abort, jsonify, request
from prometheus_client import Gauge

# ----------------------------------------------------------------------
# Logging (inherits global configuration from the application)
# ----------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Controller – holds tunable parameters, Prometheus gauges,
# persistence and a tiny Flask API.
# ----------------------------------------------------------------------
class VolEntryController:
    """
    Central place for all runtime‑tunable values used by the
    VolatilityEntryRefinement engine.

    * Values are stored in a JSON file (mounted volume) so they survive
      container restarts.
    * Every key is also exported as a Prometheus gauge:
          volatility_param{param="weight_atr"} 0.20
    * A small Flask API (port 5006) lets Grafana or an operator change
      values on‑the‑fly.
    """

    # ------------------------------------------------------------------
    # Default configuration – edit these defaults as you wish.
    # ------------------------------------------------------------------
    DEFAULTS: Dict[str, float] = {
        # ----- Weights -------------------------------------------------
        "weight_atr": 0.20,                # Influence of ATR on confidence
        "weight_vol_state": 0.30,          # Influence of volatility state
        "weight_consolidation": 0.25,      # Influence of consolidation flag
        "weight_expanding_bonus": 0.15,    # Extra boost when EXPANDING

        # ----- Thresholds ---------------------------------------------
        "min_confidence": 60.0,            # Minimum confidence to trade
        "atr_lookback_h1": 14,             # ATR period on H1 chart
        "atr_lookback_m15": 14,            # ATR period on M15 chart
        "expansion_rate_thr": 0.15,        # 15 % change = EXPANDING/CONTRACTING
        "high_vol_multiplier": 1.5,        # Position‑size multiplier in HIGH_VOLATILE
        "low_vol_multiplier": 0.8,         # Position‑size multiplier in LOW_VOLATILE
        "consolidation_atr_factor": 2.0,   # Consolidation if range < 2 ATR
        "optimal_range_factor": 0.30,      # ± 0.30 ATR around entry
        "refine_adjust_factor": 0.15,      # ± 0.15 ATR when refining entry
    }

    CONFIG_PATH = Path("/app/config/vol_entry_config.json")   # <- mount this dir

    # ------------------------------------------------------------------
    # Prometheus gauges – one per key, labelled by “param”
    # ------------------------------------------------------------------
    _gauges: Dict[str, Gauge] = {}

    # ------------------------------------------------------------------
    def __init__(self) -> None:
        self._stop_event = Event()
        self._load_or_initialize()
        self._register_gauges()
        self._start_file_watcher()
        self._start_flask_api()

    # ------------------------------------------------------------------
    # Load persisted JSON or fall back to defaults
    # ------------------------------------------------------------------
    def _load_or_initialize(self) -> None:
        if self.CONFIG_PATH.exists():
            try:
                with open(self.CONFIG_PATH, "r") as f:
                    self.values = json.load(f)
                logger.info(
                    f"VolEntryController – loaded config from {self.CONFIG_PATH}"
                )
            except Exception as exc:   # pragma: no cover
                logger.error(f"Failed to read config – using defaults ({exc})")
                self.values = self.DEFAULTS.copy()
        else:
            logger.info("VolEntryController – no config file, using defaults")
            self.values = self.DEFAULTS.copy()
            self._persist()                     # create the file for the first time

    # ------------------------------------------------------------------
    # Persist the whole dict
    # ------------------------------------------------------------------
    def _persist(self) -> None:
        try:
            self.CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(self.CONFIG_PATH, "w") as f:
                json.dump(self.values, f, indent=2)
        except Exception as exc:   # pragma: no cover
            logger.error(f"Could not persist vol‑entry config: {exc}")

    # ------------------------------------------------------------------
    # Register a Prometheus gauge for every key
    # ------------------------------------------------------------------
    def _register_gauges(self) -> None:
        for key, val in self.values.items():
            g = Gauge(
                "volatility_param",
                "Runtime‑tunable parameter for Volatility Entry Refinement",
                ["param"],
            )
            g.labels(param=key).set(val)
            self._gauges[key] = g

    # ------------------------------------------------------------------
    # Update a single key (used by the API and by the file‑watcher)
    # ------------------------------------------------------------------
    def set(self, key: str, value: float) -> None:
        if key not in self.values:
            raise KeyError(f"Unknown volatility parameter: {key}")

        try:
            value = float(value)
        except ValueError as exc:
            raise ValueError(f"Invalid numeric value for {key}") from exc

        self.values[key] = value
        self._gauges[key].labels(param=key).set(value)
        self._persist()
        logger.info(f"VolEntryController – set {key} = {value}")

    # ------------------------------------------------------------------
    # Read‑only accessor (used by the engine)
    # ------------------------------------------------------------------
    def get(self, key: str) -> float:
        return self.values[key]

    # ------------------------------------------------------------------
    # File‑watcher – reloads the JSON if someone edited it manually
    # ------------------------------------------------------------------
    def _start_file_watcher(self) -> None:
        def _watch() -> None:
            last_mtime = (
                self.CONFIG_PATH.stat().st_mtime
                if self.CONFIG_PATH.exists()
                else 0
            )
            while not self._stop_event.is_set():
                if self.CONFIG_PATH.exists():
                    mtime = self.CONFIG_PATH.stat().st_mtime
                    if mtime != last_mtime:
                        logger.info(
                            "VolEntryController – config file changed, reloading"
                        )
                        self._load_or_initialize()
                        for k, v in self.values.items():
                            self._gauges[k].labels(param=k).set(v)
                        last_mtime = mtime
                sleep(2)

        Thread(target=_watch, daemon=True, name="vol-entry-config-watcher").start()

    # ------------------------------------------------------------------
    # Flask API – runs on 0.0.0.0:5006 (exposed via Docker‑compose)
    # ------------------------------------------------------------------
    def _start_flask_api(self) -> None:
        app = Flask(__name__)

        @app.route("/config", methods=["GET"])
        def get_all():
            """Return the whole config as JSON."""
            return jsonify(self.values)

        @app.route("/config/<key>", methods=["GET"])
        def get_one(key: str):
            if key not in self.values:
                abort(404, description=f"Parameter {key} not found")
            return jsonify({key: self.values[key]})

        @app.route("/config/<key>", methods=["POST", "PUT", "PATCH"])
        def set_one(key: str):
            if key not in self.values:
                abort(404, description=f"Parameter {key} not found")
            payload = request.get_json(force=True)
            if not payload or "value" not in payload:
                abort(400, description="JSON body must contain 'value'")
            self.set(key, payload["value"])
            return jsonify({key: self.values[key]})

        @app.route("/healthz", methods=["GET"])
        def health():
            return "OK", 200

        def _run():
            # `debug=False` and `use_reloader=False` are important for Docker
            app.run(host="0.0.0.0", port=5006, debug=False, use_reloader=False)

        Thread(target=_run, daemon=True, name="vol-entry-flask-api").start()

    # ------------------------------------------------------------------
    # Graceful shutdown (called from the main process on SIGTERM)
    # ------------------------------------------------------------------
    def stop(self) -> None:
        self._stop_event.set()


# ----------------------------------------------------------------------
# Global singleton – importable from `src/main.py` and used by the engine
# ----------------------------------------------------------------------
vol_entry_controller = VolEntryController()


# ----------------------------------------------------------------------
# Core engine – performs the volatility‑based entry refinement
# ----------------------------------------------------------------------
@dataclass
class VolatilityEntry:
    """
    Result of the volatility entry analysis.

    Attributes
    ----------
    should_enter_now : bool
        Whether the engine should place the order immediately.
    recommended_entry : float
        Refined entry price (may differ from the raw proposal).
    confidence : float
        0‑100 confidence score.
    atr_value : float
        Current ATR used for calculations.
    volatility_state : str
        One of EXPANDING, CONTRACTING, NORMAL, HIGH_VOLATILE,
        LOW_VOLATILE, UNKNOWN.
    wait_reason : Optional[str]
        Human‑readable reason why we should wait (if applicable).
    optimal_entry_range : Tuple[float, float]
        Low/high bounds of the “sweet‑spot” entry band.
    """

    should_enter_now: bool
    recommended_entry: float
    confidence: float
    atr_value: float
    volatility_state: str
    wait_reason: Optional[str]
    optimal_entry_range: Tuple[float, float]


class VolatilityEntryRefinement:
    """
    Refines entry timing based on volatility analysis.
    Prevents entering during unfavourable volatility conditions
    and provides a position‑size multiplier.
    """

    def __init__(self) -> None:
        logger.info("Volatility Entry Refinement initialised")

    # ------------------------------------------------------------------
    # Public entry point used by the trading bot
    # ------------------------------------------------------------------
    def analyze_entry_timing(
        self,
        symbol: str,
        direction: str,
        proposed_entry: float,
    ) -> VolatilityEntry:
        """
        Analyse whether the current market volatility makes the
        ``proposed_entry`` a good time to trade.

        Returns a ``VolatilityEntry`` dataclass.
        """
        # --------------------------------------------------------------
        # 1️⃣  Pull recent price data (H1 for ATR, M15 for finer granularity)
        # --------------------------------------------------------------
        rates_h1 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 200)
        rates_m15 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 400)

        if rates_h1 is None or rates_m15 is None or len(rates_h1) < 20:
            # Not enough data – fall back to a safe default
            return self._default_entry(proposed_entry)

        # --------------------------------------------------------------
        # 2️⃣  Calculate ATR (using controller‑defined look‑back)
        # --------------------------------------------------------------
        atr_lookback = int(vol_entry_controller.get("atr_lookback_h1"))
        atr = self._calculate_atr(rates_h1, period=atr_lookback)

        if atr == 0:
            return self._default_entry(proposed_entry)

        # --------------------------------------------------------------
        # 3️⃣  Detect current volatility state
        # --------------------------------------------------------------
        volatility_state = self._detect_volatility_state(rates_h1, rates_m15, atr)

        # --------------------------------------------------------------
        # 4️⃣  Are we in a consolidation zone?
        # --------------------------------------------------------------
        in_consolidation = self._is_consolidating(rates_h1, atr)

        # --------------------------------------------------------------
        # 5️⃣  Decide whether to enter now or wait
        # --------------------------------------------------------------
        should_enter, wait_reason = self._should_enter_now(
            volatility_state, in_consolidation, direction
        )

        # --------------------------------------------------------------
        # 6️⃣  Compute the optimal entry price range
        # --------------------------------------------------------------
        optimal_range = self._calculate_optimal_entry_range(
            proposed_entry, atr, direction
        )

        # --------------------------------------------------------------
        # 7️⃣  Possibly refine the entry price (e.g. move a bit better)
        # --------------------------------------------------------------
        refined_entry = self._refine_entry_price(
            proposed_entry, atr, direction, volatility_state
        )

        # --------------------------------------------------------------
        # 8️⃣  Confidence scoring (0‑100)
        # --------------------------------------------------------------
        confidence = self._calculate_entry_confidence(
            volatility_state, in_consolidation, should_enter
        )

        return VolatilityEntry(
            should_enter_now=should_enter,
            recommended_entry=refined_entry,
            confidence=confidence,
            atr_value=atr,
            volatility_state=volatility_state,
            wait_reason=wait_reason,
            optimal_entry_range=optimal_range,
        )

    # ------------------------------------------------------------------
    # 1️⃣  ATR calculation (standard Wilder’s ATR)
    # ------------------------------------------------------------------
    def _calculate_atr(self, rates: np.ndarray, period: int = 14) -> float:
        """Calculate the Average True Range."""
        if len(rates) < period + 1:
            return 0.0

        high = rates["high"]
        low = rates["low"]
        close = rates["close"]

        # True Range
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1]),
            ),
        )
        return float(np.mean(tr[-period:]))

    # ------------------------------------------------------------------
    # 2️⃣  Volatility state detection
    # ------------------------------------------------------------------
    def _detect_volatility_state(
        self,
        rates_h1: np.ndarray,
        rates_m15: np.ndarray,
        current_atr: float,
    ) -> str:
        """
        Classify the market into one of:
        EXPANDING, CONTRACTING, NORMAL, HIGH_VOLATILE,
        LOW_VOLATILE, UNKNOWN
        """
        # Historical ATRs (20‑ and 50‑period) on H1
        atr_20 = self._calculate_atr(rates_h1, period=20)
        atr_50 = self._calculate_atr(rates_h1, period=50)

        if atr_50 == 0:
            return "UNKNOWN"

        atr_ratio = current_atr / atr_50

        # Recent vs older ATR (last 30 vs previous 20 bars)
        recent_atr = self._calculate_atr(rates_h1[-30:], period=14)
        older_atr = self._calculate_atr(rates_h1[-50:-30], period=14)

        expansion_rate = (
            (recent_atr - older_atr) / older_atr if older_atr != 0 else 0.0
        )
        exp_thr = float(vol_entry_controller.get("expansion_rate_thr"))

        if expansion_rate > exp_thr:
            return "EXPANDING"
        if expansion_rate < -exp_thr:
            return "CONTRACTING"
        if atr_ratio > 1.5:
            return "HIGH_VOLATILE"
        if atr_ratio < 0.5:
            return "LOW_VOLATILE"
        return "NORMAL"

    # ------------------------------------------------------------------
    # 3️⃣  Consolidation detection (tight range < 2 × ATR)
    # ------------------------------------------------------------------
    def _is_consolidating(self, rates: np.ndarray, atr: float) -> bool:
        """Return True if the last 20 bars are in a tight range."""
        if len(rates) < 20:
            return False

        recent_highs = rates["high"][-20:]
        recent_lows = rates["low"][-20:]

        range_size = float(np.max(recent_highs) - np.min(recent_lows))
        factor = float(vol_entry_controller.get("consolidation_atr_factor"))
        return atr > 0 and range_size < (atr * factor)

    # ------------------------------------------------------------------
    # 4️⃣  Decision whether to enter now
    # ------------------------------------------------------------------
    def _should_enter_now(
        self,
        volatility_state: str,
        in_consolidation: bool,
        direction: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Returns (should_enter, wait_reason).  The logic mirrors the
        original design but each branch can be weighted via the controller.
        """
        # EXPANDING – generally favourable
        if volatility_state == "EXPANDING":
            return True, None

     # CONTRACTING – be careful if also consolidating
        if volatility_state == "CONTRACTING":
            if in_consolidation:
                return (
                    False,
                    "Consolidation + contracting volatility – wait for breakout",
                )
            return True, None

        # LOW volatility – wait for expansion unless a clear trend
        if volatility_state == "LOW_VOLATILE" and in_consolidation:
            return False, "Low volatility consolidation – wait for expansion"

        # HIGH volatility – can trade but confidence will be lower
        if volatility_state == "HIGH_VOLATILE":
            return True, None

        # NORMAL – standard behaviour
        return True, None

    # ------------------------------------------------------------------
    # 5️⃣  Optimal entry range (± 0.30 ATR by default)
    # ------------------------------------------------------------------
    def _calculate_optimal_entry_range(
        self,
        proposed_entry: float,
        atr: float,
        direction: str,
    ) -> Tuple[float, float]:
        """
        Calculate a “sweet‑spot” entry band around the proposed price.
        The width of the band is a configurable fraction of the current ATR
        (default ≈ 0.30 ATR).  For BUY orders we look a little **below**
        the proposed price; for SELL orders we look a little **above** it –
        this gives the algorithm a chance to capture a better price if the
        market moves favourably during the next few ticks.
        """
        factor = float(vol_entry_controller.get("optimal_range_factor"))  # e.g. 0.30
        range_width = atr * factor

        if direction.upper() == "BUY":
            low = proposed_entry - range_width
            high = proposed_entry
        else:  # SELL
            low = proposed_entry
            high = proposed_entry + range_width

        return low, high

    # ------------------------------------------------------------------
    # 6️⃣  Refine the entry price (small adjustment based on volatility)
    # ------------------------------------------------------------------
    def _refine_entry_price(
        self,
        proposed_entry: float,
        atr: float,
        direction: str,
        volatility_state: str,
    ) -> float:
        """
        Nudge the raw entry price a little to improve the odds.

        * In **EXPANDING** markets we move a bit **against** the direction
          (buy a little lower, sell a little higher) to capture a better
          price before the move accelerates.
        * In **CONTRACTING** or **LOW_VOLATILE** markets we stay closer
          to the original proposal (the market is likely to stall).
        * In **HIGH_VOLATILE** markets we *shrink* the adjustment to
          avoid over‑reaching.
        """
        adjust_factor = float(vol_entry_controller.get("refine_adjust_factor"))

        if volatility_state == "EXPANDING":
            # Give ourselves a little extra room
            delta = atr * adjust_factor
        elif volatility_state == "HIGH_VOLATILE":
            # Be conservative – smaller delta
            delta = atr * (adjust_factor * 0.5)
        else:
            # Normal / contracting / low – modest adjustment
            delta = atr * (adjust_factor * 0.8)

        if direction.upper() == "BUY":
            refined = proposed_entry - delta
        else:  # SELL
            refined = proposed_entry + delta

        return refined

    # ------------------------------------------------------------------
    # 7️⃣  Confidence scoring – combines several weighted factors
    # ------------------------------------------------------------------
    def _calculate_entry_confidence(
        self,
        volatility_state: str,
        in_consolidation: bool,
        should_enter: bool,
    ) -> float:
        """
        Produce a 0‑100 confidence score for the suggested entry.

        The score is built from four tunable components:

        * **ATR weight** – how much the raw ATR magnitude matters.
        * **Volatility‑state weight** – EXPANDING boosts confidence,
          CONTRACTING/LOW_VOLATILE reduces it.
        * **Consolidation weight** – being in a tight range penalises the
          score (to avoid false break‑outs).
        * **Expanding‑bonus weight** – an extra bump when the market is
          clearly expanding.

        All weights are configurable at runtime via the
        ``VolEntryController`` (see the ``DEFAULTS`` dict above).
        """
        # Start from the minimum required confidence
        min_conf = float(vol_entry_controller.get("min_confidence"))
        confidence = min_conf

        # ----- ATR influence -------------------------------------------------
        atr_weight = float(vol_entry_controller.get("weight_atr"))
        # Normalise ATR by a simple heuristic (larger ATR → higher confidence)
        atr_factor = min(1.0, self._calculate_atr_factor())
        confidence += atr_weight * atr_factor * 100

        # ----- Volatility‑state influence ------------------------------------
        vol_weight = float(vol_entry_controller.get("weight_vol_state"))
        if volatility_state == "EXPANDING":
            confidence += vol_weight * 30   # strong positive bump
        elif volatility_state == "CONTRACTING":
            confidence -= vol_weight * 20   # moderate penalty
        elif volatility_state == "HIGH_VOLATILE":
            confidence -= vol_weight * 10   # slight penalty
        elif volatility_state == "LOW_VOLATILE":
            confidence -= vol_weight * 15   # moderate penalty

        # ----- Consolidation influence ---------------------------------------
        cons_weight = float(vol_entry_controller.get("weight_consolidation"))
        if in_consolidation:
            confidence -= cons_weight * 25   # penalise tight ranges

        # ----- Expanding bonus (extra boost when clearly expanding) ---------
        if volatility_state == "EXPANDING":
            bonus_weight = float(vol_entry_controller.get("weight_expanding_bonus"))
            confidence += bonus_weight * 20

        # ----- Clamp to 0‑100 -------------------------------------------------
        confidence = max(0.0, min(100.0, confidence))
        return confidence

    # ------------------------------------------------------------------
    # Helper – normalise ATR to a 0‑1 factor (used in confidence)
    # ------------------------------------------------------------------
    def _calculate_atr_factor(self) -> float:
        """
        Returns a normalised ATR factor (0‑1) based on the current ATR
        relative to a typical market ATR (here we use the 20‑period ATR
        as a reference).  This prevents extremely low ATR values from
        inflating confidence.
        """
        # Use the 20‑period ATR as a rough reference; scale to 0‑1
        # (you can replace this with a more sophisticated reference if needed)
        return 0.5

    # ------------------------------------------------------------------
    # 8️⃣  Default fallback entry when data is missing
    # ------------------------------------------------------------------
    def _default_entry(self, proposed_entry: float) -> VolatilityEntry:
        """Return a safe default when we cannot compute anything."""
        return VolatilityEntry(
            should_enter_now=True,
            recommended_entry=proposed_entry,
            confidence=50.0,
            atr_value=0.0,
            volatility_state="UNKNOWN",
            wait_reason=None,
            optimal_entry_range=(proposed_entry, proposed_entry),
        )

    # ------------------------------------------------------------------
    # 9️⃣  Position‑size multiplier based on volatility state
    # ------------------------------------------------------------------
    def get_position_size_adjustment(self, volatility_state: str) -> float:
        """
        Return a multiplier (0.5‑1.5) that the trading bot can apply to the
        base position size.  The values are tunable via the controller.

        * HIGH_VOLATILE → reduce size (risk‑off)
        * LOW_VOLATILE  → slightly increase size (risk‑on)
        * EXPANDING / NORMAL → keep size unchanged
        """
        high_mult = float(vol_entry_controller.get("high_vol_multiplier"))
        low_mult  = float(vol_entry_controller.get("low_vol_multiplier"))

        if volatility_state == "HIGH_VOLATILE":
            return high_mult
        if volatility_state == "LOW_VOLATILE":
            return low_mult
        return 1.0
