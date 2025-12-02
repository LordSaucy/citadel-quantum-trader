#!/usr/bin/env python3
"""
Market Structure Tracker (runtime‑tunable)

Tracks real‑time bullish / bearish regime and the last pivot points
(HH/HL for bullish, LL/LH for bearish).  Provides:

* A clean Python API (`structure_tracker.analyze_structure(...)`,
  `is_structure_valid_for_trade`, `get_structure_score`, …)
* Prometheus gauges for every tunable numeric parameter
* A tiny Flask HTTP server (port 5006) that lets Grafana read / write
  those parameters in real time.
* Persistent JSON storage (`/app/config/structure_config.json`) so
  changes survive container restarts.

Author: Lawful Banker
Created: 2024‑11‑26
Version: 2.0 – Air‑Tight Money‑Printer Edition
"""

# ----------------------------------------------------------------------
# Standard library
# ----------------------------------------------------------------------
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from threading import Event, Thread
from time import sleep
from typing import Dict, List, Optional, Tuple

# ----------------------------------------------------------------------
# Third‑party
# ----------------------------------------------------------------------
import MetaTrader5 as mt5
import numpy as np
from dataclasses import dataclass
from flask import Flask, jsonify, request, abort   # pip install flask
from prometheus_client import Gauge               # already a dependency

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# 0️⃣  Data classes
# ----------------------------------------------------------------------
@dataclass
class StructurePoint:
    """A swing high or swing low point."""
    price: float
    time: datetime
    index: int
    type: str          # "HH", "HL", "LH", "LL"


@dataclass
class MarketRegime:
    """Current market structure regime."""
    regime: str                               # "BULLISH", "BEARISH", or "NEUTRAL"
    last_structure_point: Optional[StructurePoint]
    previous_structure_point: Optional[StructurePoint]
    structure_history: List[StructurePoint]
    last_shift_time: datetime


# ----------------------------------------------------------------------
# 1️⃣  Mission‑Control controller (runtime tunable parameters)
# ----------------------------------------------------------------------
class StructureController:
    """
    Holds all numeric parameters for the MarketStructureTracker,
    exposes them as Prometheus gauges and via a tiny Flask API
    (GET /config, POST /config/<key>).

    The JSON file lives in a mounted volume so it survives restarts.
    """

    # ------------------------------------------------------------------
    # Default values – you can change them at runtime via the API
    # ------------------------------------------------------------------
    DEFAULTS = {
        "lookback_period": 5,          # candles to consider for swing detection
        "debug": False,                # extra logging when True
        "min_structure_score": 0.0,    # placeholder for future extensions
    }

    CONFIG_PATH = Path("/app/config/structure_config.json")   # <-- mount this volume

    # ------------------------------------------------------------------
    # Prometheus gauges – one per key, labelled by `parameter`
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
                logger.info(f"StructureController – loaded config from {self.CONFIG_PATH}")
            except Exception as exc:   # pragma: no cover
                logger.error(f"Failed to read config – using defaults ({exc})")
                self.values = self.DEFAULTS.copy()
        else:
            logger.info("StructureController – no config file, using defaults")
            self.values = self.DEFAULTS.copy()
            self._persist()                     # create the file for the first time

    # ------------------------------------------------------------------
    # Persist the whole dict (called after every change)
    # ------------------------------------------------------------------
    def _persist(self) -> None:
        try:
            self.CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(self.CONFIG_PATH, "w") as f:
                json.dump(self.values, f, indent=2)
        except Exception as exc:   # pragma: no cover
            logger.error(f"Could not persist structure config: {exc}")

    # ------------------------------------------------------------------
    # Register a Prometheus gauge for every key
    # ------------------------------------------------------------------
    def _register_gauges(self) -> None:
        for key, val in self.values.items():
            g = Gauge(
                "structure_parameter",
                "Runtime‑tunable parameter for the Market Structure Tracker",
                ["parameter"],
            )
            g.labels(parameter=key).set(val)
            self._gauges[key] = g

    # ------------------------------------------------------------------
    # Public setter (used by the API and the file‑watcher)
    # ------------------------------------------------------------------
    def set(self, key: str, value: float) -> None:
        if key not in self.values:
            raise KeyError(f"Unknown structure parameter: {key}")

        try:
            value = float(value)
        except ValueError as exc:
            raise ValueError(f"Invalid numeric value for {key}") from exc

        self.values[key] = value
        self._gauges[key].labels(parameter=key).set(value)
        self._persist()
        logger.info(f"StructureController – set {key} = {value}")

    # ------------------------------------------------------------------
    # Public getter (used by the tracker)
    # ------------------------------------------------------------------
    def get(self, key: str) -> float:
        return self.values[key]

    # ------------------------------------------------------------------
    # File‑watcher – reloads JSON if edited manually
    # ------------------------------------------------------------------
    def _start_file_watcher(self) -> None:
        def _watch():
            last_mtime = (
                self.CONFIG_PATH.stat().st_mtime if self.CONFIG_PATH.exists() else 0
            )
            while not self._stop_event.is_set():
                if self.CONFIG_PATH.exists():
                    mtime = self.CONFIG_PATH.stat().st_mtime
                    if mtime != last_mtime:
                        logger.info("StructureController – config file changed, reloading")
                        self._load_or_initialize()
                        for k, v in self.values.items():
                            self._gauges[k].labels(parameter=k).set(v)
                        last_mtime = mtime
                sleep(2)

        Thread(target=_watch, daemon=True, name="structure-config-watcher").start()

    # ------------------------------------------------------------------
    # Flask API – runs on 0.0.0.0:5006 (exposed via docker‑compose)
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
            try:
                payload = request.get_json(force=True)
                if not payload or "value" not in payload:
                    abort(400, description="JSON body must contain 'value'")
                self.set(key, payload["value"])
                return jsonify({key: self.values[key]})
            except Exception as exc:   # pragma: no cover
                logger.error(f"API error while setting {key}: {exc}")
                abort(500, description=str(exc))

        @app.route("/healthz", methods=["GET"])
        def health():
            return "OK", 200

        def _run():
            app.run(host="0.0.0.0", port=5006, debug=False, use_reloader=False)

        Thread(target=_run, daemon=True, name="structure-flask-api").start()

    # ------------------------------------------------------------------
    # Graceful shutdown (called from the main process on SIGTERM)
    # ------------------------------------------------------------------
    def stop(self) -> None:
        self._stop_event.set()


# ----------------------------------------------------------------------
# 2️⃣  MarketStructureTracker – the actual state‑machine
# ----------------------------------------------------------------------
class MarketStructureTracker:
    """
    Real‑time market‑structure state machine.

    It uses the tunable parameters from ``structure_controller``:
        * ``lookback_period`` – number of candles on each side when detecting swings
        * ``debug`` – extra logging when True
    """

    def __init__(self) -> None:
        self.controller = structure_controller          # singleton defined below
        self.current_regime: Optional[MarketRegime] = None
        self.structure_history: List[StructurePoint] = []
        logger.info("MarketStructureTracker initialized")

    # ------------------------------------------------------------------
    # Public entry point – returns a MarketRegime object
    # ------------------------------------------------------------------
    def analyze_structure(self, symbol: str, timeframe=mt5.TIMEFRAME_H1) -> MarketRegime:
        """
        Pull recent candles, detect swing points, decide the current regime
        and store it internally.
        """
        lookback = int(self.controller.get("lookback_period"))
        debug = bool(self.controller.get("debug"))

        # ------------------------------------------------------------------
        # 1️⃣ Pull rates
        # ------------------------------------------------------------------
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 200)
        if rates is None or len(rates) < 50:
            if debug:
                logger.debug("Insufficient data – returning neutral regime")
            return self._default_regime()

        # ------------------------------------------------------------------
        # 2️⃣ Detect swing highs / lows using the *runtime* lookback
        # ------------------------------------------------------------------
        swing_highs = self._find_swing_highs(rates, lookback)
        swing_lows = self._find_swing_lows(rates, lookback)

        # ------------------------------------------------------------------
        # 3️⃣ Merge and sort by time
        # ------------------------------------------------------------------
        all_swings = self._combine_swings(swing_highs, swing_lows, rates)

        # ------------------------------------------------------------------
        # 4️⃣ Determine regime
        # ------------------------------------------------------------------
        regime = self._determine_regime(all_swings, rates)

        # Store for later queries
        self.current_regime = regime
        return regime

    # ------------------------------------------------------------------
    # Helper – swing high detection (parameterised by lookback)
    # ------------------------------------------------------------------
    def _find_swing_highs(self, rates: np.ndarray, lookback: int) -> List[int]:
        swing_highs = []
        for i in range(lookback, len(rates) - lookback):
            window = rates[i - lookback : i + lookback + 1]
            if rates[i]["high"] == np.max(window["high"]):
                swing_highs.append(i)
        return swing_highs

    # ------------------------------------------------------------------
    # Helper – swing low detection (parameterised by lookback)
    # ------------------------------------------------------------------
    def _find_swing_lows(self, rates: np.ndarray, lookback: int) -> List[int]:
        swing_lows = []
        for i in range(lookback, len(rates) - lookback):
            window = rates[i - lookback : i + lookback + 1]
            if rates[i]["low"] == np.min(window["low"]):
                swing_lows.append(i)
        return swing_lows

    # ------------------------------------------------------------------
    # Helper – combine highs & lows into StructurePoint objects
    # ------------------------------------------------------------------
    def _combine_swings(
        self,
        highs: List[int],
        lows: List[int],
        rates: np.ndarray,
    ) -> List[StructurePoint]:
        swings: List[StructurePoint] = []

        for idx in highs:
            swings.append(
                StructurePoint(
                    price=rates[idx]["high"],
                    time=datetime.fromtimestamp(rates[idx]["time"]),
                    index=idx,
                    type="HIGH",   # will be re‑classified later
                )
            )
        for idx in lows:
            swings.append(
                StructurePoint(
                    price=rates[idx]["low"],
                    time=datetime.fromtimestamp(rates[idx]["time"]),
                    index=idx,
                    type="LOW",    # will be re‑classified later
                )
            )
        swings.sort(key=lambda x: x.index)
        return swings

    # ------------------------------------------------------------------
    # Core regime determination logic (HH/HL vs LL/LH)
    # ------------------------------------------------------------------
    def _determine_regime(
        self,
        swings: List[StructurePoint],
        rates: np.ndarray,
    ) -> MarketRegime:
        """
        Implements the textbook HH‑HL / LL‑LH logic:

        * BULLISH  = Higher High + Higher Low (HH + HL)
          → stays bullish until price closes below the last HL.
        * BEARISH = Lower High + Lower Low (LH + LL)
          → stays bearish until price closes above the last LH.
        * NEUTRAL when we cannot decide.
        """
        if len(swings) < 4:
            return self._default_regime()

        recent = swings[-10:]                     # look at the last 10 pivots
        highs = [s for s in recent if s.type == "HIGH"]
        lows = [s for s in recent if s.type == "LOW"]

        if len(highs) < 2 or len(lows) < 2:
            return self._default_regime()

        # Grab the two most recent highs & lows
        last_high, prev_high = highs[-1], highs[-2]
        last_low,  prev_low  = lows[-1],  lows[-2]

        # Classify the most recent high/low
        last_high.type = "HH" if last_high.price > prev_high.price else "LH"
        last_low.type  = "HL" if last_low.price  > prev_low.price  else "LL"

        current_price = rates[-1]["close"]
        debug = bool(self.controller.get("debug"))

        # --------------------------------------------------------------
        # BULLISH case (HH + HL)
        # --------------------------------------------------------------
        if last_high.type == "HH" and last_low.type == "HL":
            regime_type = "BULLISH"
            last_structure = last_low               # HL = support
            if debug:
                logger.debug(
                    f"BULLISH regime – last HL @ {last_low.price:.5f}"
                )
            # Break of structure?
            if current_price < last_low.price:
                regime_type = "BEARISH"
                logger.info(
                    f"STRUCTURE SHIFT: Bullish → Bearish (price {current_price:.5f} broke below HL {last_low.price:.5f})"
                )

        # --------------------------------------------------------------
        # BEARISH case (LH + LL)
        # --------------------------------------------------------------
        elif last_high.type == "LH" and last_low.type == "LL":
            regime_type = "BEARISH"
            last_structure = last_high              # LH = resistance
            if debug:
                logger.debug(
                    f"BEARISH regime – last LH @ {last_high.price:.5f}"
                )
            # Break of structure?
            if current_price > last_high.price:
                regime_type = "BULLISH"
                logger.info(
                    f"STRUCTURE SHIFT: Bearish → Bullish (price {current_price:.5f} broke above LH {last_high.price:.5f})"
                )

        # --------------------------------------------------------------
        # Mixed / indeterminate
        # --------------------------------------------------------------
        else:
            regime_type = "NEUTRAL"
            # Pick the most recent point as the “last_structure”
            last_structure = (
                last_high if last_high.index > last_low.index else last_low
            )
            if debug:
                logger.debug(
                    f"Mixed signals – falling back to NEUTRAL (last point @ {last_structure.price:.5f})"
                )

        # Build the MarketRegime object
        regime = MarketRegime(
            regime=regime_type,
            last_structure_point=last_structure,
            previous_structure_point=prev_high if regime_type == "BEARISH" else prev_low,
            structure_history=recent,
            last_shift_time=datetime.now(),
        )
        return regime

    # ------------------------------------------------------------------
    # Public helper – does the current regime allow a trade in `direction`?
    # ------------------------------------------------------------------
    def is_structure_valid_for_trade(
        self,
        direction: str,
        entry_price: float,
    ) -> Tuple[bool, str]:
        """
        Returns (valid, reason).  The logic mirrors the description in the
        module doc‑string:

        * BUY  → requires BULLISH regime and entry near the last HL.
        * SELL → requires BEARISH regime and entry near the last LH.
        """
        if self.current_regime is None:
            return False, "No regime data available"

        regime = self.current_regime.regime
        last_pt = self.current_regime.last_structure_point

        if direction == "BUY":
            if regime != "BULLISH":
                return False, f"Regime is {regime} – cannot BUY (last point {last_pt.price:.5f})"
            # Proximity check (within 0.5 % of the support level)
            dist = abs(entry_price - last_pt.price) / last_pt.price
            if dist <= 0.005:
                return True, f"BULLISH – entry near HL support ({last_pt.price:.5f})"
            return True, "BULLISH – entry away from HL but regime OK"

        if direction == "SELL":
            if regime != "BEARISH":
                return False, f"Regime is {regime} – cannot SELL (last point {last_pt.price:.5f})"
            dist = abs(entry_price - last_pt.price) / last_pt.price
            if dist <= 0.005:
                return True, f"BEARISH – entry near LH resistance ({last_pt.price:.5f})"
            return True, "BEARISH – entry away from LH but regime OK"
   # ------------------------------------------------------------------
    # Score how well a prospective trade aligns with the current structure
    # ------------------------------------------------------------------
    def get_structure_score(self, direction: str, entry_price: float) -> float:
        """
        Returns a float in the range 0‑1 indicating how well the trade
        conforms to the detected market structure.

        Scoring criteria
        ----------------
        * If the trade direction does **not** match the regime → 0.0
        * Distance from the last pivot point (HL for bullish, LH for bearish)
          – ≤ 0.2 %  → 1.0
          – ≤ 0.5 %  → 0.9
          – ≤ 1 %   → 0.8
          – otherwise → 0.7
        * Small bonus (+0.1) when direction and regime are perfectly aligned
        """
        valid, _ = self.is_structure_valid_for_trade(direction, entry_price)
        if not valid:
            return 0.0

        if self.current_regime is None:
            # No regime information – give a neutral mid‑score
            return 0.5

        last_pt = self.current_regime.last_structure_point
        if last_pt is None:
            return 0.5

        # Proximity to the pivot point
        distance = abs(entry_price - last_pt.price) / last_pt.price

        if distance < 0.002:          # ≤ 0.2 %
            proximity_score = 1.0
        elif distance < 0.005:        # ≤ 0.5 %
            proximity_score = 0.9
        elif distance < 0.01:         # ≤ 1 %
            proximity_score = 0.8
        else:
            proximity_score = 0.7

        # Bonus for perfect regime‑direction match
        regime = self.current_regime.regime
        regime_bonus = 0.1 if (
            (direction == "BUY" and regime == "BULLISH") or
            (direction == "SELL" and regime == "BEARISH")
        ) else 0.0

        final_score = min(1.0, proximity_score + regime_bonus)
        return final_score

    # ------------------------------------------------------------------
    # Fallback regime when we cannot determine anything meaningful
    # ------------------------------------------------------------------
    def _default_regime(self) -> MarketRegime:
        """Neutral regime used when data is insufficient."""
        return MarketRegime(
            regime="NEUTRAL",
            last_structure_point=None,
            previous_structure_point=None,
            structure_history=[],
            last_shift_time=datetime.now(),
        )


# ----------------------------------------------------------------------
# 3️⃣  Global singletons – import these from other modules
# ----------------------------------------------------------------------
# Controller (holds tunable parameters, Prometheus gauges, Flask API)
structure_controller = StructureController()

# Tracker (state‑machine that uses the controller’s parameters)
structure_tracker = MarketStructureTracker()
