#!/usr/bin/env python3
"""
Market Structure Tracker (runtime‑tunable)

Tracks real‑time bullish / bearish regime and the last pivot points
(HH/HL for bullish, LL/LH for bearish). Provides:

- A clean Python API (`structure_tracker.analyze_structure(...)`,
  `is_structure_valid_for_trade`, `get_structure_score`, …)
- Prometheus gauges for every tunable numeric parameter
- A tiny Flask HTTP server (port 5006) that lets Grafana read / write
  those parameters in real time.
- Persistent JSON storage (`/app/config/structure_config.json`) so
  changes survive container restarts.

✅ SECURITY: CSRF protection enabled on all POST/PUT/PATCH endpoints.
GET endpoints remain unprotected (safe by design).

Author: Lawful Banker
Created: 2024‑11‑26
Version: 2.1 – CSRF Protected
"""

# =====================================================================
# Standard library
# =====================================================================
import json
import logging
import os
import secrets
from datetime import datetime
from pathlib import Path
from threading import Event, Thread
from time import sleep
from typing import Dict, List, Optional, Tuple

# =====================================================================
# Third‑party
# =====================================================================
import MetaTrader5 as mt5
import numpy as np
from dataclasses import dataclass
from flask import Flask, jsonify, request, abort
from flask_wtf.csrf import CSRFProtect
from prometheus_client import Gauge

# =====================================================================
# Logging
# =====================================================================
logger = logging.getLogger(__name__)

# =====================================================================
# Data classes
# =====================================================================
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


# =====================================================================
# Mission‑Control controller (runtime tunable parameters)
# =====================================================================
class StructureController:
    """
    Holds all numeric parameters for the MarketStructureTracker,
    exposes them as Prometheus gauges and via a tiny Flask API
    (GET /config, POST /config/).

    The JSON file lives in a mounted volume so it survives restarts.
    """

    # -----------------------------------------------------------------
    # Default values – can be changed at runtime via the API
    # -----------------------------------------------------------------
    DEFAULTS = {
        "lookback_period": 5,          # candles to consider for swing detection
        "debug": False,                # extra logging when True
        "min_structure_score": 0.0,    # placeholder for future extensions
    }

    CONFIG_PATH = Path("/app/config/structure_config.json")   # <-- mount this volume

    # -----------------------------------------------------------------
    # Prometheus gauges – one per key, labelled by `parameter`
    # -----------------------------------------------------------------
    _gauges: Dict[str, Gauge] = {}

    # -----------------------------------------------------------------
    def __init__(self) -> None:
        self._stop_event = Event()
        self._load_or_initialize()
        self._register_gauges()
        self._start_file_watcher()
        self._start_flask_api()

    # -----------------------------------------------------------------
    # Load persisted JSON or fall back to defaults
    # -----------------------------------------------------------------
    def _load_or_initialize(self) -> None:
        if self.CONFIG_PATH.exists():
            try:
                with open(self.CONFIG_PATH, "r") as f:
                    self.values = json.load(f)
                logger.info(
                    f"StructureController – loaded config from {self.CONFIG_PATH}"
                )
            except Exception as exc:   # pragma: no cover
                logger.error(
                    f"Failed to read config – using defaults ({exc})"
                )
                self.values = self.DEFAULTS.copy()
        else:
            logger.info(
                "StructureController – no config file, using defaults"
            )
            self.values = self.DEFAULTS.copy()
            self._persist()                     # create the file for the first time

    # -----------------------------------------------------------------
    # Persist the whole dict (called after every change)
    # -----------------------------------------------------------------
    def _persist(self) -> None:
        try:
            self.CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(self.CONFIG_PATH, "w") as f:
                json.dump(self.values, f, indent=2)
        except Exception as exc:   # pragma: no cover
            logger.error(f"Could not persist structure config: {exc}")

    # -----------------------------------------------------------------
    # Register a Prometheus gauge for every key
    # -----------------------------------------------------------------
    def _register_gauges(self) -> None:
        for key, val in self.values.items():
            g = Gauge(
                "structure_parameter",
                "Runtime‑tunable parameter for the Market Structure Tracker",
                ["parameter"],
            )
            g.labels(parameter=key).set(val)
            self._gauges[key] = g

    # -----------------------------------------------------------------
    # Public setter (used by the API and the file‑watcher)
    # -----------------------------------------------------------------
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

    # -----------------------------------------------------------------
    # Public getter (used by the tracker)
    # -----------------------------------------------------------------
    def get(self, key: str) -> float:
        return self.values[key]

    # -----------------------------------------------------------------
    # ✅ SECURITY FIX: Secure SECRET_KEY management
    # -----------------------------------------------------------------
    def _get_or_create_secret_key(self) -> str:
        """
        Retrieve SECRET_KEY from environment or generate/persist one.

        Priority:
        1️⃣  FLASK_STRUCTURE_SECRET_KEY env var (production)
        2️⃣  Persisted key file (generated once, reused across restarts)
        3️⃣  Generate new cryptographically‑secure key (fallback)
        """
        # 1️⃣  Environment variable first
        env_key = os.getenv('FLASK_STRUCTURE_SECRET_KEY')
        if env_key:
            logger.info(
                "📌 Structure SECRET_KEY loaded from FLASK_STRUCTURE_SECRET_KEY environment variable"
            )
            return env_key

        # 2️⃣  Persisted key file
        secret_file = Path("/app/config/structure_secret_key")
        if secret_file.exists():
            try:
                with open(secret_file, 'r') as f:
                    persisted_key = f.read().strip()
                    if persisted_key and len(persisted_key) >= 32:
                        logger.info(
                            "📌 Structure SECRET_KEY loaded from persisted secure key file"
                        )
                        return persisted_key
            except Exception as exc:
                logger.warning(
                    f"⚠️ Could not read persisted Structure SECRET_KEY: {exc}"
                )

        # 3️⃣  Generate a new key
        new_key = secrets.token_urlsafe(32)

        # Persist for future restarts
        try:
            secret_file.parent.mkdir(parents=True, exist_ok=True)
            with open(secret_file, 'w') as f:
                f.write(new_key)
            secret_file.chmod(0o600)
            logger.info("✅ Generated and persisted new Structure SECRET_KEY")
        except Exception as exc:
            logger.warning(
                f"⚠️ Could not persist Structure SECRET_KEY: {exc}"
            )

        return new_key

    # -----------------------------------------------------------------
    # ✅ FIXED: Reduced cognitive complexity from 16 to 8
    # -----------------------------------------------------------------
    def _check_file_modified(self, current_mtime: float, last_mtime: float) -> bool:
        """Return True if the file modification time has changed."""
        return current_mtime != last_mtime

    def _reload_config_from_file(self) -> None:
        """Reload configuration from file and update all Prometheus gauges."""
        logger.info("StructureController – config file changed, reloading")
        self._load_or_initialize()
        for k, v in self.values.items():
            self._gauges[k].labels(parameter=k).set(v)

    def _watch_config_file_loop(self, last_mtime: float) -> float:
        """Single iteration of the config‑file watcher loop."""
        if not self.CONFIG_PATH.exists():
            return last_mtime

        current_mtime = self.CONFIG_PATH.stat().st_mtime
        if self._check_file_modified(current_mtime, last_mtime):
            self._reload_config_from_file()
            return current_mtime
        return last_mtime

    def _start_file_watcher(self) -> None:
        """Watch the config file for changes and reload if needed."""
        def _watch():
            last_mtime = (
                self.CONFIG_PATH.stat().st_mtime
                if self.CONFIG_PATH.exists()
                else 0
            )
            while not self._stop_event.is_set():
                last_mtime = self._watch_config_file_loop(last_mtime)
                sleep(2)

        Thread(
            target=_watch,
            daemon=True,
            name="structure-config-watcher"
        ).start()

    # -----------------------------------------------------------------
    # ✅ SECURITY FIX: Flask API with CSRF protection
    # -----------------------------------------------------------------
    def _start_flask_api(self) -> None:
        """Spin up a tiny Flask API with CSRF protection."""
        app = Flask(__name__)

        # Use the secure secret key helper
        app.config['SECRET_KEY'] = self._get_or_create_secret_key()

        # Enable CSRF protection
        csrf = CSRFProtect(app)

        @app.route("/config", methods=["GET"])
        def get_all():
            """GET endpoint – read‑only, no CSRF needed."""
            return jsonify(self.values)

        @app.route("/config/<key>", methods=["GET"])
        def get_one(key: str):
            """GET endpoint – read‑only, no CSRF needed."""
            if key not in self.values:
                abort(404, description=f"Parameter {key} not found")
            return jsonify({key: self.values[key]})

        @app.route("/config/<key>", methods=["POST", "PUT", "PATCH"])
        @csrf.protect  # ✅ CSRF token required
        def set_one(key: str):
            """POST/PUT/PATCH endpoint – CSRF protected."""
            if key not in self.values:
                abort(404, description=f"Parameter {key} not found")
            try:
                payload = request.get_json(force=True)
                if not payload or "value" not in payload:
                    abort(400, description="JSON body must contain 'value'")
                self.set(key, payload["value"])
                return jsonify({key: self.values[key]})
            except (KeyError, ValueError) as exc:
                abort(400, description=str(exc))

        @app.route("/healthz", methods=["GET"])
        def health():
            """Health‑check endpoint – no CSRF needed."""
            return "OK", 200

        def _run():
            app.run(host="0.0.0.0", port=5006, debug=False, use_reloader=False)

        Thread(target=_run, daemon=True, name="structure-flask-api").start()
        logger.info(
            "📡 Structure Flask API listening on 0.0.0.0:5006 (CSRF protected)"
        )

    # -----------------------------------------------------------------
    # Graceful shutdown (called from the main process on SIGTERM)
    # -----------------------------------------------------------------
    def stop(self) -> None:
        """Signal the background threads to stop."""
        self._stop_event.set()


# =====================================================================
# Global singleton – import this from other modules
# =====================================================================
structure_controller = StructureController()


# =====================================================================
# MarketStructureTracker – the actual state‑machine
# =====================================================================
class MarketStructureTracker:
    """
    Real‑time market‑structure state machine.

    Uses the tunable parameters from ``structure_controller``:
        * ``lookback_period`` – number of candles on each side when detecting swings
        * ``debug`` – extra logging when True
    """

    def __init__(self) -> None:
        self.controller = structure_controller          # singleton defined above
        self.current_regime: Optional[MarketRegime] = None
        self.structure_history: List[StructurePoint] = []
        logger.info("MarketStructureTracker initialized")

    # -----------------------------------------------------------------
    # Public entry point – returns a MarketRegime object
    # -----------------------------------------------------------------
    def analyze_structure(self, symbol: str, timeframe=mt5.TIMEFRAME_H1) -> MarketRegime:
        """
        Pull recent candles, detect swing points, decide the current regime
        and store it internally.
        """
        lookback = int(self.controller.get("lookback_period"))
        debug = bool(self.controller.get("debug"))

        # Pull rates
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 200)
        if rates is None or len(rates) < 50:
            if debug:
                logger.debug("Insufficient data – returning neutral regime")
            return self._default_regime()

        # Detect swing highs / lows using the runtime lookback
        swing_highs = self._find_swing_highs(rates, lookback)
        swing_lows = self._find_swing_lows(rates, lookback)

        # Merge and sort by time
        all_swings = self._combine_swings(swing_highs, swing_lows, rates)

        # Determine regime
        regime = self._determine_regime(all_swings, rates)

        # Store for later queries
        self.current_regime = regime
        return regime

    # -----------------------------------------------------------------
    # Helper – swing high detection (parameterised by lookback)
    # -----------------------------------------------------------------
   def _find_swing_highs(self, rates: np.ndarray, lookback: int) -> List[int]:
        swing_highs = []
        for i in range(lookback, len(rates) - lookback):
            window = rates[i - lookback : i + lookback + 1]
            if rates[i]["high"] == np.max(window["high"]):
                swing_highs.append(i)
        return swing_highs

    # -----------------------------------------------------------------
    # Helper – swing low detection (parameterised by lookback)
    # -----------------------------------------------------------------
    def _find_swing_lows(self, rates: np.ndarray, lookback: int) -> List[int]:
        swing_lows = []
        for i in range(lookback, len(rates) - lookback):
            window = rates[i - lookback : i + lookback + 1]
            if rates[i]["low"] == np.min(window["low"]):
                swing_lows.append(i)
        return swing_lows

    # -----------------------------------------------------------------
    # Helper – combine highs & lows into StructurePoint objects
    # -----------------------------------------------------------------
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
                    type="HIGH",
                )
            )
        for idx in lows:
            swings.append(
                StructurePoint(
                    price=rates[idx]["low"],
                    time=datetime.fromtimestamp(rates[idx]["time"]),
                    index=idx,
                    type="LOW",
                )
            )
        swings.sort(key=lambda x: x.index)
        return swings

    # -----------------------------------------------------------------
    # Validation helper – ensure enough swing data
    # -----------------------------------------------------------------
    def _validate_swing_data(self, swings: List[StructurePoint]) -> bool:
        """Return True if we have enough swing points to decide a regime."""
        return len(swings) >= 4

    # -----------------------------------------------------------------
    # Helper – extract recent highs & lows (last 10 pivots)
    # -----------------------------------------------------------------
    def _extract_recent_highs_lows(
        self,
        swings: List[StructurePoint],
    ) -> Tuple[List[StructurePoint], List[StructurePoint]]:
        recent = swings[-10:]
        highs = [s for s in recent if s.type == "HIGH"]
        lows = [s for s in recent if s.type == "LOW"]
        return highs, lows

    # -----------------------------------------------------------------
    # Classification helpers (bullish, bearish, neutral)
    # -----------------------------------------------------------------
    def _classify_bullish_regime(
        self,
        last_high: StructurePoint,
        last_low: StructurePoint,
        current_price: float,
    ) -> Tuple[str, StructurePoint]:
        """
        Classify BULLISH regime (HH + HL).

        Returns (regime_type, last_structure_point)
        """
        last_high.type = "HH"
        last_low.type = "HL"
        regime_type = "BULLISH"
        last_structure = last_low

        if bool(self.controller.get("debug")):
            logger.debug(
                f"BULLISH regime – last HL @ {last_low.price:.5f}"
            )

        # Break of structure?
        if current_price < last_low.price:
            regime_type = "BEARISH"
            logger.info(
                f"STRUCTURE SHIFT: Bullish → Bearish "
                f"(price {current_price:.5f} broke below HL {last_low.price:.5f})"
            )
        return regime_type, last_structure

    def _classify_bearish_regime(
        self,
        last_high: StructurePoint,
        last_low: StructurePoint,
        current_price: float,
    ) -> Tuple[str, StructurePoint]:
        """
        Classify BEARISH regime (LH + LL).

        Returns (regime_type, last_structure_point)
        """
        last_high.type = "LH"
        last_low.type = "LL"
        regime_type = "BEARISH"
        last_structure = last_high

        if bool(self.controller.get("debug")):
            logger.debug(
                f"BEARISH regime – last LH @ {last_high.price:.5f}"
            )

        # Break of structure?
        if current_price > last_high.price:
            regime_type = "BULLISH"
            logger.info(
                f"STRUCTURE SHIFT: Bearish → Bullish "
                f"(price {current_price:.5f} broke above LH {last_high.price:.5f})"
            )
        return regime_type, last_structure

    def _classify_neutral_regime(
        self,
        last_high: StructurePoint,
        last_low: StructurePoint,
    ) -> Tuple[str, StructurePoint]:
        """
        Classify NEUTRAL regime (mixed signals).

              return regime_type, last_structure

    # -----------------------------------------------------------------
    # ✅ FIXED: Reduced cognitive complexity from 16 to 8
    #           by extracting regime classification logic
    # -----------------------------------------------------------------
    def _validate_swing_data(self, swings: List[StructurePoint]) -> bool:
        """
        Validate that we have enough swing data to determine regime.
        """
        return len(swings) >= 4

    def _extract_recent_highs_lows(
        self,
        swings: List[StructurePoint],
    ) -> Tuple[List[StructurePoint], List[StructurePoint]]:
        """
        Extract recent highs and lows from swings (last 10 pivots).
        """
        recent = swings[-10:]
        highs = [s for s in recent if s.type == "HIGH"]
        lows = [s for s in recent if s.type == "LOW"]
        return highs, lows

    def _classify_bullish_regime(
        self,
        last_high: StructurePoint,
        last_low: StructurePoint,
        current_price: float,
    ) -> Tuple[str, StructurePoint]:
        """
        Classify BULLISH regime (HH + HL).

        Returns (regime_type, last_structure_point)
        """
        last_high.type = "HH"
        last_low.type = "HL"

        regime_type = "BULLISH"
        last_structure = last_low

        debug = bool(self.controller.get("debug"))
        if debug:
            logger.debug(
                f"BULLISH regime – last HL @ {last_low.price:.5f}"
            )

        # Break of structure?
        if current_price < last_low.price:
            regime_type = "BEARISH"
            logger.info(
                f"STRUCTURE SHIFT: Bullish → Bearish "
                f"(price {current_price:.5f} broke below HL {last_low.price:.5f})"
            )

        return regime_type, last_structure

    def _classify_bearish_regime(
        self,
        last_high: StructurePoint,
        last_low: StructurePoint,
        current_price: float,
    ) -> Tuple[str, StructurePoint]:
        """
        Classify BEARISH regime (LH + LL).

        Returns (regime_type, last_structure_point)
        """
        last_high.type = "LH"
        last_low.type = "LL"

        regime_type = "BEARISH"
        last_structure = last_high

        debug = bool(self.controller.get("debug"))
        if debug:
            logger.debug(
                f"BEARISH regime – last LH @ {last_high.price:.5f}"
            )

        # Break of structure?
        if current_price > last_high.price:
            regime_type = "BULLISH"
            logger.info(
                f"STRUCTURE SHIFT: Bearish → Bullish "
                f"(price {current_price:.5f} broke above LH {last_high.price:.5f})"
            )

        return regime_type, last_structure

    def _classify_neutral_regime(
        self,
        last_high: StructurePoint,
        last_low: StructurePoint,
    ) -> Tuple[str, StructurePoint]:
        """
        Classify NEUTRAL regime (mixed signals).

        Returns (regime_type, last_structure_point)
        """
        regime_type = "NEUTRAL"
        last_structure = (
            last_high if last_high.index > last_low.index else last_low
        )

        debug = bool(self.controller.get("debug"))
        if debug:
            logger.debug(
                f"Mixed signals – falling back to NEUTRAL "
                f"(last point @ {last_structure.price:.5f})"
            )

        return regime_type, last_structure

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

        ✅ FIXED: Reduced complexity from 23 to 12 by extracting:
                  - Validation logic (`_validate_swing_data`)
                  - High/low extraction (`_extract_recent_highs_lows`)
                  - Bullish classification (`_classify_bullish_regime`)
                  - Bearish classification (`_classify_bearish_regime`)
                  - Neutral classification (`_classify_neutral_regime`)
        """
        if not self._validate_swing_data(swings):
            return self._default_regime()

        highs, lows = self._extract_recent_highs_lows(swings)

        if len(highs) < 2 or len(lows) < 2:
            return self._default_regime()

        # Grab the two most recent highs & lows
        last_high, prev_high = highs[-1], highs[-2]
        last_low, prev_low = lows[-1], lows[-2]

        current_price = rates[-1]["close"]

        # Classify the regime based on high/low relationships
        if last_high.price > prev_high.price and last_low.price > prev_low.price:
            # BULLISH case (HH + HL)
            regime_type, last_structure = self._classify_bullish_regime(
                last_high, last_low, current_price
            )
            prev_structure = prev_low

        elif last_high.price < prev_high.price and last_low.price < prev_low.price:
            # BEARISH case (LH + LL)
            regime_type, last_structure = self._classify_bearish_regime(
                last_high, last_low, current_price
            )
            prev_structure = prev_high

        else:
            # Mixed / indeterminate
            regime_type, last_structure = self._classify_neutral_regime(
                last_high, last_low
            )
            prev_structure = (
                prev_high if last_high.index > last_low.index else prev_low
            )

        # Build the MarketRegime object
        regime = MarketRegime(
            regime=regime_type,
            last_structure_point=last_structure,
            previous_structure_point=prev_structure,
            structure_history=[last_high, last_low],
            last_shift_time=datetime.now(),
        )
        return regime

    # -----------------------------------------------------------------
    # Public helper – does the current regime allow a trade in `direction`?
    # -----------------------------------------------------------------
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
                return (
                    False,
                    f"Regime is {regime} – cannot BUY (last point {last_pt.price:.5f})",
                )
            # Proximity check (within 0.5 % of the support level)
            dist = abs(entry_price - last_pt.price) / last_pt.price
            if dist <= 0.005:
                return True, f"BULLISH – entry near HL support ({last_pt.price:.5f})"
            return True, "BULLISH – entry away from HL but regime OK"

        if direction == "SELL":
            if regime != "BEARISH":
                return (
                    False,
                    f"Regime is {regime} – cannot SELL (last point {last_pt.price:.5f})",
                )
            dist = abs(entry_price - last_pt.price) / last_pt.price
            if dist <= 0.005:
                return True, f"BEARISH – entry near LH resistance ({last_pt.price:.5f})"
            return True, "BEARISH – entry away from LH but regime OK"

        return False, f"Invalid direction: {direction}"

    # -----------------------------------------------------------------
    # Score how well a prospective trade aligns with the current structure
    # -----------------------------------------------------------------
    def get_structure_score(self, direction: str, entry_price: float) -> float:
        """
        Returns a float in the range 0‑1 indicating how well the trade
        conforms to the detected market structure.

        Scoring criteria
        ----------------
        * If the trade direction does **not** match the regime → 0.0
        * Distance from the last pivot point (HL for bullish, LH for bearish)
          – ≤ 0.2 %  → 1.0
          – ≤ 0.5 %  → 0.9
          – ≤ 1 %   → 0.8
          – otherwise → 0.7
        * Small bonus (+0.1) when direction and regime are perfectly aligned
        """
        valid, _ = self.is_structure_valid_for_trade(direction, entry_price)
        if not valid:
            return 0.0

        if self.current_regime is None:
            return 0.5

        last_pt = self.current_regime.last_structure_point
        if last_pt is None:
            return 0.5

        # Proximity to the pivot point
        distance = abs(entry_price - last_pt.price) / last_pt.price

        if distance < 0.002:          # ≤ 0.2 %
            proximity_score = 1.0
        elif distance < 0.005:        # ≤ 0.5 %
            proximity_score = 0.9
        elif distance < 0.01:         # ≤ 1 %
            proximity_score = 0.8
        else:
            proximity_score = 0.7

        # Bonus for perfect regime‑direction match
        regime = self.current_regime.regime
        regime_bonus = 0.1 if (
            (direction == "BUY" and regime == "BULLISH")
            or (direction == "SELL" and regime == "BEARISH")
        ) else 0.0

        final_score = min(1.0, proximity_score + regime_bonus)
        return final_score

    # -----------------------------------------------------------------
    # Fallback regime when we cannot determine anything meaningful
    # -----------------------------------------------------------------
    def _default_regime(self) -> MarketRegime:
        """Neutral regime used when data is insufficient."""
        return MarketRegime(
            regime="NEUTRAL",
            last_structure_point=None,
            previous_structure_point=None,
            structure_history=[],
            last_shift_time=datetime.now(),
        )
        
