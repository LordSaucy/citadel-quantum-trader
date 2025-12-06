#!/usr/bin/env python3
"""
VolEntry.py – Production‑ready volatility‑entry refinement system.
"""

import json
import logging
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Thread
from time import sleep
from typing import Dict, Optional, Tuple

import MetaTrader5 as mt5
import numpy as np
from flask import Flask, abort, jsonify, request
from flask_wtf.csrf import CSRFProtect
from prometheus_client import Gauge

logger = logging.getLogger(__name__)


class VolEntryController:
    """Central controller for VolatilityEntryRefinement parameters."""

    DEFAULTS: Dict[str, float] = {
        "weight_atr": 0.20,
        "weight_vol_state": 0.30,
        "weight_consolidation": 0.25,
        "weight_expanding_bonus": 0.15,
        "min_confidence": 60.0,
        "atr_lookback_h1": 14,
        "atr_lookback_m15": 14,
        "expansion_rate_thr": 0.15,
        "high_vol_multiplier": 1.5,
        "low_vol_multiplier": 0.8,
        "consolidation_atr_factor": 2.0,
        "optimal_range_factor": 0.30,
        "refine_adjust_factor": 0.15,
    }

    CONFIG_PATH = Path("/app/config/vol_entry_config.json")
    _gauges: Dict[str, Gauge] = {}

    def __init__(self) -> None:
        self._stop_event = Event()
        self._load_or_initialize()
        self._register_gauges()
        self._start_file_watcher()
        self._start_flask_api()

    def _load_or_initialize(self) -> None:
        if self.CONFIG_PATH.exists():
            try:
                with open(self.CONFIG_PATH, "r") as f:
                    self.values = json.load(f)
                logger.info(f"VolEntryController – loaded config from {self.CONFIG_PATH}")
            except Exception as exc:
                logger.error(f"Failed to read config – using defaults ({exc})")
                self.values = self.DEFAULTS.copy()
        else:
            logger.info("VolEntryController – no config file, using defaults")
            self.values = self.DEFAULTS.copy()
            self._persist()

    def _persist(self) -> None:
        try:
            self.CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(self.CONFIG_PATH, "w") as f:
                json.dump(self.values, f, indent=2)
        except Exception as exc:
            logger.error(f"Could not persist vol‑entry config: {exc}")

    def _register_gauges(self) -> None:
        for key, val in self.values.items():
            g = Gauge(
                "volatility_param",
                "Runtime‑tunable parameter for Volatility Entry Refinement",
                ["param"],
            )
            g.labels(param=key).set(val)
            self._gauges[key] = g

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

    def get(self, key: str) -> float:
        return self.values[key]

    def _get_or_create_secret_key(self) -> str:
        """
        Retrieve SECRET_KEY from environment or generate/persist one.
        """
        env_key = os.getenv('FLASK_VOLENTRY_SECRET_KEY')
        if env_key:
            logger.info("📌 VolEntry SECRET_KEY loaded from FLASK_VOLENTRY_SECRET_KEY environment variable")
            return env_key
        
        secret_file = Path("/app/config/volentry_secret_key")
        if secret_file.exists():
            try:
                with open(secret_file, 'r') as f:
                    persisted_key = f.read().strip()
                    if persisted_key and len(persisted_key) >= 32:
                        logger.info("📌 VolEntry SECRET_KEY loaded from persisted secure key file")
                        return persisted_key
            except Exception as exc:
                logger.warning(f"⚠️ Could not read persisted VolEntry SECRET_KEY: {exc}")
        
        new_key = secrets.token_urlsafe(32)
        
        try:
            secret_file.parent.mkdir(parents=True, exist_ok=True)
            with open(secret_file, 'w') as f:
                f.write(new_key)
            secret_file.chmod(0o600)
            logger.info("✅ Generated and persisted new VolEntry SECRET_KEY")
        except Exception as exc:
            logger.warning(f"⚠️ Could not persist VolEntry SECRET_KEY: {exc}")
        
        return new_key

    def _start_file_watcher(self) -> None:
        def _watch() -> None:
            last_mtime = (
                self.CONFIG_PATH.stat().st_mtime
                if self.CONFIG_PATH.exists()
                else 0
            )
            while not self._stop_event.is_set():
                if self._config_file_changed(last_mtime):
                    logger.info("VolEntryController – config file changed, reloading")
                    self._load_or_initialize()
                    for k, v in self.values.items():
                        self._gauges[k].labels(param=k).set(v)
                    last_mtime = self.CONFIG_PATH.stat().st_mtime
                sleep(2)

        Thread(target=_watch, daemon=True, name="vol-entry-config-watcher").start()

    def _config_file_changed(self, last_mtime: float) -> bool:
        """Check if the config file has been modified."""
        if not self.CONFIG_PATH.exists():
            return False
        mtime = self.CONFIG_PATH.stat().st_mtime
        return mtime != last_mtime

    def _start_flask_api(self) -> None:
        """Spin up Flask API with CSRF protection."""
        app = Flask(__name__)
        app.config['SECRET_KEY'] = self._get_or_create_secret_key()
        csrf = CSRFProtect(app)

        @app.route("/config", methods=["GET"])
        def get_all():
            return jsonify(self.values)

        @app.route("/config/<key>", methods=["GET"])
        def get_one(key: str):
            if key not in self.values:
                abort(404, description=f"Parameter {key} not found")
            return jsonify({key: self.values[key]})

        @app.route("/config/<key>", methods=["POST", "PUT", "PATCH"])
        @csrf.protect
        def set_one(key: str):
            if key not in self.values:
                abort(404, description=f"Parameter {key} not found")
            payload = request.get_json(force=True)
            if not payload or "value" not in payload:
                abort(400, description="JSON body must contain 'value'")
            try:
                self.set(key, payload["value"])
                return jsonify({key: self.values[key]})
            except (KeyError, ValueError) as exc:
                abort(400, description=str(exc))

        @app.route("/healthz", methods=["GET"])
        def health():
            return "OK", 200

        def _run():
            app.run(host="0.0.0.0", port=5006, debug=False, use_reloader=False)

        Thread(target=_run, daemon=True, name="vol-entry-flask-api").start()
        logger.info("📡 VolEntry Flask API listening on 0.0.0.0:5006 (CSRF protected)")

    def stop(self) -> None:
        self._stop_event.set()


vol_entry_controller = VolEntryController()


@dataclass
class VolatilityEntry:
    """Result of the volatility entry analysis."""
    should_enter_now: bool
    recommended_entry: float
    confidence: float
    atr_value: float
    volatility_state: str
    wait_reason: Optional[str]
    optimal_entry_range: Tuple[float, float]


class VolatilityEntryRefinement:
    """Refines entry timing based on volatility analysis."""

    def __init__(self) -> None:
        logger.info("Volatility Entry Refinement initialised")

    def analyze_entry_timing(
        self,
        symbol: str,
        direction: str,
        proposed_entry: float,
    ) -> VolatilityEntry:
        """Analyse entry timing based on volatility."""
    
        rates_h1 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 200)

        if rates_h1 is None or len(rates_h1) < 20:
            return self._default_entry(proposed_entry)

        volatility_state = self._detect_volatility_state(rates_h1)
        atr = self._calculate_atr(rates_h1)
        in_consolidation = self._is_consolidating(rates_h1, atr)
        should_enter, wait_reason = self._should_enter_now(volatility_state, in_consolidation)
        confidence = self._calculate_entry_confidence(volatility_state, in_consolidation)
        refined_entry = self._refine_entry_price(proposed_entry, atr, direction, volatility_state)
        optimal_range = self._calculate_optimal_entry_range(proposed_entry, atr, direction)

        return VolatilityEntry(
            should_enter_now=should_enter,
            recommended_entry=refined_entry,
            confidence=confidence,
            atr_value=atr,
            volatility_state=volatility_state,
            wait_reason=wait_reason,
            optimal_entry_range=optimal_range,
        )

    def _calculate_atr(self, rates: np.ndarray, period: int = 14) -> float:
        """Calculate ATR."""
        if len(rates) < period + 1:
            return 0.0
        high = rates["high"]
        low = rates["low"]
        close = rates["close"]
        tr = np.maximum(high[1:] - low[1:], 
                       np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
        return float(np.mean(tr[-period:]))

    def _detect_volatility_state(self, rates_h1: np.ndarray) -> str:
        """Classify volatility state."""
        atr_50 = self._calculate_atr(rates_h1, period=50)
        current_atr = self._calculate_atr(rates_h1, period=14)
        if atr_50 == 0:
            return "UNKNOWN"
        atr_ratio = current_atr / atr_50
        recent_atr = self._calculate_atr(rates_h1[-30:], period=14)
        older_atr = self._calculate_atr(rates_h1[-50:-30], period=14)
        expansion_rate = ((recent_atr - older_atr) / older_atr) if older_atr != 0 else 0.0
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

    def _is_consolidating(self, rates: np.ndarray, atr: float) -> bool:
        """Check if consolidating."""
        if len(rates) < 20:
            return False
        range_size = float(np.max(rates["high"][-20:]) - np.min(rates["low"][-20:]))
        factor = float(vol_entry_controller.get("consolidation_atr_factor"))
        return atr > 0 and range_size < (atr * factor)

    def _should_enter_now(self, volatility_state: str, in_consolidation: bool) -> Tuple[bool, Optional[str]]:
        """Determine if we should enter now."""
        if volatility_state == "EXPANDING":
            return True, None
        if volatility_state == "CONTRACTING" and in_consolidation:
            return False, "Consolidation + contracting – wait for breakout"
        if volatility_state == "LOW_VOLATILE" and in_consolidation:
            return False, "Low volatility consolidation – wait for expansion"
        return True, None

    def _calculate_optimal_entry_range(
        self,
        proposed_entry: float,
        atr: float,
        direction: str,
    ) -> Tuple[float, float]:
        """Calculate optimal entry range."""
        factor = float(vol_entry_controller.get("optimal_range_factor"))
        range_width = atr * factor
        if direction.upper() == "BUY":
            return proposed_entry - range_width, proposed_entry
        return proposed_entry, proposed_entry + range_width

    def _refine_entry_price(
        self,
        proposed_entry: float,
        atr: float,
        direction: str,
        volatility_state: str,
    ) -> float:
        """Refine entry price."""
        adjust_factor = float(vol_entry_controller.get("refine_adjust_factor"))
        if volatility_state == "EXPANDING":
            delta = atr * adjust_factor
        elif volatility_state == "HIGH_VOLATILE":
            delta = atr * (adjust_factor * 0.5)
        else:
            delta = atr * (adjust_factor * 0.8)
        return proposed_entry - delta if direction.upper() == "BUY" else proposed_entry + delta

    def _calculate_entry_confidence(
        self,
        volatility_state: str,
        in_consolidation: bool,
    ) -> float:
        """Calculate confidence score."""
        confidence = float(vol_entry_controller.get("min_confidence"))
        if volatility_state == "EXPANDING":
            confidence += 30
        elif volatility_state == "CONTRACTING":
            confidence -= 20
        if in_consolidation:
            confidence -= 25
        return max(0.0, min(100.0, confidence))

    def _calculate_atr_factor(self) -> float:
        """Calculate ATR factor."""
        return 0.5

    def _default_entry(self, proposed_entry: float) -> VolatilityEntry:
        """Default entry."""
        return VolatilityEntry(
            True,
            proposed_entry,
            50.0,
            0.0,
            "UNKNOWN",
            None,
            (proposed_entry, proposed_entry)
        )

    def get_position_size_adjustment(self, volatility_state: str) -> float:
        """Get position size adjustment."""
        if volatility_state == "HIGH_VOLATILE":
            return float(vol_entry_controller.get("high_vol_multiplier"))
        if volatility_state == "LOW_VOLATILE":
            return float(vol_entry_controller.get("low_vol_multiplier"))
        return 1.0
