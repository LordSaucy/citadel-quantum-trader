#!/usr/bin/env python3
"""
AOI Parameter Controller

Keeps the three AOI thresholds (MIN_TOUCHES, PRICE_TOLERANCE,
AT_AOI_TOLERANCE) live, exposes them as Prometheus gauges and via a tiny
HTTP API so Grafana can read / write them in real time.
"""

import json
import logging
from pathlib import Path
from threading import Event, Thread
from time import sleep
from typing import Dict

from flask import Flask, jsonify, request, abort
from prometheus_client import Gauge

logger = logging.getLogger(__name__)

class AOIController:
    """
    Holds mutable AOI parameters, publishes them as Prometheus gauges,
    persists them to disk and offers a Flask API for live updates.
    """

    # ------------------------------------------------------------------
    # Default values (these are the same numbers you originally hard‑coded)
    # ------------------------------------------------------------------
    DEFAULTS: Dict[str, float] = {
        "min_touches": 3,          # integer but stored as float for the gauge
        "price_tolerance": 0.003,  # 0.3 %
        "at_aoi_tolerance": 0.002, # 0.2 %
    }

    # Where the JSON file lives – mount a volume to /app/config in Docker
    CONFIG_PATH = Path("/app/config/aoi_params.json")

    # ------------------------------------------------------------------
    def __init__(self) -> None:
        self._stop = Event()
        self._load_or_create()
        self._register_gauges()
        self._start_watcher()
        self._start_api()

    # ------------------------------------------------------------------
    # Load persisted values or fall back to defaults
    # ------------------------------------------------------------------
    def _load_or_create(self) -> None:
        if self.CONFIG_PATH.is_file():
            try:
                with open(self.CONFIG_PATH, "r") as f:
                    self.values = json.load(f)
                logger.info(f"AOIController – loaded config from {self.CONFIG_PATH}")
            except Exception as exc:   # pragma: no cover
                logger.error(f"Failed to read AOI config – using defaults ({exc})")
                self.values = self.DEFAULTS.copy()
        else:
            logger.info("AOIController – no config file, using defaults")
            self.values = self.DEFAULTS.copy()
            self._persist()

    # ------------------------------------------------------------------
    # Persist the dict to JSON (called after every change)
    # ------------------------------------------------------------------
    def _persist(self) -> None:
        try:
            self.CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(self.CONFIG_PATH, "w") as f:
                json.dump(self.values, f, indent=2)
        except Exception as exc:   # pragma: no cover
            logger.error(f"Could not persist AOI config: {exc}")

    # ------------------------------------------------------------------
    # Register a Prometheus gauge for each key
    # ------------------------------------------------------------------
    def _register_gauges(self) -> None:
        self._gauges: Dict[str, Gauge] = {}
        for key, val in self.values.items():
            g = Gauge(
                "aoi_parameter",
                "Tunable AOI parameter (runtime)",
                ["parameter"],
            )
            g.labels(parameter=key).set(val)
            self._gauges[key] = g

    # ------------------------------------------------------------------
    # Public getter (used by the validator)
    # ------------------------------------------------------------------
    def get(self, key: str) -> float:
        """Return the current value for *key* (raises KeyError if unknown)."""
        if key not in self.values:
            raise KeyError(f"AOI parameter '{key}' not recognised")
        return float(self.values[key])

    # ------------------------------------------------------------------
    # Public setter – updates dict, gauge and persists
    # ------------------------------------------------------------------
    def set(self, key: str, value: float) -> None:
        if key not in self.values:
            raise KeyError(f"AOI parameter '{key}' not recognised")
        self.values[key] = float(value)
        self._gauges[key].labels(parameter=key).set(value)
        self._persist()
        logger.info(f"AOIController – set {key} = {value}")

    # ------------------------------------------------------------------
    # File‑watcher – reload if the JSON file is edited manually
    # ------------------------------------------------------------------
    def _start_watcher(self) -> None:
        def _watch():
            last_mtime = self.CONFIG_PATH.stat().st_mtime if self.CONFIG_PATH.is_file() else 0
            while not self._stop.is_set():
                if self.CONFIG_PATH.is_file():
                    mtime = self.CONFIG_PATH.stat().st_mtime
                    if mtime != last_mtime:
                        logger.info("AOIController – config file changed, reloading")
                        self._load_or_create()
                        for k, v in self.values.items():
                            self._gauges[k].labels(parameter=k).set(v)
                        last_mtime = mtime
                sleep(2)

        Thread(target=_watch, daemon=True, name="aoi-config-watcher").start()

    # ------------------------------------------------------------------
    # Flask API – expose GET/POST endpoints
    # ------------------------------------------------------------------
    def _start_api(self) -> None:
        app = Flask(__name__)

        @app.route("/aoi/config", methods=["GET"])
        def get_all():
            return jsonify(self.values)

        @app.route("/aoi/config/<key>", methods=["GET"])
        def get_one(key: str):
            try:
                return jsonify({key: self.get(key)})
            except KeyError:
                abort(404, description=f"Parameter {key} not found")

        @app.route("/aoi/config/<key>", methods=["POST", "PUT", "PATCH"])
        def set_one(key: str):
            try:
                payload = request.get_json(force=True)
                if not payload or "value" not in payload:
                    abort(400, description="JSON body must contain 'value'")
                self.set(key, payload["value"])
                return jsonify({key: self.get(key)})
            except Exception as exc:   # pragma: no cover
                logger.error(f"AOI API error: {exc}")
                abort(500, description=str(exc))

        @app.route("/healthz", methods=["GET"])
        def health():
            return "OK", 200

        def _run():
            # Bind to 0.0.0.0 so Grafana (outside the container) can reach it
            app.run(host="0.0.0.0", port=5006, debug=False, use_reloader=False)

        Thread(target=_run, daemon=True, name="aoi-flask-api").start()

    # ------------------------------------------------------------------
    def stop(self) -> None:
        self._stop.set()


# ----------------------------------------------------------------------
# Global singleton – import this from anywhere (e.g. aoi_validator.py)
# ----------------------------------------------------------------------
aoi_controller = AOIController()
