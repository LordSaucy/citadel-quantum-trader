#!/usr/bin/env python3
"""
CAPITAL MANAGEMENT SYSTEM

Manages capital allocation, deployment, risk‑per‑trade sizing and
profit‑withdrawal strategies.  Includes a runtime‑tunable controller
that publishes Prometheus metrics and offers a tiny Flask HTTP API so
Grafana (or any external tool) can read / modify the parameters
without restarting the bot.

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
from typing import Dict, Optional

# ----------------------------------------------------------------------
# Third‑party
# ----------------------------------------------------------------------
import flask                     # pip install flask
from flask import Flask, jsonify, request, abort
from prometheus_client import Gauge   # already a dependency of the project

# ----------------------------------------------------------------------
# Logging configuration (uses the global logging config of the app)
# ----------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# 1️⃣  Data structures
# ----------------------------------------------------------------------
@dataclass
class CapitalAllocation:
    """Capital allocation breakdown."""
    total_capital: float
    deployed_capital: float
    reserve_capital: float
    deployed_pct: float
    reserve_pct: float


# ----------------------------------------------------------------------
# 2️⃣  Core capital manager
# ----------------------------------------------------------------------
class CapitalManager:
    """
    Manages total capital, deployment vs. reserve and provides helpers
    for risk‑per‑trade and position‑size calculations.
    """

    def __init__(self, total_capital: float, deployment_pct: float = 80.0):
        """
        Initialise the manager.

        Args:
            total_capital: Total trading capital (USD‑equivalent).
            deployment_pct: Percentage of capital to be deployed (rest is reserve).
        """
        self.total_capital = total_capital
        self.deployment_pct = deployment_pct
        self._recalc_deployment()

        logger.info("Capital Manager initialised:")
        logger.info(f"  Total   : ${self.total_capital:,.2f}")
        logger.info(f"  Deployed: ${self.deployed_capital:,.2f} ({self.deployment_pct:.0f} %)")
        logger.info(f"  Reserve : ${self.reserve_capital:,.2f} ({100-self.deployment_pct:.0f} %)")

    # ------------------------------------------------------------------
    def _recalc_deployment(self) -> None:
        """Re‑calculate deployed / reserve amounts from the current totals."""
        self.deployed_capital = self.total_capital * (self.deployment_pct / 100.0)
        self.reserve_capital = self.total_capital - self.deployed_capital

    # ------------------------------------------------------------------
    def get_deployed_capital(self) -> float:
        """Current amount that is allowed to be used for trading."""
        return self.deployed_capital

    # ------------------------------------------------------------------
    def get_total_capital(self) -> float:
        """Total capital (deployed + reserve)."""
        return self.total_capital

    # ------------------------------------------------------------------
    def get_recommended_risk_pct(self) -> float:
        """
        Recommended risk per trade (percentage of *deployed* capital).

        The rule‑of‑thumb scales with account size:
            • < $10 k  → 5‑7 %
            • $10‑50 k → 3‑5 %
            • > $50 k → 2‑3 %
        """
        if self.deployed_capital < 10_000:
            return 5.0
        if self.deployed_capital < 50_000:
            return 3.5
        return 2.5

    # ------------------------------------------------------------------
    def calculate_position_size(self,
                                risk_pct: float,
                                entry: float,
                                stop_loss: float,
                                pip_value: float = 0.0001) -> float:
        """
        Return the position size in *lots*.

        Args:
            risk_pct:   Risk percentage of deployed capital to use.
            entry:      Entry price.
            stop_loss:  Stop‑loss price.
            pip_value:  Size of one pip for the instrument (default 0.0001 for most FX).

        Returns:
            Position size rounded to two decimal places.
        """
        risk_amount = self.deployed_capital * (risk_pct / 100.0)
        risk_pips = abs(entry - stop_loss) / pip_value

        if risk_pips == 0:
            logger.warning("Zero distance between entry and stop‑loss – returning 0 lot")
            return 0.0

        # Simplified lot calculation – 100 000 units per standard lot.
        lot_size = risk_amount / (risk_pips * 100_000.0)
        lot_size = round(lot_size, 2)

        logger.debug(
            f"Calculated lot size: {lot_size} lots "
            f"(risk ${risk_amount:,.2f}, distance {risk_pips:.1f} pips)"
        )
        return lot_size

    # ------------------------------------------------------------------
    def update_capital(self, new_total: float) -> None:
        """Replace total capital and recompute deployment / reserve."""
        self.total_capital = new_total
        self._recalc_deployment()
        logger.info(f"Capital updated: ${new_total:,.2f}")
        logger.info(f"  Deployed: ${self.deployed_capital:,.2f}")
        logger.info(f"  Reserve : ${self.reserve_capital:,.2f}")

    # ------------------------------------------------------------------
    def get_allocation(self) -> CapitalAllocation:
        """Return a snapshot of the current allocation."""
        return CapitalAllocation(
            total_capital=self.total_capital,
            deployed_capital=self.deployed_capital,
            reserve_capital=self.reserve_capital,
            deployed_pct=self.deployment_pct,
            reserve_pct=100.0 - self.deployment_pct,
        )


# ----------------------------------------------------------------------
# 3️⃣  Withdrawal strategy
# ----------------------------------------------------------------------
class WithdrawalStrategy:
    """
    Handles profit‑withdrawal policies.  Three preset strategies are
    provided (aggressive, balanced, conservative) but the percentages
    can be changed at runtime via the controller.
    """

    STRATEGIES = {
        "aggressive": {
            "name": "Aggressive Growth",
            "withdraw_pct": 0,          # reinvest everything
            "description": "Compound all profits for maximum growth",
        },
        "balanced": {
            "name": "Balanced",
            "withdraw_pct": 50,         # withdraw half of profits
            "description": "Take regular income while still growing capital",
        },
        "conservative": {
            "name": "Conservative",
            "withdraw_pct": 75,         # withdraw most profits
            "description": "Preserve capital, withdraw majority of gains",
        },
    }

    def __init__(self, strategy: str = "balanced"):
        self.strategy_key = strategy if strategy in self.STRATEGIES else "balanced"
        self.strategy = self.STRATEGIES[self.strategy_key]
        logger.info(f"Withdrawal strategy set to: {self.strategy['name']}")

    # ------------------------------------------------------------------
    def set_strategy(self, strategy: str) -> None:
        """Switch to another predefined strategy at runtime."""
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy '{strategy}'. Choose from {list(self.STRATEGIES)}")
        self.strategy_key = strategy
        self.strategy = self.STRATEGIES[strategy]
        logger.info(f"Withdrawal strategy changed to: {self.strategy['name']}")

    # ------------------------------------------------------------------
    def calculate_withdrawal(self, profits: float, capital: float) -> float:
        """
        Compute the amount to withdraw given the current profit figure.

        The algorithm never lets the reserve drop below 90 % of the total
        capital (protects the core bankroll).
        """
        withdraw_pct = self.strategy["withdraw_pct"]
        if withdraw_pct == 0:
            return 0.0

        withdrawal = profits * (withdraw_pct / 100.0)

        # Protect the core capital – keep at least 90 % of total in reserve
        min_allowed_reserve = capital * 0.90
        if capital - withdrawal < min_allowed_reserve:
            withdrawal = max(0.0, capital - min_allowed_reserve)

        logger.debug(
            f"Withdrawal calculation: profits=${profits:,.2f}, "
            f"pct={withdraw_pct} → withdraw=${withdrawal:,.2f}"
        )
        return withdrawal

    # ------------------------------------------------------------------
    def get_strategy_description(self) -> str:
        """Human‑readable description of the active strategy."""
        return f"{self.strategy['name']}: {self.strategy['description']}"


# ----------------------------------------------------------------------
# 4️⃣  Runtime‑tunable controller (Mission‑Control)
# ----------------------------------------------------------------------
class CapitalController:
    """
    Holds all tunable parameters for the capital subsystem, exposes them
    as Prometheus gauges and provides a tiny Flask API so Grafana (or any
    external UI) can read / modify them in real time.

    The configuration is persisted to JSON on disk (mounted volume) so
    changes survive container restarts.
    """

    # ------------------------------------------------------------------
    # Default configuration – can be overridden at runtime via the API.
    # ------------------------------------------------------------------
    DEFAULTS = {
        # ----- deployment -------------------------------------------------
        "deployment_pct": 80.0,          # % of total capital to deploy
        # ----- risk thresholds -------------------------------------------
        "min_risk_pct_small": 5.0,      # < $10k accounts
        "min_risk_pct_medium": 3.5,    # $10k‑$50k accounts
        "min_risk_pct_large": 2.5,     # > $50k accounts
        # ----- withdrawal -------------------------------------------------
        "withdrawal_strategy": "balanced",   # default strategy key
        # ----- debug -----------------------------------------------------
        "debug": False,
    }

    CONFIG_PATH = Path("/app/config/capital_config.json")   # <- bind‑mount a volume

    # ------------------------------------------------------------------
    # Prometheus gauges – one per config key
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
                logger.info(f"CapitalController – loaded config from {self.CONFIG_PATH}")
            except Exception as exc:   # pragma: no cover
                logger.error(f"Failed to read capital config – using defaults ({exc})")
                self.values = self.DEFAULTS.copy()
        else:
            logger.info("CapitalController – no config file, using defaults")
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
            logger.error(f"Could not persist capital config: {exc}")

    # ------------------------------------------------------------------
    # Register a Prometheus gauge for each key
    # ------------------------------------------------------------------
    def _register_gauges(self) -> None:
        for key, val in self.values.items():
            g = Gauge(
                "capital_parameter",
                "Runtime‑tunable capital‑management parameter",
                ["parameter"],
            )
            g.labels(parameter=key).set(val)
            self._gauges[key] = g

    # ------------------------------------------------------------------
    # Update a single key (used by the API and by the file‑watcher)
    # ------------------------------------------------------------------
    def set(self, key: str, value: float) -> None:
        if key not in self.values:
            raise KeyError(f"Unknown capital parameter: {key}")

        try:
            value = float(value)
        except ValueError as exc:
            raise ValueError(f"Invalid numeric value for {key}") from exc

        self.values[key] = value
        self._gauges[key].labels(parameter=key).set(value)
        self._persist()
        logger.info(f"CapitalController – set {key} = {value}")

    # ------------------------------------------------------------------
    # Read‑only accessor (used by the rest of the system)
    # ------------------------------------------------------------------
    def get(self, key: str) -> float:
        return self.values[key]

    # ------------------------------------------------------------------
    # File‑watcher – reloads the JSON if someone edited it manually
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
                        logger.info("CapitalController – config file changed, reloading")
                        self._load_or_initialize()
                        for k, v in self.values.items():
                            self._gauges[k].labels(parameter=k).set(v)
                        last_mtime = mtime
                sleep(2)

        Thread(target=_watch, daemon=True, name="capital-config-watcher").start()

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
            # 0.0.0.0 so it is reachable from the host / Grafana
            app.run(host="0.0.0.0", port=5006, debug=False, use_reloader=False)

        Thread(target=_run, daemon=True, name="capital-flask-api").start()

    # ------------------------------------------------------------------
    # Graceful shutdown (called from the main process on SIGTERM)
    # ------------------------------------------------------------------
    def stop(self) -> None:
        self._stop_event.set()


# ----------------------------------------------------------------------
# 5️⃣  Global singletons – importable from any other module
# ----------------------------------------------------------------------
capital_manager = CapitalManager(total_capital=100_000.0, deployment_pct=80.0)
withdrawal_strategy = WithdrawalStrategy(strategy="balanced")
capital_controller = CapitalController()

