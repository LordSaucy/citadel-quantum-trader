#!/usr/bin/env python3
"""
MASTER WIN‑RATE OPTIMIZER

Integrates all 7 optimization levers for maximum win‑rate.
Acts as the master coordinator – every trade must pass through it.

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
from datetime import datetime
from pathlib import Path
from threading import Event, Thread
from typing import Dict, List, Tuple

# ----------------------------------------------------------------------
# Third‑party
# ----------------------------------------------------------------------
from flask import Flask, jsonify, request, abort   # pip install flask
from prometheus_client import Gauge               # already a dependency

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# 0️⃣  Helper – controller that holds tunable parameters
# ----------------------------------------------------------------------
class OptimizerController:
    """
    Holds all runtime‑tunable parameters for the Master Optimizer,
    persists them to JSON, registers a Prometheus gauge for each,
    and exposes a tiny Flask API (`GET /config`, `POST /config/<key>`).

    Grafana can read the gauges and write new values via the API,
    allowing live re‑tuning without restarting the bot.
    """

    # ------------------------------------------------------------------
    # Default values – these mirror the numbers used in the original
    # implementation.  They can be overridden at runtime via the API.
    # ------------------------------------------------------------------
    DEFAULTS: Dict[str, float] = {
        # ----- Weights (must sum to 1.0) ---------------------------------
        "weight_entry_quality": 0.25,
        "weight_regime": 0.15,
        "weight_mtf": 0.20,
        "weight_confluence": 0.25,
        "weight_session": 0.10,
        "weight_volatility": 0.05,
        # ----- Thresholds -------------------------------------------------
        "min_overall_score": 70.0,          # overall score needed to approve
        "min_entry_quality": 75.0,          # entry quality score threshold
        "min_mtf_alignment": 60.0,          # MTF alignment % needed
        "min_confluence_score": 70.0,       # confluence total score
        "min_session_liquidity": 60.0,      # session liquidity % needed
        "min_volatility_confidence": 50.0,  # volatility confidence %
        # ----- Risk adjustment -------------------------------------------
        "base_risk_multiplier": 1.0,        # global risk multiplier
        # ----- Debug ------------------------------------------------------
        "debug": 0,                         # 0 = off, 1 = on
    }

    CONFIG_PATH = Path("/app/config/optimizer_config.json")   # mounted volume

    # ------------------------------------------------------------------
    # Prometheus gauges – one per key
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
                logger.info(f"OptimizerController – loaded config from {self.CONFIG_PATH}")
            except Exception as exc:   # pragma: no cover
                logger.error(f"Failed to read optimizer config – using defaults ({exc})")
                self.values = self.DEFAULTS.copy()
        else:
            logger.info("OptimizerController – no config file, using defaults")
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
            logger.error(f"Could not persist optimizer config: {exc}")

    # ------------------------------------------------------------------
    # Register a Prometheus gauge for every key
    # ------------------------------------------------------------------
    def _register_gauges(self) -> None:
        for key, val in self.values.items():
            g = Gauge(
                "master_optimizer_parameter",
                "Runtime‑tunable parameter for the Master Win‑Rate Optimizer",
                ["parameter"],
            )
            g.labels(parameter=key).set(val)
            self._gauges[key] = g

    # ------------------------------------------------------------------
    # Update a single key (used by the API and file‑watcher)
    # ------------------------------------------------------------------
    def set(self, key: str, value: float) -> None:
        if key not in self.values:
            raise KeyError(f"Unknown optimizer parameter: {key}")

        try:
            value = float(value)
        except ValueError as exc:
            raise ValueError(f"Invalid numeric value for {key}") from exc

        self.values[key] = value
        self._gauges[key].labels(parameter=key).set(value)
        self._persist()
        logger.info(f"OptimizerController – set {key} = {value}")

    # ------------------------------------------------------------------
    # Read‑only accessor (used by the optimizer)
    # ------------------------------------------------------------------
    def get(self, key: str) -> float:
        return self.values[key]

    # ------------------------------------------------------------------
    # File‑watcher – reloads JSON if edited manually
    # ------------------------------------------------------------------
    def _start_file_watcher(self) -> None:
        def _watch():
            last_mtime = (
                self.CONFIG_PATH.stat().st_mtime
                if self.CONFIG_PATH.exists()
                else 0
            )
            while not self._stop_event.is_set():
                if self.CONFIG_PATH.exists():
                    mtime = self.CONFIG_PATH.stat().st_mtime
                    if mtime != last_mtime:
                        logger.info("OptimizerController – config file changed, reloading")
                        self._load_or_initialize()
                        for k, v in self.values.items():
                            self._gauges[k].labels(parameter=k).set(v)
                        last_mtime = mtime
                sleep(2)

        Thread(target=_watch, daemon=True, name="optimizer-config-watcher").start()

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
            # 0.0.0.0 so Grafana can reach it
            app.run(host="0.0.0.0", port=5006, debug=False, use_reloader=False)

        Thread(target=_run, daemon=True, name="optimizer-flask-api").start()

    # ------------------------------------------------------------------
    # Graceful shutdown (called from the main process on SIGTERM)
    # ------------------------------------------------------------------
    def stop(self) -> None:
        self._stop_event.set()


# ----------------------------------------------------------------------
# Global singleton – importable from any module
# ----------------------------------------------------------------------
optimizer_controller = OptimizerController()


# ----------------------------------------------------------------------
# 2️⃣  Dataclass that aggregates every lever’s result
# ----------------------------------------------------------------------
@dataclass
class ComprehensiveTradeAnalysis:
    """Complete trade analysis from all systems."""

    # Lever 1: Entry Quality
    entry_quality_score: float
    entry_quality_grade: str

    # Lever 2: Exit Plan
    exit_strategy: Dict

    # Lever 3: Market Regime
    market_regime: str
    regime_aligned: bool

    # Lever 4: MTF
    mtf_alignment_score: float
    mtf_approved: bool

    # Lever 5: Confluence
    confluence_score: float
    confluence_grade: str

    # Lever 6: Session
    session_approved: bool
    liquidity_score: float

    # Lever 7: Volatility Entry
    volatility_entry_approved: bool
    entry_confidence: float

    # Final decision
    overall_approved: bool
    overall_score: float
    final_recommendation: str
    risk_adjustment_multiplier: float


# ----------------------------------------------------------------------
# 3️⃣  Master Optimizer – orchestrates the seven levers
# ----------------------------------------------------------------------
class MasterWinRateOptimizer:
    """
    Integrates all 7 levers for maximum win‑rate optimization.
    Every trade must pass through ``comprehensive_trade_analysis``.
    """

    # ------------------------------------------------------------------
    # Constructor – lazy‑load the external subsystems
    # ------------------------------------------------------------------
    def __init__(self) -> None:
        # Import subsystems only when the optimizer is instantiated.
        # This avoids circular imports and makes unit‑testing easier.
        from advanced_entry_filter import entry_quality_filter
        from advanced_exit_system import dynamic_exit_manager
        from market_regime_detector import market_regime_detector
        from multi_timeframe_confirmation import mtf_confirmation
        from advanced_confluence_system import confluence_system
        from session_time_filter import session_filter
        from volatility_entry_refinement import volatility_entry

        self.entry_quality = entry_quality_filter
        self.exit_manager = dynamic_exit_manager
        self.regime_detector = market_regime_detector
        self.mtf = mtf_confirmation
        self.confluence = confluence_system
        self.session = session_filter
        self.volatility = volatility_entry

        logger.info("🎯 Master Win‑Rate Optimizer initialized with all 7 levers")

    # ------------------------------------------------------------------
    # Helper – fetch a tunable weight/threshold from the controller
    # ------------------------------------------------------------------
    def _param(self, name: str, fallback: float) -> float:
        """
        Retrieve a numeric parameter from ``optimizer_controller``.
        If the key does not exist we return the supplied ``fallback``.
        """
        try:
            return optimizer_controller.get(name)
        except Exception:                     # pragma: no cover
            return fallback

    # ------------------------------------------------------------------
    # QUICK PRE‑CHECK – cheap filters before the full analysis
    # ------------------------------------------------------------------
    def quick_check(self, symbol: str, direction: str) -> Tuple[bool, str]:
        """
        Fast pre‑check that runs only the cheapest levers (session &
        regime).  Returns ``(can_proceed, reason)``.
        """
        # 1️⃣ Session filter (very cheap)
        session = self.session.analyze_current_session(symbol)
        if not session.should_trade:
            return False, f"Session rejected: {session.reasoning}"

        # 2️⃣ Regime filter
        regime = self.regime_detector.analyze_market_regime(symbol)
        can_trade, reason = self.regime_detector.should_trade_in_current_regime(direction)
        if not can_trade:
            return False, f"Regime rejected: {reason}"

        return True, "Pre‑checks passed – proceeding to full analysis"

    # ------------------------------------------------------------------
    # MAIN ANALYSIS – runs all 7 levers, applies tunable weights,
    #                 and returns a ``ComprehensiveTradeAnalysis`` object.
    # ------------------------------------------------------------------
    def comprehensive_trade_analysis(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
    ) -> ComprehensiveTradeAnalysis:
        """
        Run the full 7‑lever analysis for ``symbol`` in ``direction``.
        Returns a ``ComprehensiveTradeAnalysis`` dataclass instance.
        """
        logger.info("=" * 80)
        logger.info(f"🔍 COMPREHENSIVE ANALYSIS: {symbol} {direction} @ {entry_price}")
        logger.info("=" * 80)

        # --------------------------------------------------------------
        # 1️⃣ Entry Quality
        # --------------------------------------------------------------
        logger.info("\n📊 LEVER 1: Entry Quality Analysis...")
        entry_analysis = self.entry_quality.assess_entry_quality(
            symbol, direction, entry_price, stop_loss
        )
        logger.info(f"  Score: {entry_analysis.total_score:.1f}/100")
        logger.info(f"  {entry_analysis.reasoning}")

        # --------------------------------------------------------------
        # 2️⃣ Exit Strategy (setup only – live monitoring elsewhere)
        # --------------------------------------------------------------
        logger.info("\n🎯 LEVER 2: Exit Strategy Setup...")
        exit_strategy = {
            "dynamic_exits": True,
            "momentum_monitoring": True,
            "pattern_recognition": True,
        }
        logger.info("  ✅ Dynamic exit system ready")

        # --------------------------------------------------------------
        # 3️⃣ Market Regime
        # --------------------------------------------------------------
        logger.info("\n🌐 LEVER 3: Market Regime Detection...")
        regime_analysis = self.regime_detector.analyze_market_regime(symbol)
        regime_aligned, regime_reason = self.regime_detector.should_trade_in_current_regime(
            direction
        )
        logger.info(f"  Regime: {regime_analysis.regime.value}")
        logger.info(f"  Aligned: {regime_aligned} - {regime_reason}")

        # --------------------------------------------------------------
        # 4️⃣ Multi‑Timeframe Confirmation
        # --------------------------------------------------------------
        logger.info("\n⏰ LEVER 4: Multi‑Timeframe Analysis...")
        mtf_approved, mtf_reason, mtf_analysis = self.mtf.should_approve_trade(
            symbol, direction, entry_price
        )
        logger.info(f"  Alignment: {mtf_analysis.alignment_score:.1f}%")
        logger.info(f"  Approved: {mtf_approved} - {mtf_reason}")

        # --------------------------------------------------------------
        # 5️⃣ Advanced Confluence
        # --------------------------------------------------------------
        logger.info("\n🎲 LEVER 5: Confluence Analysis...")
        confluence_analysis = self.confluence.analyze_confluence(
            symbol,
            direction,
            entry_price,
            stop_loss,
            mtf_data={"alignment_score": mtf_analysis.alignment_score},
        )
        logger.info(f"  Score: {confluence_analysis.total_score:.1f}/100")
        logger.info(f"  Grade: {confluence_analysis.grade}")
        logger.info(
            f"  Factors: {confluence_analysis.num_factors_present} present"
        )

        # --------------------------------------------------------------
        # 6️⃣ Session / Time Filter
        # --------------------------------------------------------------
        logger.info("\n🕐 LEVER 6: Session Analysis...")
        session_analysis = self.session.analyze_current_session(symbol)
        logger.info(f"  Session: {session_analysis.current_session.value}")
        logger.info(f"  Liquidity: {session_analysis.liquidity_score:.0f}/100")
        logger.info(f"  Approved: {session_analysis.should_trade}")

        # --------------------------------------------------------------
        # 7️⃣ Volatility Entry Timing
        # --------------------------------------------------------------
        logger.info("\n📈 LEVER 7: Volatility Entry Analysis...")
        volatility_analysis = self.volatility.analyze_entry_timing(
            symbol, direction, entry_price
        )
        logger.info(f"  State: {volatility_analysis.volatility_state}")
        logger.info(f"  Enter Now: {volatility_analysis.should_enter_now}")
        logger.info(f"  Confidence: {volatility_analysis.confidence:.1f}%")

        # --------------------------------------------------------------
        # FINAL DECISION – weighted overall score
        # --------------------------------------------------------------
        logger.info("\n" + "=" * 80)
        logger.info("📋 FINAL DECISION")
        logger.info("=" * 80)

        # ----- fetch tunable weights / thresholds --------------------
        w_entry = self._param("weight_entry_quality", 0.25)
        w_regime = self._param("weight_regime", 0.15)
        w_mtf = self._param("weight_mtf", 0.20)
        w_confluence = self._param("weight_confluence", 0.25)
        w_session = self._param("weight_session", 0.10)
        w_volatility = self._param("weight_volatility", 0.05)

        # ----- compute weighted overall score -------------------------
        overall_score = (
            entry_analysis.total_score * w_entry
            + (100 if regime_aligned else 30) * w_regime
            + mtf_analysis.alignment_score * w_mtf
            + confluence_analysis.total_score * w_confluence
            + session_analysis.liquidity_score * w_session
            + volatility_analysis.confidence * w_volatility
        )

        # ----- thresholds -----------------------------------------------
       # ----- thresholds -------------------------------------------------
        min_overall = self._param("min_overall_score", 70.0)
        min_entry = self._param("min_entry_quality", 75.0)
        min_mtf = self._param("min_mtf_alignment", 60.0)
        min_confluence = self._param("min_confluence_score", 70.0)
        min_session_liq = self._param("min_session_liquidity", 60.0)
        min_vol_conf = self._param("min_volatility_confidence", 50.0)

        # ----- evaluate each lever against its threshold -----------------
        passed_checks: List[str] = []
        failed_checks: List[str] = []

        # Lever 1 – Entry quality
        if entry_analysis.total_score >= min_entry:
            passed_checks.append("Entry quality ✅")
        else:
            failed_checks.append(
                f"Entry quality ❌ ({entry_analysis.total_score:.1f}< {min_entry})"
            )

        # Lever 2 – Regime alignment
        if regime_aligned:
            passed_checks.append("Regime aligned ✅")
        else:
            failed_checks.append(f"Regime mis‑aligned ❌ ({regime_reason})")

        # Lever 3 – MTF
        if mtf_approved and mtf_analysis.alignment_score >= min_mtf:
            passed_checks.append("MTF ✅")
        else:
            failed_checks.append(
                f"MTF ❌ (align={mtf_analysis.alignment_score:.1f}% < {min_mtf})"
            )

        # Lever 4 – Confluence
        if confluence_analysis.should_trade and confluence_analysis.total_score >= min_confluence:
            passed_checks.append("Confluence ✅")
        else:
            failed_checks.append(
                f"Confluence ❌ (score={confluence_analysis.total_score:.1f}< {min_confluence})"
            )

        # Lever 5 – Session / Liquidity
        if session_analysis.should_trade and session_analysis.liquidity_score >= min_session_liq:
            passed_checks.append("Session ✅")
        else:
            failed_checks.append(
                f"Session ❌ (liq={session_analysis.liquidity_score:.1f}< {min_session_liq})"
            )

        # Lever 6 – Volatility entry timing
        if volatility_analysis.should_enter_now and volatility_analysis.confidence >= min_vol_conf:
            passed_checks.append("Volatility ✅")
        else:
            failed_checks.append(
                f"Volatility ❌ (conf={volatility_analysis.confidence:.1f}% < {min_vol_conf})"
            )

        # ----- overall approval -----------------------------------------
        overall_approved = (overall_score >= min_overall) and (len(failed_checks) <= 1)

        # ----- risk‑adjustment multiplier --------------------------------
        # Base multiplier can be tuned at runtime; we also apply a tiny
        # regime‑based tweak (example only – can be expanded later).
        base_multiplier = self._param("base_risk_multiplier", 1.0)

        # Example: tighten risk a bit on bearish regimes when short‑selling
        regime_adj = 0.9 if (direction == "SELL" and regime_analysis.regime.value == "BEARISH") else 1.0
        session_adj = 0.95 if session_analysis.liquidity_score < 50 else 1.0

        risk_multiplier = base_multiplier * regime_adj * session_adj

        # ----- human‑readable recommendation -----------------------------
        if overall_approved:
            recommendation = "✅ TRADE APPROVED\n"
            recommendation += f"Overall score: {overall_score:.1f}/100 (≥ {min_overall})\n"
            recommendation += f"Passed checks ({len(passed_checks)}/6):\n"
            for chk in passed_checks:
                recommendation += f"  • {chk}\n"
            if failed_checks:
                recommendation += "Minor concerns:\n"
                for chk in failed_checks:
                    recommendation += f"  • {chk}\n"
            recommendation += f"\nRisk multiplier: {risk_multiplier:.2f}×"
        else:
            recommendation = "❌ TRADE REJECTED\n"
            recommendation += f"Overall score: {overall_score:.1f}/100 (< {min_overall})\n"
            recommendation += f"Failed checks ({len(failed_checks)}):\n"
            for chk in failed_checks:
                recommendation += f"  • {chk}\n"
            recommendation += "\nAwait better market conditions."

        logger.info(recommendation)
        logger.info("=" * 80 + "\n")

        # ------------------------------------------------------------------
        # Return the dataclass instance with every piece of information
        # ------------------------------------------------------------------
        return ComprehensiveTradeAnalysis(
            entry_quality_score=entry_analysis.total_score,
            entry_quality_grade=entry_analysis.reasoning.split("\n")[0].split(":")[1].strip().split()[0],
            exit_strategy=exit_strategy,
            market_regime=regime_analysis.regime.value,
            regime_aligned=regime_aligned,
            mtf_alignment_score=mtf_analysis.alignment_score,
            mtf_approved=mtf_approved,
            confluence_score=confluence_analysis.total_score,
            confluence_grade=confluence_analysis.grade,
            session_approved=session_analysis.should_trade,
            liquidity_score=session_analysis.liquidity_score,
            volatility_entry_approved=volatility_analysis.should_enter_now,
            entry_confidence=volatility_analysis.confidence,
            overall_approved=overall_approved,
            overall_score=overall_score,
            final_recommendation=recommendation,
            risk_adjustment_multiplier=risk_multiplier,
        )
       # --------------------------------------------------------------
        # 8️⃣  Final approval logic (passes / fails)
        # --------------------------------------------------------------
        passed_checks: List[str] = []
        failed_checks: List[str] = []

        # ---- Entry quality ------------------------------------------------
        min_eq = self._param("min_entry_quality", 75.0)
        if entry_analysis.total_score >= min_eq:
            passed_checks.append("Entry quality ✅")
        else:
            failed_checks.append(
                f"Entry quality ❌ ({entry_analysis.total_score:.1f}< {min_eq})"
            )

        # ---- Regime alignment --------------------------------------------
        if regime_aligned:
            passed_checks.append("Regime aligned ✅")
        else:
            failed_checks.append(f"Regime mis‑aligned ❌ ({regime_reason})")

        # ---- MTF -----------------------------------------------------------
        min_mtf = self._param("min_mtf_alignment", 60.0)
        if mtf_approved and mtf_analysis.alignment_score >= min_mtf:
            passed_checks.append("MTF ✅")
        else:
            failed_checks.append(
                f"MTF ❌ (align={mtf_analysis.alignment_score:.1f}% < {min_mtf})"
            )

        # ---- Confluence ----------------------------------------------------
        min_cf = self._param("min_confluence_score", 70.0)
        if confluence_analysis.should_trade and confluence_analysis.total_score >= min_cf:
            passed_checks.append("Confluence ✅")
        else:
            failed_checks.append(
                f"Confluence ❌ (score={confluence_analysis.total_score:.1f}< {min_cf})"
            )

        # ---- Session -------------------------------------------------------
        min_liq = self._param("min_session_liquidity", 60.0)
        if session_analysis.should_trade and session_analysis.liquidity_score >= min_liq:
            passed_checks.append("Session ✅")
        else:
            failed_checks.append(
                f"Session ❌ (liq={session_analysis.liquidity_score:.1f}< {min_liq})"
            )

        # ---- Volatility ----------------------------------------------------
        min_vol = self._param("min_volatility_confidence", 50.0)
        if volatility_analysis.should_enter_now and volatility_analysis.confidence >= min_vol:
            passed_checks.append("Volatility ✅")
        else:
            failed_checks.append(
                f"Volatility ❌ (conf={volatility_analysis.confidence:.1f}< {min_vol})"
            )

        # ---- Overall approval -----------------------------------------------
        min_overall = self._param("min_overall_score", 70.0)
        overall_approved = (overall_score >= min_overall) and (len(failed_checks) <= 1)

        # --------------------------------------------------------------
        # 9️⃣  Risk‑adjustment multiplier (global + optional regime / session tweaks)
        # --------------------------------------------------------------
        base_risk = self._param("base_risk_multiplier", 1.0)

        # Example: tighten risk a bit if regime is bearish and we are short‑selling
        regime_risk_adj = 0.9 if (direction == "SELL" and regime_analysis.regime.value == "BEARISH") else 1.0
        session_risk_adj = 0.95 if session_analysis.liquidity_score < 50 else 1.0

        risk_multiplier = base_risk * regime_risk_adj * session_risk_adj

        # --------------------------------------------------------------
        # 10️⃣  Build the human‑readable recommendation
        # --------------------------------------------------------------
        if overall_approved:
            recommendation = "✅ TRADE APPROVED\n"
            recommendation += f"Overall score: {overall_score:.1f}/100 (>= {min_overall})\n"
            recommendation += f"Passed checks ({len(passed_checks)}/6):\n"
            for chk in passed_checks:
                recommendation += f"  • {chk}\n"
            if failed_checks:
                recommendation += "Minor concerns:\n"
                for chk in failed_checks:
                    recommendation += f"  • {chk}\n"
            recommendation += f"\nRisk multiplier: {risk_multiplier:.2f}×"
        else:
            recommendation = "❌ TRADE REJECTED\n"
            recommendation += f"Overall score: {overall_score:.1f}/100 (< {min_overall})\n"
            recommendation += f"Failed checks ({len(failed_checks)}):\n"
            for chk in failed_checks:
                recommendation += f"  • {chk}\n"
            recommendation += "\nAwait better conditions."

        logger.info(recommendation)
        logger.info("=" * 80 + "\n")

        # --------------------------------------------------------------
        # 11️⃣  Return the dataclass instance
        # --------------------------------------------------------------
        return ComprehensiveTradeAnalysis(
            entry_quality_score=entry_analysis.total_score,
            entry_quality_grade=entry_analysis.reasoning.split("\n")[0].split(":")[1].strip().split()[0],
            exit_strategy=exit_strategy,
            market_regime=regime_analysis.regime.value,
            regime_aligned=regime_aligned,
            mtf_alignment_score=mtf_analysis.alignment_score,
            mtf_approved=mtf_approved,
            confluence_score=confluence_analysis.total_score,
            confluence_grade=confluence_analysis.grade,
            session_approved=session_analysis.should_trade,
            liquidity_score=session_analysis.liquidity_score,
            volatility_entry_approved=volatility_analysis.should_enter_now,
            entry_confidence=volatility_analysis.confidence,
            overall_approved=overall_approved,
            overall_score=overall_score,
            final_recommendation=recommendation,
            risk_adjustment_multiplier=risk_multiplier,
        )


# ----------------------------------------------------------------------
# Global singleton – importable from other modules (e.g. src/main.py)
# ----------------------------------------------------------------------
master_optimizer = MasterWinRateOptimizer()
