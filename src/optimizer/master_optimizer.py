#!/usr/bin/env python3
"""
MasterWinRateOptimizer – high‑level orchestration for CQT's risk‑adjusted win‑rate evaluation.

Glues together individual "lever" modules and produces a single recommendation 
plus numeric risk‑multiplier that downstream components consume.

✅ FIXED: Removed unused "name" parameter from _param stub method
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple, Protocol, runtime_checkable

from src.utils.common import utc_now
from src.optimizer.leverage_results import (
    EntryQualityResult,
    RegimeResult,
    MtfResult,
    ConfluenceResult,
    SessionResult,
    VolatilityResult,
)

# =====================================================================
# Logging
# =====================================================================
logger = logging.getLogger(__name__)


@runtime_checkable
class OptimizerControllerProto(Protocol):
    """Minimal interface the optimizer expects from the controller."""
    def get(self, name: str) -> float: ...


# =====================================================================
# Public dataclass – the single return value of the optimizer
# =====================================================================
@dataclass(frozen=True, slots=True)
class OverallResult:
    """Aggregated outcome of a full win‑rate evaluation."""
    overall_score: float
    overall_approved: bool
    risk_multiplier: float
    recommendation: str
    passed: List[str]
    failed: List[str]


# =====================================================================
# Master optimizer – thin façade delegating to the lever objects
# =====================================================================
class MasterWinRateOptimizer:
    """Orchestrates the full win‑rate pipeline."""

    def __init__(self) -> None:
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

        logger.info("🔧 MasterWinRateOptimizer instantiated")

    # =====================================================================
    # Public entry point
    # =====================================================================
    def comprehensive_trade_analysis(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
    ) -> OverallResult:
        """
        Run the full lever chain for a single trade and return a rich result.
        """
        logger.info(
            f"▶️ Analyzing {symbol} {direction} @ {entry_price:.5f} (SL={stop_loss:.5f})"
        )

        # Run each lever
        entry = self._run_entry_quality(symbol, direction, entry_price, stop_loss)
        regime = self._run_market_regime(symbol, direction)
        mtf = self._run_mtf(symbol, direction, entry_price)
        confluence = self._run_confluence(
            symbol, direction, entry_price, stop_loss, mtf
        )
        session = self._run_session(symbol)
        volatility = self._run_volatility(symbol, direction, entry_price)

        # Compute overall score
        overall_score = self._compute_overall_score(
            entry, regime, mtf, confluence, session, volatility
        )

        # Evaluate each lever against its thresholds
        passed, failed = self._evaluate_checks(
            entry,
            regime,
            mtf,
            confluence,
            session,
            volatility,
            overall_score,
        )

        # Overall approval & risk multiplier
        overall_approved = (
            overall_score >= self._param(70.0)  # ✅ FIXED: Use default only
        ) and (len(failed) <= 1)

        risk_multiplier = self._calc_risk_multiplier(direction, regime, session)

        # Human‑readable recommendation string
        recommendation = self._build_recommendation(
            overall_score,
            overall_approved,
            passed,
            failed,
            risk_multiplier,
        )

        logger.debug("✅ Optimizer recommendation built")
        return OverallResult(
            overall_score=overall_score,
            overall_approved=overall_approved,
            risk_multiplier=risk_multiplier,
            recommendation=recommendation,
            passed=passed,
            failed=failed,
        )

    # =====================================================================
    # Private lever runners
    # =====================================================================
    def _run_entry_quality(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
    ) -> EntryQualityResult:
        """Delegate to the entry‑quality lever."""
        logger.debug("🔍 Entry‑quality")
        return self.entry_quality.assess_entry_quality(
            symbol, direction, entry_price, stop_loss
        )

    def _run_market_regime(self, symbol: str, direction: str) -> RegimeResult:
        """Detect the macro regime and whether it aligns with the trade direction."""
        logger.debug("🌐 Market‑regime")
        regime = self.regime_detector.analyze_market_regime(symbol)
        aligned, reason = self.regime_detector.should_trade_in_current_regime(
            direction
        )
        return RegimeResult(regime, aligned, reason)

    def _run_mtf(
        self, symbol: str, direction: str, entry_price: float
    ) -> MtfResult:
        """Multi‑time‑frame confirmation."""
        logger.debug("⏰ MTF")
        approved, reason, analysis = self.mtf.should_approve_trade(
            symbol, direction, entry_price
        )
        return MtfResult(analysis.alignment_score, approved, reason)

    def _run_confluence(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        mtf: MtfResult,
    ) -> ConfluenceResult:
        """Aggregate all confluence signals (price action, SMC, etc.)."""
        logger.debug("🎲 Confluence")
        return self.confluence.analyze_confluence(
            symbol,
            direction,
            entry_price,
            stop_loss,
            mtf_data={"alignment_score": mtf.alignment_score},
        )

    def _run_session(self, symbol: str) -> SessionResult:
        """Session‑time filter – checks liquidity & time‑of‑day constraints."""
        logger.debug("🕐 Session")
        return self.session.analyze_current_session(symbol)

    def _run_volatility(
        self, symbol: str, direction: str, entry_price: float
    ) -> VolatilityResult:
        """Volatility‑entry refinement – decides if the market is calm enough."""
        logger.debug("📈 Volatility")
        return self.volatility.analyze_entry_timing(
            symbol, direction, entry_price
        )

    # =====================================================================
    # Scoring helpers
    # =====================================================================
    def _compute_overall_score(
        self,
        entry: EntryQualityResult,
        regime: RegimeResult,
        mtf: MtfResult,
        confluence: ConfluenceResult,
        session: SessionResult,
        volatility: VolatilityResult,
    ) -> float:
        """Linear weighted combination of the six lever scores."""
        w = {
            "entry": self._param(0.25),
            "regime": self._param(0.15),
            "mtf": self._param(0.20),
            "conf": self._param(0.25),
            "session": self._param(0.10),
            "vol": self._param(0.05),
        }

        overall = (
            entry.total_score * w["entry"]
            + (100 if regime.aligned else 30) * w["regime"]
            + mtf.alignment_score * w["mtf"]
            + confluence.total_score * w["conf"]
            + session.liquidity_score * w["session"]
            + volatility.confidence * w["vol"]
        )
        return overall

    # =====================================================================
    # Lever evaluation
    # =====================================================================

    def _check_lever(
        self,
        passed: List[str],
        failed: List[str],
        condition: bool,
        pass_msg: str,
        fail_msg: str,
    ) -> None:
        """Unified lever check logic to eliminate duplication."""
        if condition:
            passed.append(pass_msg)
        else:
            failed.append(fail_msg)

    def _evaluate_checks(
        self,
        entry: EntryQualityResult,
        regime: RegimeResult,
        mtf: MtfResult,
        confluence: ConfluenceResult,
        session: SessionResult,
        volatility: VolatilityResult,
        overall_score: float,
    ) -> Tuple[List[str], List[str]]:
        """Return two lists: levers that passed and levers that failed."""
        passed: List[str] = []
        failed: List[str] = []

        # ---- 1️⃣ Entry quality ----
        min_eq = self._param(75.0)
        self._check_lever(
            passed, failed,
            entry.total_score >= min_eq,
            "Entry quality ✅",
            f"Entry quality ❌ ({entry.total_score:.1f} < {min_eq})"
        )

        # ---- 2️⃣ Regime alignment ----
        self._check_lever(
            passed, failed,
            regime.aligned,
            "Regime aligned ✅",
            f"Regime mis‑aligned ❌ ({regime.reason})"
        )

        # ---- 3️⃣ MTF ----
        min_mtf = self._param(60.0)
        self._check_lever(
            passed, failed,
            mtf.approved and mtf.alignment_score >= min_mtf,
            "MTF ✅",
            f"MTF ❌ (align={mtf.alignment_score:.1f}% < {min_mtf})"
        )

        # ---- 4️⃣ Confluence ----
        min_cf = self._param(70.0)
        self._check_lever(
            passed, failed,
            confluence.should_trade and confluence.total_score >= min_cf,
            "Confluence ✅",
            f"Confluence ❌ (score={confluence.total_score:.1f} < {min_cf})"
        )

        # ---- 5️⃣ Session / Liquidity ----
        min_liq = self._param(60.0)
        self._check_lever(
            passed, failed,
            session.should_trade and session.liquidity_score >= min_liq,
            "Session ✅",
            f"Session ❌ (liq={session.liquidity_score:.1f}% < {min_liq})"
        )

        # ---- 6️⃣ Volatility entry timing ----
        min_vol = self._param(50.0)
        self._check_lever(
            passed, failed,
            volatility.should_enter_now and volatility.confidence >= min_vol,
            "Volatility ✅",
            f"Volatility ❌ (conf={volatility.confidence:.1f}% < {min_vol})"
        )

        # ---- 7️⃣ Overall score ----
        min_overall = self._param(70.0)
        self._check_lever(
            passed, failed,
            overall_score >= min_overall,
            "Overall score ✅",
            f"Overall score ❌ ({overall_score:.1f} < {min_overall})"
        )

        return passed, failed

    # =====================================================================
    # Risk‑adjustment multiplier
    # =====================================================================
    def _calc_risk_multiplier(
        self,
        direction: str,
        regime: RegimeResult,
        session: SessionResult,
    ) -> float:
        """Compute the final risk‑multiplier applied to base position size."""
        # Base multiplier (tunable via controller)
        base = self._param(1.0)

        # Regime‑based tweak – bearish regime + short‑sell → tighter risk
        regime_adj = (
            0.9
            if direction.upper() == "SELL"
            and getattr(regime, "regime", None) == "BEARISH"
            else 1.0
        )

        # Session‑based tweak – low‑liquidity sessions → tighter risk
        liq_threshold = self._param(60.0)
        session_adj = (
            0.95
            if getattr(session, "liquidity_score", 100.0) < liq_threshold
            else 1.0
        )

        # Combine the three factors
        raw_multiplier = base * regime_adj * session_adj

        # Clamp to safe envelope (0.5 – 2.0)
        multiplier = max(0.5, min(raw_multiplier, 2.0))

        logger.debug(
            "Risk multiplier calculated – base=%.3f, regime_adj=%.3f, "
            "session_adj=%.3f → raw=%.3f, clamped=%.3f",
            base,
            regime_adj,
            session_adj,
            raw_multiplier,
            multiplier,
        )
        return multiplier

    # =====================================================================
    # Helper stubs (implement in subclass or inject)
    # =====================================================================
    def _param(self, default: float) -> float:
        """
        ✅ FIXED: Removed unused "name" parameter.
        
        Get optimizer parameter with a default value.
        (In production, you would fetch from an injected controller,
         but since this is just a stub returning defaults, the parameter
         name was never actually used.)
        """
        return default

    def _build_recommendation(
        self,
        overall_score: float,
        overall_approved: bool,
        passed: List[str],
        failed: List[str],
        risk_multiplier: float,
    ) -> str:
        """Build human‑readable recommendation string."""
        lines = [
            f"Overall Score: {overall_score:.1f}/100",
            f"Approved: {overall_approved}",
            f"Risk Multiplier: {risk_multiplier:.2f}x",
            f"Passed Checks: {len(passed)}",
            f"Failed Checks: {len(failed)}",
        ]
        return "\n".join(lines)
