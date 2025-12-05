# src/optimizer/master_optimizer.py
"""
MasterWinRateOptimizer – the high‑level orchestration layer for CQT’s
risk‑adjusted win‑rate evaluation.

It glues together the individual “lever” modules (entry‑quality,
market‑regime, multi‑timeframe confirmation, confluence, session,
volatility) and produces a single, human‑readable recommendation plus
a numeric risk‑multiplier that downstream components (order engine,
position sizing) consume.

All heavy lifting lives in the sub‑modules; this class is intentionally
thin, testable, and side‑effect free (apart from logging).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple, Protocol, runtime_checkable

# ----------------------------------------------------------------------
# Local imports – kept inside the file to avoid circular imports at import‑time
# ----------------------------------------------------------------------
from src.utils.common import utc_now
from src.optimizer.leverage_results import (
    EntryQualityResult,
    RegimeResult,
    MtfResult,
    ConfluenceResult,
    SessionResult,
    VolatilityResult,
)

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Typed protocol for the external optimizer controller (holds tunable params)
# ----------------------------------------------------------------------
@runtime_checkable
class OptimizerControllerProto(Protocol):
    """Minimal interface the optimizer expects from the controller."""

    def get(self, name: str) -> float: ...


# ----------------------------------------------------------------------
# Public dataclass – the single return value of the optimizer
# ----------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class OverallResult:
    """
    Aggregated outcome of a full win‑rate evaluation.

    Attributes
    ----------
    overall_score: float
        Weighted composite score (0‑100). Higher is better.
    overall_approved: bool
        Whether the trade passes the global acceptance rule.
    risk_multiplier: float
        Multiplier applied to the base position size (≥ 0.5, ≤ 2.0 by default).
    recommendation: str
        Human‑readable multi‑line explanation (suitable for logging or UI).
    passed: List[str]
        Names of levers that satisfied their thresholds.
    failed: List[str]
        Names of levers that fell short.
    """

    overall_score: float
    overall_approved: bool
    risk_multiplier: float
    recommendation: str
    passed: List[str]
    failed: List[str]


# ----------------------------------------------------------------------
# Master optimizer – thin façade delegating to the lever objects
# ----------------------------------------------------------------------
class MasterWinRateOptimizer:
    """
    Orchestrates the full win‑rate pipeline.

    The constructor lazily imports the lever implementations to avoid
    circular imports and to keep start‑up time low.  All levers are stored
    as attributes so they can be mocked easily in unit tests.
    """

    # ------------------------------------------------------------------
    # Construction – lazy imports
    # ------------------------------------------------------------------
    def __init__(self) -> None:
        # Local imports – they pull the heavy modules only when the
        # optimizer is instantiated (e.g. during a trade‑evaluation tick).
        from advanced_entry_filter import entry_quality_filter
        from advanced_exit_system import dynamic_exit_manager
        from market_regime_detector import market_regime_detector
        from multi_timeframe_confirmation import mtf_confirmation
        from advanced_confluence_system import confluence_system
        from session_time_filter import session_filter
        from volatility_entry_refinement import volatility_entry

        # Assign to instance attributes (makes them easy to patch in tests)
        self.entry_quality = entry_quality_filter
        self.exit_manager = dynamic_exit_manager
        self.regime_detector = market_regime_detector
        self.mtf = mtf_confirmation
        self.confluence = confluence_system
        self.session = session_filter
        self.volatility = volatility_entry

        logger.info("🔧 MasterWinRateOptimizer instantiated")

    # ------------------------------------------------------------------
    # Public entry point – thin wrapper around the private pipeline
    # ------------------------------------------------------------------
    def comprehensive_trade_analysis(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
    ) -> OverallResult:
        """
        Run the full lever chain for a single trade and return a rich result.

        Parameters
        ----------
        symbol : str
            Ticker (e.g. ``EURUSD``).
        direction : str
            ``"BUY"`` or ``"SELL"``.
        entry_price : float
            Proposed entry price (in the instrument’s price units).
        stop_loss : float
            Desired stop‑loss price.

        Returns
        -------
        OverallResult
            Dataclass containing the numeric score, approval flag,
            risk multiplier, a printable recommendation and the list of
            passed / failed levers.
        """
        logger.info(
            f"▶️ Analyzing {symbol} {direction} @ {entry_price:.5f} (SL={stop_loss:.5f})"
        )

        # ------------------------------------------------------------------
        # 1️⃣  Run each lever – each returns a small, typed result object
        # ------------------------------------------------------------------
        entry = self._run_entry_quality(symbol, direction, entry_price, stop_loss)
        regime = self._run_market_regime(symbol, direction)
        mtf = self._run_mtf(symbol, direction, entry_price)
        confluence = self._run_confluence(
            symbol, direction, entry_price, stop_loss, mtf
        )
        session = self._run_session(symbol)
        volatility = self._run_volatility(symbol, direction, entry_price)

        # ------------------------------------------------------------------
        # 2️⃣  Compute weighted overall score (0‑100)
        # ------------------------------------------------------------------
        overall_score = self._compute_overall_score(
            entry, regime, mtf, confluence, session, volatility
        )

        # ------------------------------------------------------------------
        # 3️⃣  Evaluate each lever against its tunable thresholds
        # ------------------------------------------------------------------
        passed, failed = self._evaluate_checks(
            entry,
            regime,
            mtf,
            confluence,
            session,
            volatility,
            overall_score,
        )

        # ------------------------------------------------------------------
        # 4️⃣  Overall approval & risk multiplier
        # ------------------------------------------------------------------
        overall_approved = (
            overall_score >= self._param("min_overall_score", 70.0)
        ) and (len(failed) <= 1)

        # NOTE: the original code passed only two args – SonarQube flagged it.
        # We now accept the *session* result as the third argument.
        risk_multiplier = self._calc_risk_multiplier(direction, regime, session)

        # ------------------------------------------------------------------
        # 5️⃣  Human‑readable recommendation string
        # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Private helpers – each stays < 30 LOC and has a cyclomatic complexity ≤ 4
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------
    def _compute_overall_score(
        self,
        entry: EntryQualityResult,
        regime: RegimeResult,
        mtf: MtfResult,
        confluence: ConfluenceResult,
        session: SessionResult,
        volatility: VolatilityResult,
    ) -> float:
        """
        Linear weighted combination of the six lever scores.

        All sub‑scores are normalised to the 0‑100 range (or 0‑1 for
        booleans).  The weight values are tunable via the optimizer
        controller (see ``_param``).
        """
        w = {
            "entry": self._param("weight_entry_quality", 0.25),
            "regime": self._param("weight_regime", 0.15),
            "mtf": self._param("weight_mtf", 0.20),
            "conf": self._param("weight_confluence", 0.25),
            "session": self._param("weight_session", 0.10),
            "vol": self._param("weight_volatility", 0.05),
        }

        # Simple linear combination – each sub‑score is already 0‑100 (or 0‑1 for booleans)
        overall = (
            entry.total_score * w["entry"]
            + (100 if regime.aligned else 30) * w["regime"]
            + mtf.alignment_score * w["mtf"]
            + confluence.total_score * w["conf"]
            + session.liquidity_score * w["session"]
            + volatility.confidence * w["vol"]
        )
        return overall

    # ------------------------------------------------------------------
    # Lever‑threshold evaluation
    # ------------------------------------------------------------------
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

        # ---- 1️⃣ Entry quality ------------------------------------------------
        min_eq = self._param("min_entry_quality", 75.0)
        if entry.total_score >= min_eq:
            passed.append("Entry quality ✅")
        else:
            failed.append(
                f"Entry quality ❌ ({entry.total_score:.1f} < {min_eq})"
            )

        # ---- 2️⃣ Regime alignment --------------------------------------------
        if regime.aligned:
            passed.append("Regime aligned ✅")
        else:
            failed.append(f"Regime mis‑aligned ❌ ({regime.reason})")

        # ---- 3️⃣ MTF -----------------------------------------------------------
        min_mtf = self._param("min_mtf_alignment", 60.0)
        if mtf.approved and mtf.alignment_score >= min_mtf:
            passed.append("MTF ✅")
        else:
            failed.append(
                f"MTF ❌ (align={mtf.alignment_score:.1f}% < {min_mtf})"
            )

        # ---- 4️⃣ Confluence ----------------------------------------------------
        min_cf = self._param("min_confluence_score", 70.0)
        if confluence.should_trade and confluence.total_score >= min_cf:
            passed.append("Confluence ✅")
        else:
            failed.append(
                f"Confluence ❌ (score={confluence.total_score:.1f} < {min_cf})"
            )

        # ---- 5️⃣ Session / Liquidity -------------------------------------------
        min_liq = self._param("min_session_liquidity", 60.0)
        if session.should_trade and session.liquidity_score >= min_liq:
            passed.append("Session ✅")
        else:
            failed.append(
                f"Session ❌ (liq={session.liquidity_score:.1f}% < {min_liq})"
            )

        # ---- 6️⃣ Volatility entry timing ---------------------------------------
        min_vol = self._param("min_volatility_confidence", 50.0)
        if volatility.should_enter_now and volatility.confidence >= min_vol:
            passed.append("Volatility ✅")
        else:
            failed.append(
                f"Volatility ❌ (conf={volatility.confidence:.1f}% < {min_vol})"
            )

        # ---- 7️⃣ Overall score -------------------------------------------------
        min_overall = self._param("min_overall_score", 70.0)
        if overall_score >= min_overall:
            passed.append("Overall score ✅")
        else:
            failed.append(
                f"Overall score ❌ ({overall_score:.1f} < {min_overall})"
            )

        return passed, failed

    # ------------------------------------------------------------------
    # Risk‑adjustment multiplier (global + optional regime / session tweaks)
    # ------------------------------------------------------------------
     def _calc_risk_multiplier(
        self,
        direction: str,
        regime: RegimeResult,
        session: SessionResult,
    ) -> float:
        """
        Compute the final risk‑multiplier that will be applied to the
        base position size.

        The multiplier is a product of three independent factors:

        1️⃣ **Base multiplier** – a global knob that can be tuned from the
           optimizer controller (default = 1.0).

        2️⃣ **Regime adjustment** – when we are short‑selling (`SELL`)
           in a *bearish* macro regime we tighten risk a little
           (multiply by 0.9).  All other combos keep the factor at 1.0.

        3️⃣ **Session adjustment** – low‑liquidity sessions are riskier.
           If the session’s liquidity score falls below the configurable
           threshold we penalise the multiplier (multiply by 0.95);
           otherwise the factor stays at 1.0.

        Finally the product is **clamped** to a sensible band
        (0.5 ≤ multiplier ≤ 2.0) so that extreme parameter values
        cannot blow up the position size.

        Returns
        -------
        float
            The risk multiplier (≥ 0.5, ≤ 2.0).
        """
        # ------------------------------------------------------------------
        # 1️⃣  Global base multiplier (tunable via the controller)
        # ------------------------------------------------------------------
        base = self._param("base_risk_multiplier", 1.0)

        # ------------------------------------------------------------------
        # 2️⃣  Regime‑based tweak – bearish regime + short‑sell → tighter risk
        # ------------------------------------------------------------------
        regime_adj = (
            0.9
            if direction.upper() == "SELL"
            and getattr(regime, "regime", None) == "BEARISH"
            else 1.0
        )

        # ------------------------------------------------------------------
        # 3️⃣  Session‑based tweak – low‑liquidity sessions → tighter risk
        # ------------------------------------------------------------------
        # ``session`` is a ``SessionResult``; it carries a ``liquidity_score``
        # in the 0‑100 range.  The threshold is configurable.
        liq_threshold = self._param("session_liquidity_threshold", 60.0)
        session_adj = (
            0.95
            if getattr(session, "liquidity_score", 100.0) < liq_threshold
            else 1.0
        )

        # ------------------------------------------------------------------
        # 4️⃣  Combine the three factors
        # ------------------------------------------------------------------
        raw_multiplier = base * regime_adj * session_adj

        # ------------------------------------------------------------------
        # 5️⃣  Clamp to a safe envelope (0.5 – 2.0)
        # ------------------------------------------------------------------
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
