# src/optimizer/master_optimizer.py
import logging
from dataclasses import dataclass
from typing import List, Tuple, Dict

from src.utils.common import utc_now
from src.optimizer.leverage_results import (
    EntryQualityResult,
    RegimeResult,
    MtfResult,
    ConfluenceResult,
    SessionResult,
    VolatilityResult,
)

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Helper dataclasses – one per lever (keeps the big return tidy)
# ----------------------------------------------------------------------
@dataclass
class OverallResult:
    overall_score: float
    overall_approved: bool
    risk_multiplier: float
    recommendation: str
    passed: List[str]
    failed: List[str]


# ----------------------------------------------------------------------
# Master Optimizer – thin façade that delegates to private helpers
# ----------------------------------------------------------------------
class MasterWinRateOptimizer:
    def __init__(self) -> None:
        # Lazy imports avoid circular deps and speed up start‑up
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
        logger.info(f"▶️ Analyzing {symbol} {direction} @ {entry_price}")

        # 1️⃣  Run each lever – each returns a small, typed result
        entry = self._run_entry_quality(symbol, direction, entry_price, stop_loss)
        regime = self._run_market_regime(symbol, direction)
        mtf = self._run_mtf(symbol, direction, entry_price)
        confluence = self._run_confluence(symbol, direction, entry_price, stop_loss, mtf)
        session = self._run_session(symbol)
        volatility = self._run_volatility(symbol, direction, entry_price)

        # 2️⃣  Compute weighted overall score
        overall_score = self._compute_overall_score(
            entry, regime, mtf, confluence, session, volatility
        )

        # 3️⃣  Evaluate each lever against its tunable thresholds
        passed, failed = self._evaluate_checks(
            entry, regime, mtf, confluence, session, volatility, overall_score
        )

        # 4️⃣  Overall approval & risk multiplier
        overall_approved = (overall_score >= self._param("min_overall_score", 70.0)) and (
            len(failed) <= 1
        )
        risk_multiplier = self._calc_risk_multiplier(direction, regime, session)

        # 5️⃣  Human‑readable recommendation
        recommendation = self._build_recommendation(
            overall_score, overall_approved, passed, failed, risk_multiplier
        )

        logger.info(recommendation)
        return OverallResult(
            overall_score=overall_score,
            overall_approved=overall_approved,
            risk_multiplier=risk_multiplier,
            recommendation=recommendation,
            passed=passed,
            failed=failed,
        )

    # ------------------------------------------------------------------
    # Private helpers – each is < 30 LOC, complexity ≤ 4
    # ------------------------------------------------------------------
    def _run_entry_quality(
        self, symbol: str, direction: str, entry_price: float, stop_loss: float
    ) -> EntryQualityResult:
        logger.debug("🔍 Entry‑quality")
        return self.entry_quality.assess_entry_quality(
            symbol, direction, entry_price, stop_loss
        )

    def _run_market_regime(self, symbol: str, direction: str) -> RegimeResult:
        logger.debug("🌐 Market‑regime")
        regime = self.regime_detector.analyze_market_regime(symbol)
        aligned, reason = self.regime_detector.should_trade_in_current_regime(
            direction
        )
        return RegimeResult(regime, aligned, reason)

    def _run_mtf(self, symbol: str, direction: str, entry_price: float) -> MtfResult:
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
        logger.debug("🎲 Confluence")
        return self.confluence.analyze_confluence(
            symbol,
            direction,
            entry_price,
            stop_loss,
            mtf_data={"alignment_score": mtf.alignment_score},
        )

    def _run_session(self, symbol: str) -> SessionResult:
        logger.debug("🕐 Session")
        return self.session.analyze_current_session(symbol)

    def _run_volatility(
        self, symbol: str, direction: str, entry_price: float
    ) -> VolatilityResult:
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
    # Evaluate every lever against its tunable thresholds
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
    def _calc_risk_multiplier(self, direction: str, regime: RegimeResult) -> float:
        base = self._param("base_risk_multiplier", 1.0)

        # Example tweak: tighten risk a bit on bearish regimes when short‑selling
        regime_adj = (
            0.9
            if direction == "SELL" and regime.regime.value == "BEARISH"
            else 1.0
        )
        # Example tweak: penalise low‑liquidity sessions
        session_adj = 0.95 if regime.aligned is False else 1.0

        return base * regime_adj * session_adj

    # ------------------------------------------------------------------
    # Human‑readable recommendation string
    # ------------------------------------------------------------------
    def _build_recommendation(
        self,
        overall_score: float,
        overall_approved: bool,
        passed: List[str],
        failed: List[str],
        risk_multiplier: float,
    ) -> str:
        lines = ["=" * 80]

        if overall_approved:
            lines.append("✅ TRADE APPROVED")
            lines.append(f"Overall score: {overall_score:.1f}/100")
            lines.append(f"Passed checks ({len(passed)}/6):")
            lines.extend([f"  • {c}" for c in passed])
            if failed:
                lines.append("Minor concerns:")
                lines.extend([f"  • {c}" for c in failed])
            lines.append(f"Risk multiplier: {risk_multiplier:.2f}×")
        else:
            lines.append("❌ TRADE REJECTED")
            lines.append(f"Overall score: {overall_score:.1f}/100")
            lines.append(f"Failed checks ({len(failed)}):")
            lines.extend([f"  • {c}" for c in failed])
            lines.append("Await better market conditions.")

        lines.append("=" * 80)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helper – fetch a tunable parameter from the controller (with fallback)
    # ------------------------------------------------------------------
    def _param(self, name: str, fallback: float) -> float:
        try:
            return optimizer_controller.get(name)
        except Exception:  # pragma: no cover
            return fallback
