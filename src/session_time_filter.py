#!/usr/bin/env python3
"""
Session and Time‑Based Trading Filter

Lever #6 of the 7‑lever win‑rate optimisation system.

Optimises trading decisions based on market sessions, time‑of‑day,
day‑of‑week liquidity patterns and symbol‑specific suitability.

The module provides:

* `SessionTimeFilter` – the core engine.
* `session_filter` – a ready‑to‑use global instance.
* A tiny JSON‑backed persistence layer for the “avoid Asian / off‑hours”
  switches so you can change them without restarting the bot.

All public methods are fully typed, heavily logged and safe for production
use inside the Citadel Quantum Trader stack.
"""

# ----------------------------------------------------------------------
# Standard library
# ----------------------------------------------------------------------
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

# ----------------------------------------------------------------------
# Third‑party
# ----------------------------------------------------------------------
import MetaTrader5 as mt5  # MT5 Python API (already a dependency of CQT)

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Enums & data structures
# ----------------------------------------------------------------------
class TradingSession(Enum):
    """Canonical trading‑session identifiers (UTC based)."""
    ASIAN = "asian"
    LONDON = "london"
    NY = "new_york"
    LONDON_NY_OVERLAP = "london_ny_overlap"
    OFF_HOURS = "off_hours"


@dataclass
class SessionAnalysis:
    """Result of a session‑analysis call."""
    current_session: TradingSession
    liquidity_score: float                 # 0‑100 (higher = more liquid)
    volatility_expected: str               # LOW / NORMAL / HIGH
    recommended_symbols: List[str]         # Symbols that shine in this session
    should_trade: bool                     # Final go/no‑go flag
    reasoning: str                         # Human‑readable explanation
    risk_adjustment: float                 # 0.5‑1.5 multiplier applied to position size


# ----------------------------------------------------------------------
# Helper – tiny persistence for the two boolean switches
# ----------------------------------------------------------------------
class _ToggleStore:
    """
    Persists the ``avoid_asian`` and ``avoid_off_hours`` flags to a JSON file.
    The file lives under ``/app/config`` (mounted as a Docker volume) so the
    settings survive container restarts.
    """
    _PATH = Path("/app/config/session_filter_toggles.json")

    @classmethod
    def load(cls) -> Dict[str, bool]:
        if cls._PATH.is_file():
            try:
                with open(cls._PATH, "r") as f:
                    data = json.load(f)
                return {
                    "avoid_asian": bool(data.get("avoid_asian", True)),
                    "avoid_off_hours": bool(data.get("avoid_off_hours", True)),
                }
            except Exception as exc:   # pragma: no cover
                logger.error(f"Failed to read session toggles: {exc}")

        # defaults if file missing / unreadable
        return {"avoid_asian": True, "avoid_off_hours": True}

    @classmethod
    def save(cls, avoid_asian: bool, avoid_off_hours: bool) -> None:
        try:
            cls._PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(cls._PATH, "w") as f:
                json.dump(
                    {"avoid_asian": avoid_asian, "avoid_off_hours": avoid_off_hours},
                    f,
                    indent=2,
                )
        except Exception as exc:   # pragma: no cover
            logger.error(f"Failed to persist session toggles: {exc}")


# ----------------------------------------------------------------------
# Core engine
# ----------------------------------------------------------------------
class SessionTimeFilter:
    """
    Filters trades based on the current market session, time‑of‑day,
    day‑of‑week liquidity patterns and symbol‑specific suitability.

    The class is deliberately stateless apart from the two *avoid* flags,
    which are persisted to disk so they survive restarts.
    """

    # ------------------------------------------------------------------
    # Session definitions (UTC times)
    # ------------------------------------------------------------------
    SESSION_TIMES: Dict[TradingSession, Dict] = {
        TradingSession.ASIAN: {
            "start": time(0, 0),
            "end": time(7, 0),
            "description": "Tokyo session",
            "liquidity": 50,
            "best_pairs": ["USDJPY", "AUDJPY", "NZDJPY", "AUDUSD", "NZDUSD"],
        },
        TradingSession.LONDON: {
            "start": time(7, 0),
            "end": time(16, 0),
            "description": "London session",
            "liquidity": 85,
            "best_pairs": ["EURUSD", "GBPUSD", "EURGBP", "GBPJPY", "XAUUSD"],
        },
        TradingSession.LONDON_NY_OVERLAP: {
            "start": time(13, 0),
            "end": time(16, 0),
            "description": "London / New York overlap",
            "liquidity": 100,  # peak liquidity
            "best_pairs": ["EURUSD", "GBPUSD", "XAUUSD", "US30", "NAS100"],
        },
        TradingSession.NY: {
            "start": time(13, 0),
            "end": time(21, 0),
            "description": "New York session",
            "liquidity": 80,
            "best_pairs": ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "US30"],
        },
        TradingSession.OFF_HOURS: {
            "start": time(21, 0),
            "end": time(0, 0),
            "description": "Off‑hours (low liquidity)",
            "liquidity": 30,
            "best_pairs": [],  # avoid trading
        },
    }

    # ------------------------------------------------------------------
    # Time‑based risk adjustments (multiplicative)
    # ------------------------------------------------------------------
    RISK_ADJUSTMENTS: Dict[TradingSession, float] = {
        TradingSession.ASIAN: 0.7,            # thin markets → reduce risk
        TradingSession.LONDON: 1.0,
        TradingSession.LONDON_NY_OVERLAP: 1.2,  # high liquidity → can increase risk
        TradingSession.NY: 1.0,
        TradingSession.OFF_HOURS: 0.5,        # very low liquidity → shrink exposure
    }

    # ------------------------------------------------------------------
    def __init__(self, avoid_asian: Optional[bool] = None, avoid_off_hours: Optional[bool] = None):
        """
        Initialise the filter.

        Args:
            avoid_asian:   If ``True`` the Asian session is completely avoided.
                           If ``None`` the value is loaded from the persisted JSON.
            avoid_off_hours: Same idea for the off‑hours period.
        """
        persisted = _ToggleStore.load()
        self.avoid_asian = avoid_asian if avoid_asian is not None else persisted["avoid_asian"]
        self.avoid_off_hours = avoid_off_hours if avoid_off_hours is not None else persisted["avoid_off_hours"]
        logger.info(
            f"Session filter initialised – avoid_asian={self.avoid_asian}, "
            f"avoid_off_hours={self.avoid_off_hours}"
        )

    # ------------------------------------------------------------------
    # Public API ---------------------------------------------------------
    # ------------------------------------------------------------------
    def analyze_current_session(self, symbol: str) -> SessionAnalysis:
        """
        Analyse the *current* UTC session for a given ``symbol`` and return a
        fully populated :class:`SessionAnalysis` object.

        The method is pure – it does **not** modify any internal state
        (apart from reading the two toggle flags).
        """
        now_utc = datetime.utcnow().time()
        current_session = self._identify_session(now_utc)
        session_cfg = self.SESSION_TIMES[current_session]

        # ---- base liquidity (session‑defined) -------------------------
        base_liquidity = session_cfg["liquidity"]

        # ---- day‑of‑week adjustment ----------------------------------
        day_adj = self._day_of_week_adjustment()
        liquidity_score = base_liquidity * day_adj

        # ---- expected volatility --------------------------------------
        volatility_expected = self._estimate_session_volatility(current_session)

        # ---- symbol suitability ---------------------------------------
        recommended_symbols = session_cfg["best_pairs"]
        symbol_optimal = symbol in recommended_symbols if recommended_symbols else True

        # ---- final go/no‑go decision ---------------------------------
        should_trade = self._should_trade_in_session(
            current_session, symbol_optimal, liquidity_score
        )

        # ---- risk multiplier -----------------------------------------
        risk_adjustment = self.RISK_ADJUSTMENTS[current_session]

        # ---- human‑readable reasoning --------------------------------
        reasoning = self._generate_session_reasoning(
            current_session,
            symbol,
            symbol_optimal,
            liquidity_score,
            should_trade,
        )

        analysis = SessionAnalysis(
            current_session=current_session,
            liquidity_score=liquidity_score,
            volatility_expected=volatility_expected,
            recommended_symbols=recommended_symbols,
            should_trade=should_trade,
            reasoning=reasoning,
            risk_adjustment=risk_adjustment,
        )

        logger.info(
            f"[{symbol}] Session={current_session.value} | "
            f"Liquidity={liquidity_score:.0f} | Trade={should_trade}"
        )
        return analysis

    # ------------------------------------------------------------------
    def get_optimal_trading_hours(self, symbol: str) -> List[int]:
        """
        Return a list of integer hours (0‑23 UTC) that are *optimal* for the
        supplied ``symbol`` according to the session definitions.
        """
        optimal: List[int] = []
        for hour in range(24):
            test_time = time(hour, 0)
            sess = self._identify_session(test_time)
            if symbol in self.SESSION_TIMES[sess].get("best_pairs", []):
                optimal.append(hour)
        return optimal

    # ------------------------------------------------------------------
    def is_market_open(self) -> bool:
        """
        Very coarse market‑open check.

        * Weekends → closed (except Sunday 22:00 UTC onward when the Asian
          session starts).
        * Weekdays → considered open 24 h (most FX brokers operate 24 h).
        """
        now = datetime.utcnow()
        weekday = now.weekday()          # 0 = Mon … 6 = Sun

        # Saturday or Sunday → closed, except Sunday evening (NY start)
        if weekday >= 5:
            if weekday == 6 and now.hour >= 22:   # Sunday 22:00 UTC = Tokyo open
                return True
            return False

        # Weekday – market is open
        return True

    # ------------------------------------------------------------------
    # Runtime configuration ------------------------------------------------
    # ------------------------------------------------------------------
    def set_avoid_asian(self, avoid: bool) -> None:
        """Enable / disable the Asian‑session avoidance flag."""
        self.avoid_asian = avoid
        _ToggleStore.save(self.avoid_asian, self.avoid_off_hours)
        logger.info(f"Session filter – avoid_asian set to {avoid}")

    def set_avoid_off_hours(self, avoid: bool) -> None:
        """Enable / disable the off‑hours avoidance flag."""
        self.avoid_off_hours = avoid
        _ToggleStore.save(self.avoid_asian, self.avoid_off_hours)
        logger.info(f"Session filter – avoid_off_hours set to {avoid}")

    # ------------------------------------------------------------------
    # Private helpers ------------------------------------------------------
    # ------------------------------------------------------------------
    def _identify_session(self, current_time: time) -> TradingSession:
        """
        Resolve the current UTC ``time`` to a :class:`TradingSession`.

        Overlap (London/NY) is checked first because it is the most
        specific period.
        """
        # ----- London / NY overlap (13‑16 UTC) -------------------------
        overlap = self.SESSION_TIMES[TradingSession.LONDON_NY_OVERLAP]
        if overlap["start"] <= current_time < overlap["end"]:
            return TradingSession.LONDON_NY_OVERLAP

        # ----- Other sessions (including those that cross midnight) ----
        for sess, cfg in self.SESSION_TIMES.items():
            if sess == TradingSession.LONDON_NY_OVERLAP:
                continue  # already handled

            start, end = cfg["start"], cfg["end"]
            if start < end:                     # normal (no midnight crossing)
                if start <= current_time < end:
                    return sess
            else:                               # crosses midnight (e.g. OFF_HOURS)
                if current_time >= start or current_time < end:
                    return sess

        # Fallback – should never happen because OFF_HOURS covers the gap
        return TradingSession.OFF_HOURS

    # ------------------------------------------------------------------
    def _day_of_week_adjustment(self) -> float:
        """
        Return a multiplicative liquidity factor based on the current day
        of the week (UTC).  Values are empirically chosen for typical FX
        market behaviour.
        """
        dow = datetime.utcnow().weekday()   # 0 = Mon … 6 = Sun
        adjustments = {
            0: 1.00,   # Monday – normal
            1: 1.10,   # Tuesday – high activity
            2: 1.10,   # Wednesday – high activity
            3: 1.05,   # Thursday – good activity
            4: 0.90,   # Friday – early closures, lower liquidity
            5: 0.50,   # Saturday – almost no liquidity
            6: 0.50,   # Sunday – very low until 22:00 UTC
        }
        return adjustments.get(dow, 1.0)

    # ------------------------------------------------------------------
    def _estimate_session_volatility(self, session: TradingSession) -> str:
        """Map a session to an expected volatility tier."""
        mapping = {
            TradingSession.ASIAN: "LOW",
            TradingSession.LONDON: "NORMAL",
            TradingSession.LONDON_NY_OVERLAP: "HIGH",
            TradingSession.NY: "NORMAL",
            TradingSession.OFF_HOURS: "LOW",
        }
        return mapping.get(session, "NORMAL")

    # ------------------------------------------------------------------
    def _should_trade_in_session(
        self,
        session: TradingSession,
        symbol_optimal: bool,
        liquidity_score: float,
    ) -> bool:
        """
        Core go/no‑go logic.

        * Off‑hours and Asian‑session avoidance are honoured if configured.
        * If the symbol is *not* in the session’s “best_pairs” list we demand a
          higher liquidity threshold.
        * A hard minimum liquidity of 40 % is enforced for any trade.
        """
        # ---- explicit avoidance flags ---------------------------------
        if self.avoid_off_hours and session == TradingSession.OFF_HOURS:
            return False
        if self.avoid_asian and session == TradingSession.ASIAN:
            return False

        # ---- symbol‑specific liquidity requirement --------------------
        if not symbol_optimal and liquidity_score < 70:
            return False

        # ---- absolute floor -------------------------------------------
        if liquidity_score < 40:
            return False

        return True

    # ------------------------------------------------------------------
    def _generate_session_reasoning(
        self,
        session: TradingSession,
        symbol: str,
        symbol_optimal: bool,
        liquidity_score: float,
        should_trade: bool,
    ) -> str:
        """
        Produce a multi‑line, human‑readable explanation that can be logged
        or displayed in Grafana (via a Text panel).
        """
        cfg = self.SESSION_TIMES[session]
        lines = [
            f"🕒 Session : {cfg['description']} ({session.value})",
            f"💧 Liquidity score : {liquidity_score:.0f}/100",
        ]

        if symbol_optimal:
            lines.append(f"✅ Symbol {symbol} is optimal for this session.")
        else:
            # Show a few better candidates for reference
            top = cfg.get("best_pairs", [])[:3]
            suggestion = ", ".join(top) if top else "none"
            lines.append(
                f"⚠️ Symbol {symbol} NOT optimal – better pairs: {suggestion}"
            )

        if should_trade:
            lines.append("✅ Trade approved for this session.")
        else:
            if session == TradingSession.OFF_HOURS:
                lines.append("❌ Rejected – off‑hours (configured to avoid).")
            elif session == TradingSession.ASIAN and self.avoid_asian:
                lines.append("❌ Rejected – Asian session (configured to avoid).")
            elif liquidity_score < 40:
                lines.append("❌ Rejected – liquidity too low (< 40 %).")
            else:
                lines.append("❌ Rejected – session criteria not satisfied.")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Utility – nice string representation for debugging
    # ------------------------------------------------------------------
    def __repr__(self) -> str:   # pragma: no cover
        return (
            f"<SessionTimeFilter avoid_asian={self.avoid_asian} "
            f"avoid_off_hours={self.avoid_off_hours}>"
        )


# ----------------------------------------------------------------------
# Global singleton – import this from ``src/main.py`` or any other module
# ----------------------------------------------------------------------
session_filter = SessionTimeFilter()
