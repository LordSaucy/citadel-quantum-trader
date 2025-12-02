#!/usr/bin/env python3
"""
DRAWNDOWN TRACKER – ADVANCED REAL‑TIME MONITORING

Features
--------
* Real‑time draw‑down calculation (peak → valley → recovery)
* Daily / weekly / monthly draw‑down tracking
* Peak‑to‑valley analysis & recovery metrics
* Alert thresholds (warning / error / critical)
* Historical persistence (JSON on a mounted volume)
* Prometheus metrics for Grafana dashboards
* Simple export / reporting utilities
"""

# ----------------------------------------------------------------------
# Standard library
# ----------------------------------------------------------------------
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ----------------------------------------------------------------------
# Third‑party
# ----------------------------------------------------------------------
from prometheus_client import Gauge   # pip install prometheus_client

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------
@dataclass
class DrawdownSnapshot:
    """One balance/equity snapshot."""
    timestamp: datetime
    balance: float
    equity: float
    drawdown_amount: float
    drawdown_percent: float
    peak_balance: float


@dataclass
class DrawdownPeriod:
    """A complete draw‑down episode (peak → valley → optional recovery)."""
    start_time: datetime
    end_time: Optional[datetime] = None
    peak_balance: float = 0.0
    valley_balance: float = 0.0
    max_drawdown_pct: float = 0.0
    recovery_time: Optional[timedelta] = None
    is_recovered: bool = False


# ----------------------------------------------------------------------
# Main tracker class
# ----------------------------------------------------------------------
class EnhancedDrawdownTracker:
    """
    Advanced draw‑down tracking system.
    All thresholds / limits are **tunable at runtime** via the
    `set_threshold()` method (or by editing the JSON file on disk).
    """

    # ------------------------------------------------------------------
    # Default thresholds – can be overridden at runtime
    # ------------------------------------------------------------------
    DEFAULTS = {
        # Alert levels (percent)
        "warning_threshold": 0.03,   # 3 %
        "alert_threshold": 0.05,     # 5 %
        "critical_threshold": 0.08,  # 8 %

        # Period limits
        "max_daily_drawdown": 0.05,   # 5 %
        "max_weekly_drawdown": 0.10,  # 10 %
        "max_monthly_drawdown": 0.15, # 15 %
    }

    # ------------------------------------------------------------------
    # Prometheus gauges (registered once per process)
    # ------------------------------------------------------------------
    _g_current = Gauge(
        "drawdown_current_pct",
        "Current draw‑down as a fraction of the peak balance",
        ["account"]
    )
    _g_max = Gauge(
        "drawdown_max_pct",
        "Maximum historic draw‑down as a fraction of the peak balance",
        ["account"]
    )
    _g_daily = Gauge(
        "drawdown_daily_pct",
        "Current daily draw‑down (relative to start‑of‑day balance)",
        ["account"]
    )
    _g_weekly = Gauge(
        "drawdown_weekly_pct",
        "Current weekly draw‑down (relative to start‑of‑week balance)",
        ["account"]
    )
    _g_monthly = Gauge(
        "drawdown_monthly_pct",
        "Current monthly draw‑down (relative to start‑of‑month balance)",
        ["account"]
    )

    # ------------------------------------------------------------------
    def __init__(self, initial_balance: float = 10_000.0,
                 account_name: str = "default"):
        """
        Initialise the tracker.

        Args
        ----
        initial_balance: Starting account balance (used for the first peak).
        account_name: Identifier used for Prometheus labels.
        """
        self.account_name = account_name

        # ------------------------------------------------------------------
        # Core state
        # ------------------------------------------------------------------
        self.initial_balance = float(initial_balance)
        self.peak_balance = self.initial_balance

        self.snapshots: List[DrawdownSnapshot] = []
        self.periods: List[DrawdownPeriod] = []

        self.current_drawdown_pct = 0.0
        self.max_drawdown_pct = 0.0
        self.max_drawdown_amount = 0.0

        # ------------------------------------------------------------------
        # Periodic start balances (reset on day/week/month roll‑over)
        # ------------------------------------------------------------------
        self.daily_start_balance = self.initial_balance
        self.weekly_start_balance = self.initial_balance
        self.monthly_start_balance = self.initial_balance

        self.daily_reset_date = datetime.now().date()
        self.weekly_reset_date = datetime.now().date()
        self.monthly_reset_date = datetime.now().date()

        # ------------------------------------------------------------------
        # Draw‑down episode tracking
        # ------------------------------------------------------------------
        self.in_drawdown = False
        self.current_period: Optional[DrawdownPeriod] = None

        # ------------------------------------------------------------------
        # Runtime‑modifiable thresholds (start with defaults)
        # ------------------------------------------------------------------
        self.thresholds: Dict[str, float] = self.DEFAULTS.copy()

        # ------------------------------------------------------------------
        # Persistence
        # ------------------------------------------------------------------
        self._persist_path = Path("/app/config/drawdown_tracker_state.json")
        self._load_state()

        # ------------------------------------------------------------------
        # Initialise Prometheus gauges
        # ------------------------------------------------------------------
        self._publish_metrics()

        logger.info(
            f"📊 DrawdownTracker initialised – balance ${self.initial_balance:,.2f}"
        )

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _load_state(self) -> None:
        """Load historic periods & thresholds from JSON (if present)."""
        if not self._persist_path.exists():
            logger.debug("No persisted drawdown state – starting fresh")
            return

        try:
            with open(self._persist_path, "r") as f:
                data = json.load(f)

            # Restore thresholds
            self.thresholds.update(data.get("thresholds", {}))

            # Restore periods
            for p in data.get("periods", []):
                period = DrawdownPeriod(
                    start_time=datetime.fromisoformat(p["start_time"]),
                    end_time=datetime.fromisoformat(p["end_time"])
                    if p.get("end_time")
                    else None,
                    peak_balance=p["peak_balance"],
                    valley_balance=p["valley_balance"],
                    max_drawdown_pct=p["max_drawdown_pct"],
                    recovery_time=timedelta(seconds=p["recovery_seconds"])
                    if p.get("recovery_seconds")
                    else None,
                    is_recovered=p.get("is_recovered", False),
                )
                self.periods.append(period)

            logger.info(
                f"🗄️ Restored {len(self.periods)} historic drawdown periods"
            )
        except Exception as exc:  # pragma: no cover
            logger.error(f"Failed to load drawdown state: {exc}")

    def _persist_state(self) -> None:
        """Write thresholds & completed periods to JSON."""
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "thresholds": self.thresholds,
                "periods": [
                    {
                        "start_time": p.start_time.isoformat(),
                        "end_time": p.end_time.isoformat() if p.end_time else None,
                        "peak_balance": p.peak_balance,
                        "valley_balance": p.valley_balance,
                        "max_drawdown_pct": p.max_drawdown_pct,
                        "recovery_seconds": p.recovery_time.total_seconds()
                        if p.recovery_time
                        else None,
                        "is_recovered": p.is_recovered,
                    }
                    for p in self.periods
                ],
            }
            with open(self._persist_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as exc:  # pragma: no cover
            logger.error(f"Could not persist drawdown state: {exc}")

    # ------------------------------------------------------------------
    # Public API – threshold manipulation
    # ------------------------------------------------------------------
    def set_threshold(self, name: str, value: float) -> None:
        """
        Change a runtime threshold.

        Example
        -------
        >>> tracker.set_threshold("warning_threshold", 0.025)   # 2.5 %
        """
        if name not in self.thresholds:
            raise KeyError(f"Unknown threshold: {name}")

        self.thresholds[name] = float(value)
        self._persist_state()
        logger.info(f"Threshold `{name}` set to {value}")

    def get_threshold(self, name: str) -> float:
        """Read a threshold (raises KeyError if unknown)."""
        return self.thresholds[name]

    # ------------------------------------------------------------------
    # Core snapshot recording
    # ------------------------------------------------------------------
    def record_snapshot(self, balance: float, equity: float) -> None:
        """
        Record the current account state.

        Args
        ----
        balance: Current account balance.
        equity:  Current account equity (including unrealised P/L).
        """
        now = datetime.now()

        # --------------------------------------------------------------
        # 1️⃣  Update peak balance
        # --------------------------------------------------------------
        if balance > self.peak_balance:
            self.peak_balance = balance
            logger.info(f"🏔️ New peak balance: ${balance:,.2f}")

            # If we were in a draw‑down, this is a recovery
            if self.in_drawdown and self.current_period:
                self.current_period.end_time = now
                self.current_period.is_recovered = True
                self.current_period.recovery_time = now - self.current_period.start_time
                self.periods.append(self.current_period)

                logger.info(
                    f"✅ Draw‑down recovered after "
                    f"{self.current_period.recovery_time}"
                )
                self.in_drawdown = False
                self.current_period = None

        # --------------------------------------------------------------
        # 2️⃣  Compute current draw‑down
        # --------------------------------------------------------------
        dd_amount = self.peak_balance - equity
        dd_pct = dd_amount / self.peak_balance if self.peak_balance else 0.0

        # Update max draw‑down ever seen
        if dd_pct > self.max_drawdown_pct:
            self.max_drawdown_pct = dd_pct
            self.max_drawdown_amount = dd_amount

        self.current_drawdown_pct = dd_pct

        # --------------------------------------------------------------
        # 3️⃣  Begin a new draw‑down period if we just entered one
        # --------------------------------------------------------------
        if dd_pct > 0.0 and not self.in_drawdown:
            self.in_drawdown = True
            self.current_period = DrawdownPeriod(
                start_time=now,
                peak_balance=self.peak_balance,
                valley_balance=equity,
                max_drawdown_pct=dd_pct,
            )
            logger.warning(
                f"⚠️ Entered draw‑down: {dd_pct * 100:.2f}% "
                f"(peak ${self.peak_balance:,.2f})"
            )

        # --------------------------------------------------------------
        # 4️⃣  Update ongoing draw‑down period
        # --------------------------------------------------------------
        if self.in_drawdown and self.current_period:
            if equity < self.current_period.valley_balance:
                self.current_period.valley_balance = equity
            if dd_pct > self.current_period.max_drawdown_pct:
                self.current_period.max_drawdown_pct = dd_pct

        # --------------------------------------------------------------
        # 5️⃣  Store snapshot (keep last 1000 only)
        # --------------------------------------------------------------
        snap = DrawdownSnapshot(
            timestamp=now,
            balance=balance,
            equity=equity,
            drawdown_amount=dd_amount,
            drawdown_percent=dd_pct,
            peak_balance=self.peak_balance,
        )
        self.snapshots.append(snap)
        if len(self.snapshots) > 1000:
            self.snapshots.pop(0)

        # --------------------------------------------------------------
        # 6️⃣  Publish Prometheus metrics
        # --------------------------------------------------------------
        self._publish_metrics()

        # --------------------------------------------------------------
        # 7️⃣  Evaluate thresholds / alerts
        # --------------------------------------------------------------
        self._evaluate_alerts(dd_pct)

        # --------------------------------------------------------------
        # 8️⃣  Reset daily/weekly/monthly trackers if needed
        # --------------------------------------------------------------
        self._reset_periods_if_needed(balance)

    # ------------------------------------------------------------------
    # Prometheus publishing
    # ------------------------------------------------------------------
    def _publish_metrics(self) -> None:
        """Push the latest draw‑down numbers to Prometheus."""
        self._g_current.labels(account=self.account_name).set(self.current_drawdown_pct)
        self._g_max.labels(account=self.account_name).set(self.max_drawdown_pct)
        self._g_daily.labels(account=self.account_name).set(self.get_daily_drawdown())
        self._g_weekly.labels(account=self.account_name).set(self.get_weekly_drawdown())
        self._g_monthly.labels(account=self.account_name).set(self.get_monthly_drawdown())

    # ------------------------------------------------------------------
    # Periodic reset handling
    # ------------------------------------------------------------------
    def _reset_periods_if_needed(self, balance: float) -> None:
        """Roll‑over daily / weekly / monthly start balances."""
        now = datetime.now()

        # ----- Daily -----
        if now.date() != self.daily_reset_date:
            self.daily_start_balance = balance
            self.daily_reset_date = now.date()
            logger.debug("Daily draw‑down baseline reset")

        # ----- Weekly (Monday) -----
        if now.weekday() == 0 and now.date() != self.weekly_reset_date:
            self.weekly_start_balance = balance
            self.weekly_reset_date = now.date()
            logger.debug("Weekly draw‑down baseline reset")

        # ----- Monthly (first of month) -----
        if now.day == 1 and now.date() != self.monthly_reset_date:
            self.monthly_start_balance = balance
            self.monthly_reset_date = now.date()
            logger.debug("Monthly draw‑down baseline reset")

    # ------------------------------------------------------------------
    # Alert evaluation
    # ------------------------------------------------------------------
    def _evaluate_alerts(self, dd_pct: float) -> None:
        """Log warnings / errors / critical messages based on thresholds."""
        if dd_pct >= self.thresholds["critical_threshold"]:
            logger.critical(f"🚨 CRITICAL DRAW‑DOWN: {dd_pct * 100:.2f}%")
        elif dd_pct >= self.thresholds["alert_threshold"]:
            logger.error(f"❌ ALERT DRAW‑DOWN: {dd_pct * 100:.2f}%")
        elif dd_pct >= self.thresholds["warning_threshold"]:
            logger.warning(f"⚠️ WARNING DRAW‑DOWN: {dd_pct * 100:.2f}%")

    # ------------------------------------------------------------------
    # Daily / weekly / monthly draw‑down getters (fraction)
    # ------------------------------------------------------------------
    def get_daily_drawdown(self) -> float:
        """Draw‑down relative to the start‑of‑day balance."""
        if not self.snapshots:
            return 0.0
        cur_eq = self.snapshots[-1].equity
        return (self.daily_start_balance - cur_eq) / self.daily_start_balance

    def get_weekly_drawdown(self) -> float:
        """Draw‑down relative to the start‑of‑week balance."""
        if not self.snapshots:
            return 0.0
        cur_eq = self.snapshots[-1].equity
        return (self.weekly_start_balance - cur_eq) / self.weekly_start_balance

    def get_monthly_drawdown(self) -> float:
        """Draw‑down relative to the start‑of‑month balance."""
        if not self.snapshots:
            return 0.0
        cur_eq = self.snapshots[-1].equity
        return (self.monthly_start_balance - cur_eq) / self.monthly_start_balance

    # ------------------------------------------------------------------
    # Limit checking (returns bool + human‑readable reason)
    # ------------------------------------------------------------------
    def is_limit_hit(self) -> Tuple[bool, str]:
        """
        Verify whether any of the daily / weekly / monthly limits have been
        breached.

        Returns
        -------
        (hit, reason) – ``hit`` is True if a limit is exceeded.
        """
        daily = self.get_daily_drawdown()
        if daily >= self.thresholds["max_daily_drawdown"]:
            return True, f"Daily limit hit: {daily * 100:.2f}%"

        weekly = self.get_weekly_drawdown()
        if weekly >= self.thresholds["max_weekly_drawdown"]:
            return True, f"Weekly limit hit: {weekly * 100:.2f}%"

        monthly = self.get_monthly_drawdown()
        if monthly >= self.thresholds["max_monthly_drawdown"]:
            return True, f"Monthly limit hit: {monthly * 100:.2f}%"

        return False, ""

    # ------------------------------------------------------------------
    # Statistics & reporting
    # ------------------------------------------------------------------
    def get_drawdown_stats(self) -> Dict:
        """Return a snapshot of the current draw‑down situation."""
        if not self.snapshots:
            return self._empty_stats()

        cur = self.snapshots[-1]
        return {
            "current_balance": cur.balance,
            "current_equity": cur.equity,
            "peak_balance": self.

       # ------------------------------------------------------------------
        # 2️⃣  Recovery statistics (average, fastest, longest)
        # ------------------------------------------------------------------
        def get_recovery_stats(self) -> Dict:
            """Return aggregated recovery metrics from completed draw‑down periods."""
            if not self.periods:
                return {
                    "recovery_count": 0,
                    "avg_recovery_seconds": None,
                    "fastest_recovery_seconds": None,
                    "slowest_recovery_seconds": None,
                }

            recovered = [p for p in self.periods if p.is_recovered and p.recovery_time]
            if not recovered:
                return {
                    "recovery_count": 0,
                    "avg_recovery_seconds": None,
                    "fastest_recovery_seconds": None,
                    "slowest_recovery_seconds": None,
                }

            secs = [p.recovery_time.total_seconds() for p in recovered]
            return {
                "recovery_count": len(recovered),
                "avg_recovery_seconds": sum(secs) / len(secs),
                "fastest_recovery_seconds": min(secs),
                "slowest_recovery_seconds": max(secs),
            }

        # ------------------------------------------------------------------
        # 3️⃣  Empty‑state helper (used when no snapshots exist)
        # ------------------------------------------------------------------
        def _empty_stats(self) -> Dict:
            """Return a minimal stats dict when no data is available."""
            return {
                "current_balance": 0.0,
                "current_equity": 0.0,
                "peak_balance": self.peak_balance,
                "current_drawdown_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "daily_drawdown_pct": 0.0,
                "weekly_drawdown_pct": 0.0,
                "monthly_drawdown_pct": 0.0,
                "in_drawdown": False,
                "total_periods": len(self.periods),
                "snapshots_recorded": len(self.snapshots),
            }

        # ------------------------------------------------------------------
        # 4️⃣  Curve generation for charting (last N days)
        # ------------------------------------------------------------------
        def get_drawdown_curve(self, days: int = 30) -> List[Dict]:
            """
            Return a time‑series of draw‑down percentages for the past ``days``.
            Each entry contains ISO‑formatted timestamp and draw‑down percent.
            """
            cutoff = datetime.now() - timedelta(days=days)
            recent = [s for s in self.snapshots if s.timestamp >= cutoff]

            return [
                {
                    "timestamp": s.timestamp.isoformat(),
                    "drawdown_pct": s.drawdown_percent * 100,
                    "balance": s.balance,
                    "equity": s.equity,
                }
                for s in recent
            ]

        # ------------------------------------------------------------------
        # 5️⃣  Export a full JSON report (useful for audits)
        # ------------------------------------------------------------------
        def export_report(self, filename: str = "drawdown_report.json") -> None:
            """Write a comprehensive draw‑down report to ``filename``."""
            report = {
                "generated_at": datetime.utcnow().isoformat(),
                "account_name": self.account_name,
                "statistics": self.get_drawdown_stats(),
                "recovery_stats": self.get_recovery_stats(),
                "drawdown_curve": self.get_drawdown_curve(days=365),
                "periods": [
                    {
                        "start": p.start_time.isoformat(),
                        "end": p.end_time.isoformat() if p.end_time else None,
                        "peak_balance": p.peak_balance,
                        "valley_balance": p.valley_balance,
                        "max_drawdown_pct": p.max_drawdown_pct * 100,
                        "recovered": p.is_recovered,
                        "recovery_seconds": p.recovery_time.total_seconds()
                        if p.recovery_time
                        else None,
                    }
                    for p in self.periods
                ],
                "thresholds": self.thresholds,
            }

            try:
                with open(filename, "w") as f:
                    json.dump(report, f, indent=2)
                logger.info(f"📊 Drawdown report exported to {filename}")
            except Exception as exc:   # pragma: no cover
                logger.error(f"Failed to export drawdown report: {exc}")

    # ----------------------------------------------------------------------
    # Global singleton – importable from anywhere in the code‑base
    # ----------------------------------------------------------------------
    drawdown_tracker = EnhancedDrawdownTracker()

    # ----------------------------------------------------------------------
    # Simple demo / sanity‑check when the file is executed directly
    # ----------------------------------------------------------------------
    if __name__ == "__main__":
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        logger.info("=" * 80)
        logger.info("🧪 DRAWNDOWN TRACKER DEMO")
        logger.info("=" * 80)

        # Initialise a fresh tracker (balance $10,000)
        demo_tracker = EnhancedDrawdownTracker(initial_balance=10_000.0, account_name="demo")

        # Simulated equity curve (balance stays constant, equity fluctuates)
        simulated_equities = [
            10_000,  # flat
            9_800,   # 2 % draw‑down
            9_600,   # 4 % draw‑down
            9_400,   # 6 % draw‑down (crosses warning)
            9_200,   # 8 % draw‑down (critical)
            9_500,   # partial recovery
            9_900,   # near‑full recovery
            10_200,  # new peak (recovery complete)
        ]

        for eq in simulated_equities:
            demo_tracker.record_snapshot(balance=10_000, equity=eq)

        # Show a few key stats
        stats = demo_tracker.get_drawdown_stats()
        logger.info("\n📈 Current Statistics")
        logger.info(f"  Peak balance          : ${stats['peak_balance']:,.2f}")
        logger.info(f"  Current equity        : ${stats['current_equity']:,.2f}")
        logger.info(f"  Current draw‑down     : {stats['current_drawdown_pct']*100:.2f}%")
        logger.info(f"  Max historic draw‑down: {stats['max_drawdown_pct']*100:.2f}%")
        logger.info(f"  In draw‑down?         : {stats['in_drawdown']}")

        # Recovery summary
        rec = demo_tracker.get_recovery_stats()
        logger.info("\n♻️  Recovery Summary")
        logger.info(f"  Completed recoveries : {rec['recovery_count']}")
        if rec["avg_recovery_seconds"] is not None:
            logger.info(f"  Avg recovery time    : {rec['avg_recovery_seconds']/3600:.2f} h")
            logger.info(f"  Fastest recovery     : {rec['fastest_recovery_seconds']/3600:.2f} h")
            logger.info(f"  Slowest recovery     : {rec['slowest_recovery_seconds']/3600:.2f} h")

        # Export a JSON report (saved alongside this script)
        demo_tracker.export_report("demo_drawdown_report.json")

        logger.info("\n✅ Demo complete – inspect 'demo_drawdown_report.json'")
        logger.info("=" * 80)

if kill_switch.update_drawdown(current_drawdown):
    # Call the API that pauses all buckets
    requests.post("http://localhost:8000/api/v1/kill_switch")
    logger.warning("Kill‑switch engaged – draw‑down sustained above %.0f%%", DD_THRESHOLD*100)
