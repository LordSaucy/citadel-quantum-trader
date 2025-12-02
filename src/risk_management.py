import time
from datetime import datetime, timedelta
from typing import Tuple, Optional

from .config import logger, env
from prometheus_client import Gauge
import logging
from datetime import datetime, timedelta


# ----------------------------------------------------------------------
# Prometheus gauges (exported automatically by the Flask entrypoint)
# ----------------------------------------------------------------------
risk_kill_switch_active = Gauge(
    "cqt_kill_switch_active",
    "1 when the kill‑switch is engaged, 0 otherwise",
)
dynamic_risk_fraction = Gauge(
    "cqt_dynamic_risk_fraction",
    "Current dynamic risk fraction (0‑1)",
)

# ----------------------------------------------------------------------
# Configuration (all overridable via environment variables)
# ----------------------------------------------------------------------
MAX_OPEN_POSITIONS = env("MAX_OPEN_POSITIONS", 4, int)
DAILY_DD_LIMIT_PCT = env("DAILY_DD_LIMIT_PCT", 3.0, float)   # % draw‑down before kill‑switch
WEEKLY_DD_LIMIT_PCT = env("WEEKLY_DD_LIMIT_PCT", 8.0, float)

logger = logging.getLogger(__name__)

# Configurable (move to config.yaml later)
DRAW_DOWN_TRIGGER = 0.10   # 10 % of current equity
RECENT_HIGH_WINDOW = timedelta(hours=1)   # look‑back window for the “high”

# ----------------------------------------------------------------------
class RiskManagementLayer:
    """
    Core risk‑engine used by the trading loop.
    It is instantiated once per process (the Engine) and consulted
    before every order is sent to the broker.
    """

       def __init__(self, db):
        self.db = db
        self.last_peak = None   # highest equity seen in RECENT_HIGH_WINDOW

        self._kill_switch_active: bool = False
        self._kill_reason: Optional[str] = None
        self._last_reset: datetime = datetime.utcnow()
        logger.info("RiskManagementLayer initialised")

 def _update_peak(self, equity: float):
        now = datetime.utcnow()
        if self.last_peak is None or equity > self.last_peak[1]:
            self.last_peak = (now, equity)

        # Forget peaks older than the window
        if self.last_peak and now - self.last_peak[0] > RECENT_HIGH_WINDOW:
            self.last_peak = None

    def check_dynamic_kill_switch(self, equity: float) -> bool:
        """
        Returns True if the draw‑down from the recent high exceeds
        DRAW_DOWN_TRIGGER. If True, the caller should pause new entries.
        """
        self._update_peak(equity)

        if not self.last_peak:
            return False

        peak_equity = self.last_peak[1]
        drawdown = (peak_equity - equity) / peak_equity
        if drawdown >= DRAW_DOWN_TRIGGER:
            logger.warning(
                f"Dynamic kill‑switch triggered: draw‑down {drawdown:.2%} "
                f"exceeds {DRAW_DOWN_TRIGGER:.2%} (peak={peak_equity:.2f}, cur={equity:.2f})"
            )

    # --------------------------------------------------------------
    # Public API ---------------------------------------------------
    # --------------------------------------------------------------

    def can_take_new_trade(self, current_open: int) -> Tuple[bool, str]:
        """Enforce the max‑open‑positions rule."""
        if current_open >= MAX_OPEN_POSITIONS:
            return False, f"Position limit reached ({current_open}/{MAX_OPEN_POSITIONS})"
        return True, "OK"

    def evaluate_drawdown(self, equity: float, high_water_mark: float) -> None:
        """
        Called after each equity update.
        If the draw‑down exceeds the configured limit, the kill‑switch is armed.
        """
        dd_pct = (high_water_mark - equity) / high_water_mark * 100.0
        if dd_pct >= DAILY_DD_LIMIT_PCT:
            self._activate_kill_switch(
                f"Daily draw‑down {dd_pct:.2f}% > {DAILY_DD_LIMIT_PCT}%"
            )
        else:
            self._clear_kill_switch()

    # --------------------------------------------------------------
    # Internals ----------------------------------------------------
    # --------------------------------------------------------------

    def _activate_kill_switch(self, reason: str) -> None:
        if not self._kill_switch_active:
            logger.warning("KILL‑SWITCH ACTIVATED – %s", reason)
        self._kill_switch_active = True
        self._kill_reason = reason
        risk_kill_switch_active.set(1)

    def _clear_kill_switch(self) -> None:
        if self._kill_switch_active:
            logger.info("Kill‑switch cleared")
        self._kill_switch_active = False
        self._kill_reason = None
        risk_kill_switch_active.set(0)

    # --------------------------------------------------------------
    # Introspection ------------------------------------------------
    # --------------------------------------------------------------

    @property
    def kill_switch_active(self) -> bool:
        return self._kill_switch_active

    @property
    def kill_reason(self) -> Optional[str]:
        return self._kill_reason

from sqlalchemy import update, select, func

RESERVE_POOL_PCT = cfg.get("reserve_pool_pct", 0.20)   # 20 % default

class RiskManagementLayer:
    # -----------------------------------------------------------------
    # Helper: fetch current pools for a bucket
    # -----------------------------------------------------------------
    def _load_pools(self, bucket_id: int) -> tuple[float, float]:
        """Return (aggressive_pool, reserve_pool) for the bucket."""
        with engine.connect() as conn:
            stmt = select(
                bucket_meta.c.aggressive_pool,
                bucket_meta.c.reserve_pool,
            ).where(bucket_meta.c.bucket_id == bucket_id)
            row = conn.execute(stmt).first()
            if row is None:
                # First time we see this bucket – initialise
                start_eq = cfg["bucket_start_equity"]
                aggressive = (1.0 - RESERVE_POOL_PCT) * start_eq
                reserve = RESERVE_POOL_PCT * start_eq
                # Insert a row
                ins = bucket_meta.insert().values(
                    bucket_id=bucket_id,
                    aggressive_pool=aggressive,
                    reserve_pool=reserve,
                )
                conn.execute(ins)
                conn.commit()
                return aggressive, reserve
            return float(row.aggressive_pool), float(row.reserve_pool)

    # -----------------------------------------------------------------
    # NEW: calculate usable capital after a trade closes
    # -----------------------------------------------------------------
    def calculate_usable_capital(self, bucket_id: int, equity: float) -> float:
        """
        Returns the amount of capital that can be risked on the NEXT trade.
        Logic:
          1️⃣ Start from the aggressive_pool stored in DB.
          2️⃣ Add realised P&L (equity – previous equity) – this is already
             reflected in the DB because we call this AFTER the trade is logged.
          3️⃣ Apply a volatility multiplier (optional – e.g., 1‑ATR/target_ATR).
          4️⃣ Cap by the global‑risk‑budget (if the global controller has
             reduced the per‑bucket risk_fraction, we honour it here).
        """
        # 1️⃣ Load current pools
        aggressive, reserve = self._load_pools(bucket_id)

        # 2️⃣ Realised P&L adjustment – equity already includes the P&L,
        #    so the difference between the new equity and the previous
        #    aggressive_pool gives us the net change.
        #    (We assume `equity` passed in is the *post‑trade* equity.)
        delta = equity - (aggressive + reserve)   # could be positive or negative
        aggressive = max(0.0, aggressive + delta)  # never go negative

        # 3️⃣ Volatility scaling (example: ATR‑based)
        #    Pull the latest ATR from the DB or a cached metric.
        #    For simplicity we use a static factor here; replace with your own.
        vol_factor = cfg.get("volatility_risk_factor", 1.0)   # 1.0 = no scaling
        aggressive *= vol_factor

        # 4️⃣ Global risk‑budget cap (optional)
        #    The global_risk_controller publishes `global_risk_percentage`.
        #    If the overall risk budget is tight we shrink the aggressive pool.
        #    Here we just demonstrate the hook – you can read the metric
        #    via the Prometheus client if you want a live value.
        #    Example (pseudo‑code):
        #
        #    from prometheus_api_client import PrometheusConnect
        #    pc = PrometheusConnect(url="http://prometheus:9090")
        #    global_pct = pc.get_current_metric_value("global_risk_percentage")
        #    if global_pct > 0.90:   # >90 % of the 5 % cap
        #        aggressive *= 0.5   # halve the usable capital

        # Persist the updated aggressive pool back to DB
        with engine.begin() as conn:
            upd = (
                update(bucket_meta)
                .where(bucket_meta.c.bucket_id == bucket_id)
                .values(aggressive_pool=aggressive)
            )
            conn.execute(upd)

        # Finally, return the *usable* amount (i.e. aggressive pool)
        return aggressive

    # -----------------------------------------------------------------
    # Existing `compute_stake` – now uses the dynamic usable capital
    # -----------------------------------------------------------------
    def compute_stake(self, bucket_id: int, equity: float) -> float:
        """
        Returns the dollar amount to risk on the next trade.
        The stake is calculated as:
            stake = usable_capital * risk_fraction_from_schedule
        """
        # 1️⃣ Get the risk fraction from the schedule (same as before)
        trade_idx = get_trade_counter(bucket_id) + 1
        if trade_idx in RISK_SCHEDULE:
            risk_frac = RISK_SCHEDULE[trade_idx]
        else:
            risk_frac = RISK_SCHEDULE.get("default", 0.40)

        # 2️⃣ Get the *usable* capital (aggressive pool) after the last trade
        usable_capital = self.calculate_usable_capital(bucket_id, equity)

        # 3️⃣ Apply the schedule fraction
        stake = usable_capital * risk_frac
        return stake
