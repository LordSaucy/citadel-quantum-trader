import time
from datetime import datetime, timedelta
from typing import Tuple, Optional

from .config import logger, env
from prometheus_client import Gauge
import logging
from datetime import datetime, timedelta

import datetime
from sqlalchemy import select, update, insert, delete
from src.edge_decay_detector import EdgeDecayDetector  # forward reference (optional)
from redis import Redis
import math
from config_loader import Config
from prometheus_client import Gauge


log = logging.getLogger("citadel_bot")

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
r = redis.from_url(REDIS_URL)
KILL_SWITCH_KEY = "kill_switch_active"

def kill_switch_active() -> bool:
    val = r.get(KILL_SWITCH_KEY)
    return bool(int(val or 0))

# Inside your infinite trading loop:
while True:
    if kill_switch_active():
        log.warning("⚠️ Kill‑switch ACTIVE – sleeping 5 s")
        time.sleep(5)
        continue   # skip all trading logic until cleared

    # … existing signal generation, risk checks, execution …


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
redis_client = Redis(host="redis", port=6379, db=0)

def should_run() -> bool:
    # 0 = off, 1 = kill‑switch active
    return not bool(int(redis_client.get("kill_switch_active") or 0))

while True:
    if not should_run():
        logger.info("⚠️ Kill‑switch active – pausing trading loop")
        time.sleep(5)          # back‑off while the flag is set
        continue
    # … normal signal generation, risk checks, execution …

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

     # ... existing code ...

    # -------------------------------------------------
    # 1️⃣ Apply a temporary multiplier for a bucket
    # -------------------------------------------------
    def apply_temporary_modifier(self, bucket_id: int, multiplier: float, trades: int):
        """Create or update a row in edge_decay_modifiers."""
        with self.engine.begin() as conn:
            stmt = select(edge_decay_modifiers.c.bucket_id).where(
                edge_decay_modifiers.c.bucket_id == bucket_id
            )
            exists = conn.execute(stmt).scalar()
            if exists:
                upd = (
                    edge_decay_modifiers.update()
                    .where(edge_decay_modifiers.c.bucket_id == bucket_id)
                    .values(multiplier=multiplier, remaining_trades=trades,
                            created_at=datetime.datetime.utcnow())
                )
                conn.execute(upd)
            else:
                ins = edge_decay_modifiers.insert().values(
                    bucket_id=bucket_id,
                    multiplier=multiplier,
                    remaining_trades=trades,
                    created_at=datetime.datetime.utcnow(),
                )
                conn.execute(ins)

    # -------------------------------------------------
    # 2️⃣ Get the *effective* risk fraction for a bucket
    # -------------------------------------------------
    def get_effective_risk_fraction(self, bucket_id: int, base_fraction: float) -> float:
        """Return base_fraction adjusted by any active edge‑decay modifier."""
        with self.engine.begin() as conn:
            row = conn.execute(
                select(edge_decay_modifiers.c.multiplier,
                       edge_decay_modifiers.c.remaining_trades)
                .where(edge_decay_modifiers.c.bucket_id == bucket_id)
            ).first()
            if row:
                # Decrement the counter (one trade consumed)
                new_remaining = max(row.remaining_trades - 1, 0)
                if new_remaining == 0:
                    # Modifier expired – delete it
                    conn.execute(
                        edge_decay_modifiers.delete().where(
                            edge_decay_modifiers.c.bucket_id == bucket_id
                        )
                    )
                else:
                    conn.execute(
                        edge_decay_modifiers.update()
                        .where(edge_decay_modifiers.c.bucket_id == bucket_id)
                        .values(remaining_trades=new_remaining)
                    )
                return base_fraction * float(row.multiplier)
        return base_fraction

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
        with engine.connect() as conn:
        f = get_risk_fraction(bucket_id, conn)

        """
        Returns the dollar amount to risk on the next trade.
        The stake is calculated as:
            stake = usable_capital * risk_fraction_from_schedule
        """
          with engine.connect() as conn:
        # 1️⃣ Base fraction (schedule or bucket_meta)
        base_frac = _get_base_fraction(bucket_id, conn)

        # 2️⃣ Apply any active edge‑decay modifier
        final_frac = _apply_edge_modifier(bucket_id, conn, base_frac)

    # 3️⃣ Compute stake in dollars
    stake = equity * final_frac
    return stake

        # 1️⃣ Get the risk fraction from the schedule (same as before)
        trade_idx = get_trade_counter(bucket_id) + 1
        if trade_idx in RISK_SCHEDULE:
            risk_frac = RISK_SCHEDULE[trade_idx]
        else:
            risk_frac = RISK_SCHEDULE.get("default", 0.40)

        effective_frac = self.get_effective_risk_fraction(bucket_id, base_frac)

        # 2️⃣ Get the *usable* capital (aggressive pool) after the last trade
        usable_capital = self.calculate_usable_capital(bucket_id, equity)

        # 3️⃣ Apply the schedule fraction
        stake = equity * effective_frac
        return stake

def get_risk_fraction(bucket_id: int, conn) -> float:
    cur = conn.cursor()
    cur.execute(
        "SELECT risk_fraction FROM bucket_meta WHERE bucket_id = %s",
        (bucket_id,)
    )
    row = cur.fetchone()
    if row:
        return float(row[0])
    # fallback to schedule dict if meta missing
    return RISK_SCHEDULE.get(bucket_id, RISK_SCHEDULE.get("default", 0.40))

def _get_base_fraction(bucket_id: int, conn) -> float:
    """
    Returns the base risk fraction for the bucket:
    1️⃣ First look in bucket_meta (global‑risk controller updates)
    2️⃣ Fall back to the static schedule dict
    """
    cur = conn.cursor()
    cur.execute(
        "SELECT risk_fraction FROM bucket_meta WHERE bucket_id = %s",
        (bucket_id,),
    )
    row = cur.fetchone()
    if row:
        return float(row[0])

    # static schedule fallback (same logic you already have)
    trade_idx = get_trade_counter(bucket_id) + 1
    if trade_idx in cfg["risk_schedule"]:
        return cfg["risk_schedule"][trade_idx]
    return cfg["risk_schedule"].get("default", 0.40)


def _apply_edge_modifier(bucket_id: int, conn, base_fraction: float) -> float:
    """
    If a temporary edge‑decay row exists, apply its multiplier
    and decrement the remaining‑trades counter.
    When the counter reaches 0 the row is deleted.
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT multiplier, remaining_trades
        FROM risk_modifier
        WHERE bucket_id = %s
        FOR UPDATE
        """,
        (bucket_id,),
    )
    row = cur.fetchone()
    if not row:
        return base_fraction

    multiplier, remaining = row
    new_fraction = base_fraction * multiplier

    # Decrement the counter
    if remaining <= 1:
        cur.execute(
            "DELETE FROM risk_modifier WHERE bucket_id = %s", (bucket_id,)
        )
    else:
        cur.execute(
            """
            UPDATE risk_modifier
            SET remaining_trades = remaining_trades - 1
            WHERE bucket_id = %s
            """,
            (bucket_id,),
        )
    conn.commit()
    return new_fraction

def log_depth_check(self, **kwargs):
    from .trade_logger import TradeLogger
    TradeLogger().log_depth_check(**kwargs)

# -------------------------------------------------
# New Prometheus gauge – shows the *effective* leverage
# -------------------------------------------------
effective_leverage_gauge = Gauge(
    "effective_leverage",
    "Leverage implied by the current lot size (1 = 1:1, 100 = 1:100)",
    ["bucket_id"]
)

# -------------------------------------------------
# Helper: compute the *maximum* lot size that satisfies
# the broker’s margin requirement given current equity.
# -------------------------------------------------
def max_lot_by_margin(equity: float, risk_frac: float, cfg: dict) -> float:
    """
    equity      – current bucket equity (USD)
    risk_frac   – fraction of equity we are willing to risk on the trade
    cfg         – full config dict (contains broker params)
    Returns the largest lot size (in broker lot units) that can be opened
    without violating the broker’s margin rule.
    """
    # 1️⃣ Desired risk amount (the “R” we are willing to lose)
    risk_amount = equity * risk_frac          # $ at risk

    # 2️⃣ Broker parameters
    contract_notional = cfg.get("broker", {}).get("contract_notional", 100_000)
    max_leverage      = cfg.get("broker", {}).get("max_leverage", 100)
    margin_factor     = cfg.get("broker", {}).get("margin_factor", 1.0 / max_leverage)

    # 3️⃣ Margin required for ONE LOT at the *desired* risk amount:
    #    margin_per_lot = contract_notional * margin_factor
    margin_per_lot = contract_notional * margin_factor

    # 4️⃣ Maximum lot we can afford with the *available* equity:
    #    available_margin = equity - (equity - risk_amount)   <-- we keep the rest as reserve
    #    But a simpler safe bound is: we cannot allocate more margin than the
    #    *risk amount* itself (otherwise we would be risking > risk_frac).
    #    So we cap the lot by the smaller of:
    #       a) risk_amount / (contract_notional * risk_frac)   (the “risk‑based” lot)
    #       b) equity / margin_per_lot                        (the “margin‑based” lot)
    #
    #    The formula below does exactly that.
    lot_by_risk   = risk_amount / (contract_notional * risk_frac)   # = 1.0 normally
    lot_by_margin = equity / margin_per_lot

    # Choose the stricter (smaller) lot size
    max_lot = min(lot_by_risk, lot_by_margin)

    # Ensure we never exceed the broker’s hard max_lot (if defined)
    broker_max_lot = cfg.get("broker_max_lot", float("inf"))
    max_lot = min(max_lot, broker_max_lot)

    # Return a *float* – the caller will later round to the broker’s step size
    return max_lot


# -------------------------------------------------
# Update compute_stake() to use the dynamic lot size
# -------------------------------------------------
def compute_stake(bucket_id: int, equity: float) -> float:
    """
    Returns the *dollar* amount to risk on the next trade.
    This function now:
    1️⃣ Reads the configured risk_fraction (schedule or default)
    2️⃣ Calculates the *maximum* lot allowed by margin
    3️⃣ If the requested lot (equity * f / contract_notional) exceeds that,
       we *scale down* the risk_fraction so the lot fits.
    4️⃣ Emits the effective leverage as a Prometheus gauge.
    """
    # ----- 1️⃣ Get the schedule fraction -----
    trade_idx = get_trade_counter(bucket_id) + 1
    f = RISK_SCHEDULE.get(trade_idx, RISK_SCHEDULE.get("default", 0.40))

    # ----- 2️⃣ Compute the lot we *want* to trade -----
    desired_lot = (equity * f) / cfg["contract_notional"]

    # ----- 3️⃣ Compute the broker‑allowed max lot -----
    max_allowed_lot = max_lot_by_margin(equity, f, cfg)

    # ----- 4️⃣ If we exceed, scale down the fraction -----
    if desired_lot > max_allowed_lot:
        # Scale the risk fraction proportionally so the lot fits exactly
        scaling = max_allowed_lot / desired_lot
        f = f * scaling
        # Re‑compute the stake with the scaled fraction
        stake = equity * f
    else:
        stake = equity * f

    # ----- 5️⃣ Record effective leverage for monitoring -----
    # Effective leverage = (lot * contract_notional) / stake
    # (how many times the risk amount is amplified by the position size)
    effective_leverage = (desired_lot * cfg["contract_notional"]) / (equity * f)
    effective_leverage_gauge.labels(bucket_id=str(bucket_id)).set(effective_leverage)

    # ----- 6️⃣ Return the *dollar* stake (the broker will convert to lot) -----
    return stake

