import time
from datetime import datetime, timedelta
from typing import Tuple, Optional

from .config import logger, env
from prometheus_client import Gauge

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

# ----------------------------------------------------------------------
class RiskManagementLayer:
    """
    Core risk‑engine used by the trading loop.
    It is instantiated once per process (the Engine) and consulted
    before every order is sent to the broker.
    """

    def __init__(self) -> None:
        self._kill_switch_active: bool = False
        self._kill_reason: Optional[str] = None
        self._last_reset: datetime = datetime.utcnow()
        logger.info("RiskManagementLayer initialised")

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
