# src/kill_switch.py  (or wherever the DD monitor lives)
from risk_management_layer import current_max_dd_allowed
from collections import deque
from config_loader import Config


def should_kill_switch(drawdown_pct: float) -> bool:
    # drawdown_pct is a positive fraction (e.g., 0.12 = 12 %)
    return drawdown_pct >= current_max_dd_allowed()


cfg = Config().settings
DD_THRESHOLD = cfg.get("kill_switch_dd_threshold", 0.15)   # 15 %
DD_HYSTERESIS_COUNT = cfg.get("dd_hysteresis_count", 3)   # 3 consecutive checks

dd_history = deque(maxlen=DD_HYSTERESIS_COUNT)

def update_drawdown(drawdown_pct: float) -> bool:
    """
    Returns True if the kill‑switch should fire.
    """
    dd_history.append(drawdown_pct)
    if len(dd_history) < DD_HYSTERESIS_COUNT:
        return False
    # Fire only if *every* entry in the window exceeds the threshold
    return all(d >= DD_THRESHOLD for d in dd_history)
