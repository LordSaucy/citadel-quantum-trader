# src/kill_switch.py  (or wherever the DD monitor lives)
from risk_management_layer import current_max_dd_allowed

def should_kill_switch(drawdown_pct: float) -> bool:
    # drawdown_pct is a positive fraction (e.g., 0.12 = 12 %)
    return drawdown_pct >= current_max_dd_allowed()
