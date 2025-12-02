import logging
from prometheus_client import Counter, Gauge

log = logging.getLogger("citadel.volatility_guard")

vol_spike_hits = Counter(
    "volatility_spike_guard_hits_total",
    "Trades rejected because of a volatility spike"
)

vol_spike_active = Gauge(
    "volatility_spike_active",
    "1 when a volatility spike is currently active"
)

class VolatilityGuard:
    """
    Simple ATR‑based spike detector.
    You already compute `atr_14` somewhere in the market‑data pipeline
    (e.g. `src/market_data_manager.py`). Pass that value in when you call
    `check(...)`.
    """

    def __init__(self, cfg):
        """
        cfg – expects:
            volatility_spike:
                enabled: true
                atr_multiplier: 2.0   # spike if current ATR > 2× baseline
                baseline_atr: 0.0005   # you can compute this once per day
        """
        self.enabled = cfg.get("volatility_spike", {}).get("enabled", True)
        self.mult = cfg["volatility_spike"].get("atr_multiplier", 2.0)
        self.base = cfg["volatility_spike"].get("baseline_atr", 0.0005)

    def check(self, current_atr: float) -> bool:
        """
        Returns True if the trade may proceed, False if a spike is detected.
        """
        if not self.enabled:
            vol_spike_active.set(0)
            return True

        if current_atr > self.base * self.mult:
            vol_spike_active.set(1)
            vol_spike_hits.inc()
            log.warning("[VolatilityGuard] Spike detected – ATR %.6f > %.6f",
                        current_atr, self.base * self.mult)
            return False

        vol_spike_active.set(0)
        return True
