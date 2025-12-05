import collections
import numpy as np
import time
from prometheus_client import Gauge

# Prometheus metrics
VOL_GAUGE = Gauge('cqt_volatility', 'Real‑time volatility (ATR)', ['symbol'])
DRF_GAUGE = Gauge('cqt_dynamic_risk_fraction',
                  'Dynamic risk‑fraction after volatility scaling', ['symbol'])

class VolatilityCalculator:
    """
    Maintains a rolling window of price bars and returns a volatility
    estimate (e.g., ATR) for a given symbol.
    """
    def __init__(self, window_minutes: int = 30):
        self.window = collections.deque(maxlen=window_minutes)
        self.window_minutes = window_minutes

    def add_bar(self, price: float):
        """Append a new close price; keep timestamp for sanity."""
        self.window.append((time.time(), price))

    def atr(self) -> float:
        """Average True Range over the stored window."""
        if len(self.window) < 2:
            return 0.0
        "_", "_", "_" = [], [], []
        # we only have close prices; approximate TR = |Δprice|
        prev_price = self.window[0][1]
        tr_vals = []
        for _, price in list(self.window)[1:]:
            tr = abs(price - prev_price)
            tr_vals.append(tr)
            prev_price = price
        return float(np.mean(tr_vals))

    def volatility(self) -> float:
        """Alias – you can replace with a GARCH estimator later."""
        return self.atr()
