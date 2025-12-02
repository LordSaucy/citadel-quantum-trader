# src/strategy.py
from typing import Dict
import pandas as pd

def build_signal_function(params: dict) -> Callable[[pd.DataFrame], Dict]:
    """
    Returns a callable that receives a DataFrame (historical up‑to‑current bar)
    and returns a signal dict compatible with BacktestValidator.
    """
    def signal(data: pd.DataFrame) -> Dict:
        # Your actual logic – e.g. confluence scoring, optimiser output, etc.
        # Must return a dict with keys: symbol, direction, entry_price,
        # stop_loss, take_profit, volume (optional).
        …
        return signal_dict
