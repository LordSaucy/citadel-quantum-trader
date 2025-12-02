# ---------------------------------------------------------
# features.py – pure‑function feature library for CQT
# ---------------------------------------------------------
import pandas as pd
import numpy as np
from typing import Tuple

# ------------------------------------------------------------------
# Helper: safe rolling window that returns NaN when not enough data
# ------------------------------------------------------------------
def _rolling_safe(series: pd.Series, window: int, func):
    if len(series) < window:
        return pd.Series([np.nan] * len(series), index=series.index)
    return series.rolling(window).apply(func, raw=True)


# ------------------------------------------------------------------
# 1️⃣  Dynamic ATR‑scaled stop‑loss
# ------------------------------------------------------------------
def atr_scaled_stop(df: pd.DataFrame, k: float = 1.5, period: int = 14) -> pd.Series:
    """
    Returns a stop‑loss price series:
        SL = entry_price - k * ATR(period)

    Parameters
    ----------
    df : DataFrame
        Must contain columns ['high','low','close'].
    k : float
        Multiplier of ATR (optimisable, typical 1.0‑2.5).
    period : int
        ATR look‑back period (default 14).

    Returns
    -------
    pd.Series of stop‑loss prices (NaN where not enough data).
    """
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low']  - df['close'].shift()).abs()
    ], axis=1).max(axis=1)

    atr = _rolling_safe(tr, period, lambda x: x.mean())
    # Assume entry_price is the close of the bar that generated the signal.
    # The caller will shift the series appropriately (e.g., entry_price = close.shift(1)).
    stop = df['close'] - k * atr
    return stop


# ------------------------------------------------------------------
# 2️⃣  VWAP‑bias indicator (binary flag)
# ------------------------------------------------------------------
def vwap_bias(df: pd.DataFrame, window_minutes: int = 30) -> pd.Series:
    """
    Binary flag: 1 if price is above VWAP over the last N minutes,
    0 otherwise. Uses the typical VWAP formula:
        VWAP = Σ(price * volume) / Σ(volume)
    """
    # Convert minutes → number of rows (assumes 1‑minute candles)
    window = window_minutes
    if len(df) < window:
        return pd.Series([np.nan] * len(df), index=df.index)

    pv = (df['close'] * df['volume']).rolling(window).sum()
    vol = df['volume'].rolling(window).sum()
    vwap = pv / vol
    flag = (df['close'] > vwap).astype(int)
    return flag


# ------------------------------------------------------------------
# 3️⃣  Bill Williams Fractals (swing high / low distance)
# ------------------------------------------------------------------
def fractal_distance(df: pd.DataFrame, lookback: int = 10) -> pd.Series:
    """
    Returns the absolute distance (in price) from the current close
    to the nearest fractal high or low within the lookback window.
    Positive value = distance to nearest fractal high,
    Negative value = distance to nearest fractal low.
    """
    def _is_fractal_high(i):
        # A high is a fractal if it is greater than the two bars before
        # and two bars after it.
        if i < 2 or i > len(df) - 3:
            return False
        cur = df['high'].iloc[i]
        prev = df['high'].iloc[i-2:i]
        nxt  = df['high'].iloc[i+1:i+3]
        return cur > prev.max() and cur > nxt.max()

    def _is_fractal_low(i):
        if i < 2 or i > len(df) - 3:
            return False
        cur = df['low'].iloc[i]
        prev = df['low'].iloc[i-2:i]
        nxt  = df['low'].iloc[i+1:i+3]
        return cur < prev.min() and cur < nxt.min()

    highs = [i for i in range(len(df)) if _is_fractal_high(i)]
    lows  = [i for i in range(len(df)) if _is_fractal_low(i)]

    # Build series of distances
    dist = pd.Series(index=df.index, dtype=float)

    for idx in range(len(df)):
        # distance to nearest high
        if highs:
            nearest_high = min(highs, key=lambda h: abs(h - idx))
            d_high = df['close'].iloc[idx] - df['high'].iloc[nearest_high]
        else:
            d_high = np.nan

        # distance to nearest low
        if lows:
            nearest_low = min(lows, key=lambda l: abs(l - idx))
            d_low = df['close'].iloc[idx] - df['low'].iloc[nearest_low]
        else:
            d_low = np.nan

        # Choose the *smaller* absolute distance (closest swing)
        if np.isnan(d_high) and np.isnan(d_low):
            dist.iloc[idx] = np.nan
        elif np.isnan(d_high):
            dist.iloc[idx] = d_low
        elif np.isnan(d_low):
            dist.iloc[idx] = d_high
        else:
            dist.iloc[idx] = d_high if abs(d_high) < abs(d_low) else d_low

    return dist


# ------------------------------------------------------------------
# 4️⃣  Time‑of‑day decay factor (sinusoidal weighting)
# ------------------------------------------------------------------
def time_of_day_decay(df: pd.DataFrame,
                      peak_start: str = "08:00",
                      peak_end:   str = "12:00") -> pd.Series:
    """
    Returns a weight ∈ [0, 1] that peaks during the most liquid session.
    The shape is a simple cosine that goes from 0 → 1 → 0 across the window.
    Times are interpreted in the data’s timezone (usually UTC).
    """
    # Convert strings to minutes since midnight
    def _to_minutes(t: str) -> int:
        h, m = map(int, t.split(":"))
        return h * 60 + m

    start_min = _to_minutes(peak_start)
    end_min   = _to_minutes(peak_end)

    # If the window wraps midnight (unlikely for our 8‑12 window) handle it:
    window_len = (end_min - start_min) % (24 * 60)

    # Extract minute‑of‑day from the index (assumes DateTimeIndex)
    minutes = df.index.time
    minutes_since_midnight = np.array([t.hour * 60 + t.minute for t in minutes])

    # Compute distance to the centre of the window
    centre = (start_min + window_len / 2) % (24 * 60)
    dist = np.abs(minutes_since_midnight - centre)
    # Wrap around midnight
    dist = np.minimum(dist, 24 * 60 - dist)

    # Cosine weighting: 1 at centre, 0 at edges
    weight = 0.5 * (1 + np.cos(np.pi * dist / (window_len / 2)))
    weight[dist > window_len / 2] = 0.0   # outside the window → zero weight
    return pd.Series(weight, index=df.index)


# ------------------------------------------------------------------
# 5️⃣  Regime‑specific lever weights loader
# ------------------------------------------------------------------
def load_regime_weights(cfg: dict, regime: str) -> pd.Series:
    """
    Reads a JSON file that contains the weight vector for the given regime.
    The JSON format is: {"feature_name": weight, ...}
    """
    import json, pathlib
    path = pathlib.Path(cfg['features']['regime_weights'][regime])
    with open(path) as f:
        w = json.load(f)
    return pd.Series(w)


# ------------------------------------------------------------------
# 6️⃣  Order‑flow delta (buy‑sell imbalance)
# ------------------------------------------------------------------
def order_flow_delta(df: pd.DataFrame, lookback_ticks: int = 200) -> pd.Series:
    """
    Cumulative imbalance over the last N ticks:
        delta = Σ(buy_volume) - Σ(sell_volume)

    The source DataFrame must contain columns:
        - buy_volume
        - sell_volume
    """
    buy_cum  = df['buy_volume'].rolling(window=lookback_ticks).sum()
    sell_cum = df['sell_volume'].rolling(window=lookback_ticks).sum()
    delta = buy_cum - sell_cum
    return delta


# ------------------------------------------------------------------
# FEATURE REGISTRY – what the signal engine will iterate over
# ------------------------------------------------------------------
FEATURES = {
    "atr_stop":          atr_scaled_stop,
    "vwap_bias":         vwap_bias,
    "fractal_dist":      fractal_distance,
    "tod_decay":         time_of_day_decay,
    "order_flow_delta":  order_flow_delta,
    # add new features here – just import the function and add a key
}
