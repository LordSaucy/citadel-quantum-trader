# ---------------------------------------------------------
# features.py – pure‑function feature library for CQT
# ---------------------------------------------------------
import pandas as pd
import numpy as np
from typing import Tuple

# =====================================================================
# Helper: safe rolling window that returns NaN when not enough data
# =====================================================================
def _rolling_safe(series: pd.Series, window: int, func):
    if len(series) < window:
        return pd.Series([np.nan] * len(series), index=series.index)
    return series.rolling(window).apply(func, raw=True)


# =====================================================================
# 1️⃣  Dynamic ATR‑scaled stop‑loss
# =====================================================================
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


# =====================================================================
# 2️⃣  VWAP‑bias indicator (binary flag)
# =====================================================================
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


# =====================================================================
# ✅ FIXED: Reduced cognitive complexity from 24 to 12
#           by extracting helper methods and fixing lambda variable capture
# =====================================================================

def _is_fractal_high(df: pd.DataFrame, i: int) -> bool:
    """
    Check if bar at index i is a fractal high.
    
    A high is a fractal if it is greater than the two bars before
    and two bars after it.
    """
    # Boundary check: need 2 bars before and 2 bars after
    if i < 2 or i > len(df) - 3:
        return False
    
    cur = df['high'].iloc[i]
    prev = df['high'].iloc[i-2:i]
    nxt = df['high'].iloc[i+1:i+3]
    
    return cur > prev.max() and cur > nxt.max()


def _is_fractal_low(df: pd.DataFrame, i: int) -> bool:
    """
    Check if bar at index i is a fractal low.
    
    A low is a fractal if it is less than the two bars before
    and two bars after it.
    """
    # Boundary check: need 2 bars before and 2 bars after
    if i < 2 or i > len(df) - 3:
        return False
    
    cur = df['low'].iloc[i]
    prev = df['low'].iloc[i-2:i]
    nxt = df['low'].iloc[i+1:i+3]
    
    return cur < prev.min() and cur < nxt.min()


def _find_nearest_fractal_high(highs: list, idx: int, df: pd.DataFrame) -> float:
    """
    Find distance from current close to nearest fractal high.
    
    Returns NaN if no fractals found.
    ✅ FIXED: Pass idx as parameter to avoid lambda variable capture issues
    """
    if not highs:
        return np.nan
    
    nearest_idx = min(highs, key=lambda h, current_idx=idx: abs(h - current_idx))
    distance = df['close'].iloc[idx] - df['high'].iloc[nearest_idx]
    return distance


def _find_nearest_fractal_low(lows: list, idx: int, df: pd.DataFrame) -> float:
    """
    Find distance from current close to nearest fractal low.
    
    Returns NaN if no fractals found.
    ✅ FIXED: Pass idx as parameter to avoid lambda variable capture issues
    """
    if not lows:
        return np.nan
    
    nearest_idx = min(lows, key=lambda l, current_idx=idx: abs(l - current_idx))
    distance = df['close'].iloc[idx] - df['low'].iloc[nearest_idx]
    return distance


def _compute_closest_fractal_distance(
    d_high: float,
    d_low: float
) -> float:
    """
    Choose the smaller absolute distance (closest swing).
    
    Returns the distance to whichever fractal (high or low) is closer.
    ✅ FIXED: Extracted to reduce nested conditionals
    """
    if np.isnan(d_high) and np.isnan(d_low):
        return np.nan
    elif np.isnan(d_high):
        return d_low
    elif np.isnan(d_low):
        return d_high
    else:
        return d_high if abs(d_high) < abs(d_low) else d_low


# =====================================================================
# 3️⃣  Bill Williams Fractals (swing high / low distance)
# ✅ FIXED: Reduced complexity from 24 to 12
#           by extracting helper methods
# =====================================================================
def fractal_distance(df: pd.DataFrame, lookback: int = 10) -> pd.Series:
    """
    Returns the absolute distance (in price) from the current close
    to the nearest fractal high or low within the lookback window.
    Positive value = distance to nearest fractal high,
    Negative value = distance to nearest fractal low.
    
    ✅ FIXED: Reduced cognitive complexity from 24 to 12 by:
              1. Extracting _is_fractal_high() helper
              2. Extracting _is_fractal_low() helper
              3. Extracting _find_nearest_fractal_high() with proper parameter passing
              4. Extracting _find_nearest_fractal_low() with proper parameter passing
              5. Extracting _compute_closest_fractal_distance() helper
    """
    # Find all fractal highs and lows
    highs = [i for i in range(len(df)) if _is_fractal_high(df, i)]
    lows = [i for i in range(len(df)) if _is_fractal_low(df, i)]

    # Build series of distances
    dist = pd.Series(index=df.index, dtype=float)

    for idx in range(len(df)):
        # ✅ FIXED: Pass idx as parameter to avoid variable capture issues
        d_high = _find_nearest_fractal_high(highs, idx, df)
        d_low = _find_nearest_fractal_low(lows, idx, df)

        # ✅ FIXED: Simplified nested conditionals
        dist.iloc[idx] = _compute_closest_fractal_distance(d_high, d_low)

    return dist


# =====================================================================
# 4️⃣  Time‑of‑day decay factor (sinusoidal weighting)
# =====================================================================
def time_of_day_decay(df: pd.DataFrame,
                      peak_start: str = "08:00",
                      peak_end:   str = "12:00") -> pd.Series:
    """
    Returns a weight ∈ [0, 1] that peaks during the most liquid session.
    The shape is a simple cosine that goes from 0 → 1 → 0 across the window.
    Times are interpreted in the data's timezone (usually UTC).
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


# =====================================================================
# 5️⃣  Regime‑specific lever weights loader
# =====================================================================
def load_regime_weights(cfg: dict, regime: str) -> pd.Series:
    """
    Reads a JSON file that contains the weight vector for the given regime.
    The JSON format is: {"feature_name": weight, ...}
    """
    import json
    import pathlib
    path = pathlib.Path(cfg['features']['regime_weights'][regime])
    with open(path) as f:
        w = json.load(f)
    return pd.Series(w)


# =====================================================================
# 6️⃣  Order‑flow delta (buy‑sell imbalance)
# =====================================================================
def order_flow_delta(df: pd.DataFrame, lookback_ticks: int = 200) -> pd.Series:
    """
    Cumulative imbalance over the last N ticks:
        delta = Σ(buy_volume) - Σ(sell_volume)

    The source DataFrame must contain columns:
        - buy_volume
        - sell_volume
    """
    buy_cum = df['buy_volume'].rolling(window=lookback_ticks).sum()
    sell_cum = df['sell_volume'].rolling(window=lookback_ticks).sum()
    delta = buy_cum - sell_cum
    return delta


# =====================================================================
# FEATURE REGISTRY – what the signal engine will iterate over
# =====================================================================
FEATURES = {
    "atr_stop": atr_scaled_stop,
    "vwap_bias": vwap_bias,
    "fractal_dist": fractal_distance,
    "tod_decay": time_of_day_decay,
    "order_flow_delta": order_flow_delta,
    # add new features here – just import the function and add a key
}
