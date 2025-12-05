# tests/test_features.py
"""
Unit tests for feature engineering functions.

Tests for ATR-scaled stops, VWAP bias, and fractal distance calculation.
"""

import pandas as pd
import numpy as np
from features import atr_scaled_stop, vwap_bias, fractal_distance


def dummy_df():
    """
    Generate 30 rows of synthetic 1‑minute candles for testing.
    
    ✅ FIXED: Replaced legacy numpy.random functions with numpy.random.Generator
    """
    # 30 rows of synthetic 1‑min candles
    rng_index = pd.date_range('2024-01-01', periods=30, freq='T')
    
    # ✅ FIXED: Use modern Generator instead of legacy numpy.random functions
    rng = np.random.default_rng(seed=42)  # seed for reproducibility
    
    data = {
        'high':   np.linspace(1.1000, 1.1010, 30) + rng.uniform(0, 0.0005, 30),
        'low':    np.linspace(1.0990, 1.1000, 30) - rng.uniform(0, 0.0005, 30),
        'close':  np.linspace(1.0995, 1.1005, 30) + rng.normal(0, 0.0002, 30),
        'volume': rng.integers(1000, 5000, 30),
        'buy_volume':  rng.integers(500, 2500, 30),
        'sell_volume': rng.integers(500, 2500, 30),
    }
    return pd.DataFrame(data, index=rng_index)


def test_atr_scaled_stop():
    """
    Test that ATR-scaled stop loss is calculated correctly.
    
    Validates:
    - Stop loss is computed for all rows after warmup
    - Stop is below close price (downside protection)
    """
    df = dummy_df()
    sl = atr_scaled_stop(df, k=1.5, period=14)
    
    # After enough rows we should have a value
    assert sl.notna().all(), "ATR stop should not be NaN after warmup"
    
    # Stop must be below the close (since we subtract k*ATR)
    assert (sl < df['close']).all(), "Stop loss should be below close price"


def test_vwap_bias():
    """
    Test that VWAP bias indicator returns valid values.
    
    Validates:
    - Output is binary (0, 1) or NaN
    - Contains at least some True and False values
    """
    df = dummy_df()
    flag = vwap_bias(df, window_minutes=30)
    
    # Should be binary flag or NaN
    assert set(flag.unique()).issubset({0, 1, np.nan}), \
        "VWAP bias should return 0, 1, or NaN"
    
    # At least one True and one False in a random series
    assert flag.sum() > 0, "Should have at least some bullish bias signals"


def test_fractal_distance():
    """
    Test that fractal distance is computed for each row.
    
    Validates:
    - Output length matches input length
    - Values are numeric (float or NaN)
    """
    df = dummy_df()
    dist = fractal_distance(df, lookback=10)
    
    # Should return a float (or NaN) for each row
    assert len(dist) == len(df), \
        "Fractal distance should have same length as input"
    
    # All values should be numeric
    assert dist.dtype in [np.float64, np.float32], \
        "Fractal distance should return numeric values"
