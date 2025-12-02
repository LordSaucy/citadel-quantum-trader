# tests/test_features.py
import pandas as pd
import numpy as np
from features import atr_scaled_stop, vwap_bias, fractal_distance

def dummy_df():
    # 30 rows of synthetic 1‑min candles
    rng = pd.date_range('2024-01-01', periods=30, freq='T')
    data = {
        'high':   np.linspace(1.1000, 1.1010, 30) + np.random.rand(30)*0.0005,
        'low':    np.linspace(1.0990, 1.1000, 30) - np.random.rand(30)*0.0005,
        'close':  np.linspace(1.0995, 1.1005, 30) + np.random.randn(30)*0.0002,
        'volume': np.random.randint(1000, 5000, 30),
        'buy_volume':  np.random.randint(500, 2500, 30),
        'sell_volume': np.random.randint(500, 2500, 30),
    }
    return pd.DataFrame(data, index=rng)

def test_atr_scaled_stop():
    df = dummy_df()
    sl = atr_scaled_stop(df, k=1.5, period=14)
    assert sl.notna().all()   # after enough rows we should have a value
    # stop must be below the close (since we subtract k*ATR)
    assert (sl < df['close']).all()

def test_vwap_bias():
    df = dummy_df()
    flag = vwap_bias(df, window_minutes=30)
    assert set(flag.unique()).issubset({0,1,np.nan})
    # At least one True and one False in a random series
    assert flag.sum() > 0

def test_fractal_distance():
    df = dummy_df()
    dist = fractal_distance(df, lookback=10)
    # Should return a float (or NaN) for each row
    assert len(dist) == len(df)
