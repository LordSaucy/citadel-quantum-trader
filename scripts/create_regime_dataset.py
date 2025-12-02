# scripts/create_regime_dataset.py
import pandas as pd
import numpy as np

df = pd.read_csv('data/ohlc_1min.csv')   # columns: timestamp, open, high, low, close
df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
df['atr_1h'] = df['close'].rolling(window=60).apply(lambda x: np.max(x)-np.min(x))

def label(row):
    price = row['close']
    if row['ema20'] > row['ema50'] and row['atr_1h'] > 0.005 * price:
        return 0          # Trend
    if abs(row['ema20'] - row['ema50']) / price < 0.001 and row['atr_1h'] < 0.002 * price:
        return 1          # Range
    if row['atr_1h'] > 0.015 * price:
        return 2          # High‑Vol
    return 1              # Default to Range if none match

df['regime'] = df.apply(label, axis=1)

# Keep only the macro features we will feed the classifier
features = df[['ema20','ema50','atr_1h','close']].shift(1).fillna(method='bfill')
features['regime'] = df['regime']
features.to_csv('data/regime_training.csv', index=False)
