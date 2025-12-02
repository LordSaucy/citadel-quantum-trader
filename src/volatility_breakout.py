import pandas as pd
import talib

# Configurable parameters – you can expose them in config.yaml
ATR_WINDOW = 14          # number of bars for ATR
MULTIPLIER = 2.0         # ATR × multiplier must exceed current bar range

def _atr(series: pd.DataFrame) -> pd.Series:
    """Calculate ATR for a DataFrame with columns: high, low, close."""
    return talib.ATR(series['high'], series['low'], series['close'], timeperiod=ATR_WINDOW)

def is_expanding(current_bar: dict, prev_bar: dict, hist_df: pd.DataFrame) -> bool:
    """
    current_bar / prev_bar: dicts with keys ['high','low','close','timestamp']
    hist_df: DataFrame of the *previous* N bars (including prev_bar) used to compute ATR.
    Returns True if the current bar’s total range > MULTIPLIER * ATR and also > previous bar range.
    """
    # Append the previous bar to the history (so ATR includes it)
    df = hist_df.copy()
    df = df.append(prev_bar, ignore_index=True)

    atr_series = _atr(df)
    # Use the most recent ATR value (the last one corresponds to prev_bar)
    latest_atr = atr_series.iloc[-1]

    # Bar ranges
    curr_range = current_bar['high'] - current_bar['low']
    prev_range = prev_bar['high'] - prev_bar['low']

    # Condition: current range > MULTIPLIER * ATR  AND > previous range
    return (curr_range > MULTIPLIER * latest_atr) and (curr_range > prev_range)
