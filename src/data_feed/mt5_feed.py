# src/data_feed/mt5_feed.py
from .abstract_feed import DataFeed
import MetaTrader5 as mt5
import pandas as pd


class MT5Feed(DataFeed):
    def __init__(self):
        if not mt5.initialize():
            raise RuntimeError("MT5 init failed")


    def fetch(self, symbol: str, timeframe: str) -> pd.DataFrame:
        tf_map = {"M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5,
                  "H1": mt5.TIMEFRAME_H1, "D1": mt5.TIMEFRAME_D1}
        rates = mt5.copy_rates_range(symbol, tf_map[timeframe],
                                     mt5.time_from_string("2020-01-01"),
                                     mt5.time_to_string("2025-12-31"))
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        return df[["time","open","high","low","close","tick_volume"]]


    def latest_tick(self, symbol: str) -> dict:
        tick = mt5.symbol_info_tick(symbol)
        return {"price": tick.bid, "volume": tick.volume, "time": tick.time}
