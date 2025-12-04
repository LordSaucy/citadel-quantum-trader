# src/data_feed/ibkr_feed.py
from .abstract_feed import DataFeed
from ib_insync import IB, util
import pandas as pd


class IBKRFeed(DataFeed):
    def __init__(self):
        self.ib = IB()
        self.ib.connect("127.0.0.1", 7497, clientId=1)   # assumes local TWS/IB Gateway


    def fetch(self, symbol: str, timeframe: str) -> pd.DataFrame:
        contract = util.create_contract(symbol)   # e.g. "AAPL"
        duration = "1 Y"
        barSize = {"M1":"1 min","M5":"5 mins","H1":"1 hour","D1":"1 day"}[timeframe]
        bars = self.ib.reqHistoricalData(contract,
                                         endDateTime='',
                                         durationStr=duration,
                                         barSizeSetting=barSize,
                                         whatToShow='TRADES',
                                         useRTH=False,
                                         formatDate=1)
        df = util.df(bars)
        df["time"] = pd.to_datetime(df["date"])
        return df[["time","open","high","low","close","volume"]]


    def latest_tick(self, symbol: str) -> dict:
        contract = util.create_contract(symbol)
        tick = self.ib.reqMktData(contract, "", False, False)
        return {"price": tick.last, "volume": tick.lastSize, "time": tick.time}
