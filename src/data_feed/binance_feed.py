# src/data_feed/binance_feed.py
from .abstract_feed import DataFeed
from binance import AsyncClient, BinanceSocketManager
import pandas as pd
import asyncio


class BinanceFeed(DataFeed):
    def __init__(self):
        self.client = None


    async def _ensure_client(self):
        if not self.client:
            self.client = await AsyncClient.create()


    async def fetch(self, symbol: str, timeframe: str) -> pd.DataFrame:
        await self._ensure_client()
        klines = await self.client.get_klines(symbol=symbol,
                                             interval=timeframe,
                                             limit=1000)
        df = pd.DataFrame(klines,
                          columns=["open_time","open","high","low","close",
                                   "volume","close_time","quote_av","trades",
                                   "tb_base_av","tb_quote_av","ignore"])
        df["time"] = pd.to_datetime(df["open_time"], unit="ms")
        df = df.astype({"open":"float","high":"float","low":"float",
                        "close":"float","volume":"float"})
        return df[["time","open","high","low","close","volume"]]


    async def latest_tick(self, symbol: str) -> dict:
        await self._ensure_client()
        ticker = await self.client.get_ticker(symbol=symbol)
        return {"price": float(ticker["lastPrice"]),
                "volume": float(ticker["volume"]),
                "time": pd.Timestamp.now()}
