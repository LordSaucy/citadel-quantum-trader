# src/data_feed/abstract_feed.py
from abc import ABC, abstractmethod
import pandas as pd


class DataFeed(ABC):
    """Common contract for all market data providers."""


    @abstractmethod
    def fetch(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Return a DataFrame with at least ['time','open','high','low','close','volume']."""
        ...


    @abstractmethod
    def latest_tick(self, symbol: str) -> dict:
        """Return the most recent tick dict (price, volume, timestamp)."""
        ...
