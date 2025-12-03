from broker_interface import BrokerInterface
import requests
import json
import time

class CTraderAdapter(BrokerInterface):
    def __init__(self, cfg):
        self.base_url = cfg["ctrader"]["api_url"]
        self.api_key = cfg["ctrader"]["api_key"]
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

    def connect(self) -> bool:
        # cTrader does not need a persistent socket; just test the endpoint
        r = self.session.get(f"{self.base_url}/ping")
        return r.ok

    def get_price(self, symbol: str, timeframe: str):
        r = self.session.get(
            f"{self.base_url}/prices",
            params={"symbol": symbol, "timeframe": timeframe, "count": 1},
        )
        r.raise_for_status()
        data = r.json()
        # Return a list of dicts matching the shape used elsewhere
        return [{
            "time": int(time.time()),
            "open": float(data["open"]),
            "high": float(data["high"]),
            "low": float(data["low"]),
            "close": float(data["close"]),
            "volume": float(data["volume"]),
        }]

    def send_order(self, symbol, volume, side, price=None, sl=None, tp=None, comment=""):
        payload = {
            "symbol": symbol,
            "volume": volume,
            "side": side,
            "price": price,
            "stopLoss": sl,
            "takeProfit": tp,
            "comment": comment,
        }
        r = self.session.post(f"{self.base_url}/orders", json=payload)
        r.raise_for_status()
        return r.json()   # contains orderId, status, filledQty, etc.

    def get_open_positions(self):
        r = self.session.get(f"{self.base_url}/positions")
        r.raise_for_status()
        return r.json()["positions"]

    def close_position(self, ticket, volume=None):
        payload = {"ticket": ticket, "volume": volume}
        r = self.session.post(f"{self.base_url}/positions/close", json=payload)
        r.raise_for_status()
        return r.json()

    def get_market_depth(self, symbol, depth=20):
        r = self.session.get(
            f"{self.base_url}/orderbook",
            params={"symbol": symbol, "depth": depth},
        )
        r.raise_for_status()
        return r.json()["levels"]
