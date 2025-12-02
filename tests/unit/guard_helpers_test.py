import pytest
from src.guard_helpers import (
    check_depth,
    check_latency,
    check_spread,
    check_volatility,
)

class DummyBroker:
    def __init__(self, depth=None, latency=0.05, spread=0.3, price=1.0):
        self._depth = depth or []
        self._latency = latency
        self._spread = spread
        self._price = price

    def ping(self):
        pass  # always succeeds

    def get_market_depth(self, symbol, depth=20):
        return self._depth

    def get_quote(self, symbol):
        return {"bid": self._price - self._spread/2,
                "ask": self._price + self._spread/2}

    def get_price(self, symbol):
        return self._price

class DummyTechCalc:
    def __init__(self, atr, avg_atr, price):
        self._atr = atr
        self._avg_atr = avg_atr
        self._price = price

    def get_atr(self, symbol, lookback=14):
        return self._atr

    def get_atr_average(self, symbol, lookback=100):
        return self._avg_atr

    def get_price(self, symbol):
        return self._price

def test_depth_guard_pass():
    broker = DummyBroker(depth=[
        {"side":"bid","price":1.0,"bid_volume":200,"ask_volume":0},
        {"side":"ask","price":1.0,"bid_volume":0,"ask_volume":200},
    ])
    assert check_depth(broker, "EURUSD", required_volume=100, min_lir=0.5)

def test_depth_guard_fail_low_lir():
    broker = DummyBroker(depth=[
        {"side":"bid","price":1.0,"bid_volume":10,"ask_volume":0},
        {"side":"ask","price":1.0,"bid_volume":0,"ask_volume":90},
    ])
    assert not check_depth(broker, "EURUSD", required_volume=50, min_lir=0.5)

def test_latency_guard_pass():
    broker = DummyBroker(latency=0.08)
    assert check_latency(broker, max_latency_sec=0.15)

def test_latency_guard_fail():
    broker = DummyBroker(latency=0.25)
    assert not check_latency(broker, max_latency_sec=0.15)

def test_spread_guard_pass():
    broker = DummyBroker(spread=0.3)   # 0.3 pips
    assert check_spread(broker, "EURUSD", max_spread_pips=0.5)

def test_spread_guard_fail():
    broker = DummyBroker(spread=0.8)   # 0.8 pips
    assert not check_spread(broker, "EURUSD", max_spread_pips=0.5)

def test_volatility_guard_pass():
    tech = DummyTechCalc(atr=0.001, avg_atr=0.001, price=1.0)
    assert check_volatility(tech, "EURUSD", max_atr_pct=0.20)

def test_volatility_guard_fail():
    tech = DummyTechCalc(atr=0.003, avg_atr=0.001, price=1.0)  # 200 % spike
    assert not check_volatility(tech, "EURUSD", max_atr_pct=0.20)
