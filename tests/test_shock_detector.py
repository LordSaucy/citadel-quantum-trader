# tests/unit/test_shock_detector.py
import time
import redis
import pytest
from shock_detector import should_block_trade, cfg

@pytest.fixture(autouse=True)
def reset_redis(monkeypatch):
    # Use a fresh Redis instance (or a mock) for each test
    client = redis.StrictRedis(host="localhost", port=6379, db=15, decode_responses=True)
    client.flushdb()
    monkeypatch.setattr('shock_detector.r', client)

def test_news_sentiment_block(monkeypatch):
    # Arrange
    r = shock_detector.r
    r.set("sentiment:latest", "-0.9")   # extreme bearish
    cfg["risk_shocks"]["sentiment"]["low"] = -0.5

    # Act
    blocked, reason = should_block_trade("EURUSD")

    # Assert
    assert blocked is True
    assert reason == "news_sentiment"

def test_volatility_spike_block(monkeypatch):
    r = shock_detector.r
    r.set("atr:EURUSD", "0.03")
    r.set("atr_ema30:EURUSD", "0.008")
    cfg["risk_shocks"]["atr_spike_multiplier"] = 2.0

    blocked, reason = should_block_trade("EURUSD")
    assert blocked is True
    assert reason == "vol_spike"

def test_no_block_when_all_clear(monkeypatch):
    # No keys set → everything should be clear
    blocked, reason = should_block_trade("EURUSD")
    assert blocked is False
    assert reason == ""
