import pytest
import redis
from src.guards import SentimentGuard, CalendarLockout, VolatilityGuard, ShockDetector
from datetime import datetime, timedelta
import pytz

@pytest.fixture
def redis_client(monkeypatch):
    # Simple in‑memory mock (you can use fakeredis if you prefer)
    class DummyRedis:
        _store = {}
        def get(self, key):
            return self._store.get(key)
        def set(self, key, val):
            self._store[key] = val
    return DummyRedis()

def test_sentiment_guard_pass(redis_client):
    cfg = {"sentiment": {"guard_enabled": True,
                        "positive_threshold": 0.3,
                        "negative_threshold": -0.3}}
    guard = SentimentGuard(cfg, redis_client)
    redis_client.set("sentiment:latest", b"0.10")
    assert guard.check() is True

def test_sentiment_guard_reject_positive(redis_client):
    cfg = {"sentiment": {"guard_enabled": True,
                        "positive_threshold": 0.3,
                        "negative_threshold": -0.3}}
    guard = SentimentGuard(cfg, redis_client)
    redis_client.set("sentiment:latest", b"0.45")
    assert guard.check() is False

def test_calendar_lockout(monkeypatch):
    # Mock economic_calendar module
    class DummyCal:
        @staticmethod
        def get_upcoming_events():
            now = datetime.now().replace(tzinfo=pytz.UTC)
            return [{
                "title": "CPI Release",
                "start_utc": now - timedelta(minutes=10),
                "end_utc": now + timedelta(minutes=10),
                "importance": "high"
            }]
    cfg = {"calendar": {"lockout_enabled": True,
                       "lockout_margin_minutes": 30}}
    lock = CalendarLockout(cfg, DummyCal)
    assert lock.is_locked() is True

def test_volatility_guard():
    cfg = {"volatility_spike": {"enabled": True,
                                "atr_multiplier": 2.0,
                                "baseline_atr": 0.001}}
    guard = VolatilityGuard(cfg)
    assert guard.check(0.0015) is True          # below 2× baseline
    assert guard.check(0.003) is False          # above 2× baseline

def test_shock_detector():
    cfg = {"shock_detector": {"enabled": True,
                              "spread_multiplier": 3.0,
                              "max_tick_age_secs": 2,
                              "lir_threshold": 0.6,
                              "min_depth": 0.02}}
    detector = ShockDetector(cfg)

    # 1️⃣ Spread explosion
    snap = {"bid": 1.0000, "ask": 1.0100,
            "spread_history": [0.001, 0.0012, 0.0011],
            "timestamp": datetime.now().replace(tzinfo=pytz.UTC),
            "bid_volume": 0.05, "ask_volume": 0.05,
            "depth": 0.03}
    assert detector.check(snap) is False
