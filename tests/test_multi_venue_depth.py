import pytest
from citadel.trading.venue_manager import VenueManager
from citadel.trading.config_loader import Config

@pytest.fixture
def fake_cfg(monkeypatch):
    # Patch Config to return a minimal config with two fake venues
    cfg = {
        "venues": [
            {"name": "fake_mt5", "type": "mt5", "vault_path": "dummy"},
            {"name": "fake_ibkr", "type": "ibkr", "vault_path": "dummy"},
        ],
        "min_depth_multiplier": 2.0,
        "contract_notional": 100_000,
    }
    monkeypatch.setattr(Config, "settings", cfg, raising=False)
    return cfg

def test_aggregate_depth(monkeypatch, fake_cfg):
    # Stub the two adapters so they return deterministic depth
    class StubAdapter:
        def __init__(self, *a, **kw): pass
        def get_market_depth(self, symbol, depth=20):
            # 0.5 lot on each side for both venues
            return [
                {"price": 1.0, "volume": 0.5, "side": "bid"},
                {"price": 1.0, "volume": 0.5, "side": "ask"},
            ]

    monkeypatch.setattr("citadel.trading.mt5_adapter.MT5Adapter", StubAdapter)
    monkeypatch.setattr("citadel.trading.ibkr_adapter.IBKRAdapter", StubAdapter)

    vm = VenueManager()
    agg = vm.aggregate_depth("EURUSD")
    assert agg["bid_volume"] == 1.0   # 0.5 + 0.5
    assert agg["ask_volume"] == 1.0

    # Required lot = stake / contract_notional; assume stake = $2000 → lot = 0.02
    required_lot = 0.02
    assert vm.meets_minimum(required_lot, agg)  # 2×0.02 = 0.04 < 1.0 → True
