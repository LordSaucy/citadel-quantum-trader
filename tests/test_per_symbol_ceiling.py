import pytest
from src.risk_management_layer import RiskManager

@pytest.fixture
def rm(mock_db):
    # mock_db is a simple in‑memory stub that mimics the DB API used by RiskManager
    cfg = {
        "aggressive_pool_fraction": 0.30,
        "symbol_exposure_ceiling": 0.12,
    }
    return RiskManager(mock_db, cfg)

def test_symbol_ceiling_blocks_excessive_exposure(rm):
    equity = 1000.0
    bucket_id = 1
    symbol = "EURUSD"
    # First trade – 0.5 % risk → $5 stake
    stake1 = rm.allocate_for_trade(bucket_id, symbol, equity, 0.005)
    assert stake1 == 5.0

    # Simulate many winning trades that push exposure close to the ceiling
    rm.symbol_exposure[symbol] = 0.11 * rm.aggressive_pool   # 11 % used

    # Next trade should be trimmed so total ≤ 12 %
    stake2 = rm.allocate_for_trade(bucket_id, symbol, equity, 0.005)
    # aggressive_pool ≈ 0.30 * total_equity = 0.30 * 2000 = 600
    # ceiling = 0.12 * 600 = 72
    # already used ≈ 66 → only $6 left
    assert pytest.approx(stake2, rel=1e-2) == 6.0
