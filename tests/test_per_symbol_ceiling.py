# tests/test_per_symbol_ceiling.py
import pytest
from src.risk_management.risk_manager import RiskManagementLayer


@pytest.fixture
def rm(mock_db):
    """
    ``mock_db`` is a lightweight in‑memory stub that mimics the DB API
    used by ``RiskManagementLayer``.  The fixture injects a minimal config
    needed for the ceiling calculations.
    """
    cfg = {
        "aggressive_pool_fraction": 0.30,   # 30 % of total equity is aggressive pool
        "symbol_exposure_ceiling": 0.12,    # 12 % of aggressive pool per symbol
    }
    return RiskManagementLayer(mock_db, cfg)


def test_symbol_ceiling_blocks_excessive_exposure(rm):
    equity = 1000.0
    bucket_id = 1
    symbol = "EURUSD"

    # First trade – 0.5 % risk → $5 stake
    stake1 = rm.allocate_for_trade(bucket_id, symbol, equity, 0.005)
    # Use approx with a very tight tolerance (1 e‑6) because the value is exact
    assert pytest.approx(stake1, rel=1e-6) == 5.0

    # Simulate many winning trades that push exposure close to the ceiling
    # aggressive_pool ≈ 0.30 * total_equity (here total_equity = 2 * equity)
    #   = 0.30 * 2000 = 600
    # ceiling = 0.12 * 600 = 72
    # already used ≈ 0.11 * 600 = 66 → only $6 left
    rm.symbol_exposure[symbol] = 0.11 * rm.aggressive_pool

    # Next trade should be trimmed so total ≤ 12 %
    stake2 = rm.allocate_for_trade(bucket_id, symbol, equity, 0.005)

    # Allow a relative tolerance of 1 % (1e‑2) – the exact value is $6.00
    assert pytest.approx(stake2, rel=1e-2) == 6.0
