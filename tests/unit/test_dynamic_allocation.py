"""
test_dynamic_allocation.py

Unit tests for dynamic capital allocation and risk management layer.

Tests validate:
* Aggressive pool growth with profits
* Aggressive pool shrinkage with losses
* Volatility factor application
* Proper floating-point comparisons using pytest.approx()
"""

import pytest
from src.risk_management_layer import RiskManagementLayer, Config
from src.risk_management_layer import engine, bucket_meta


@pytest.fixture(autouse=True)
def clean_db():
    """Truncate tables before each test for isolation."""
    with engine.begin() as conn:
        conn.execute(bucket_meta.delete())
        conn.execute("DELETE FROM trades")
    yield


# =========================================================================
# ✅ FIXED: All floating-point assertions use pytest.approx() correctly
#           and all commented-out code has been removed
# =========================================================================

def test_aggressive_pool_grows_with_profit():
    """
    Verify that the aggressive pool grows when a profitable trade is executed.
    
    Setup:
    * Initial equity: 100.0
    * Initial aggressive pool: 80.0 (with 20% reserve)
    * Trade P&L: +10
    
    Expected:
    * Usable capital = 90.0 (original 80 + 10 profit)
    """
    rml = RiskManagementLayer()
    bucket_id = 1
    start_eq = 100.0
    
    # ✅ FIXED: Removed commented-out code
    rml._load_pools(bucket_id)
    
    # Simulate a winning trade (+10)
    equity_after = start_eq + 10
    usable = rml.calculate_usable_capital(bucket_id, equity_after)
    
    # ✅ FIXED: Correct pytest.approx() syntax - value on LEFT, approx() on RIGHT
    assert usable == pytest.approx(90.0, rel=1e-3)


def test_aggressive_pool_shrinks_on_loss():
    """
    Verify that the aggressive pool shrinks when a losing trade is executed.
    
    Setup:
    * Initial equity: 100.0
    * Initial aggressive pool: 80.0 (with 20% reserve)
    * Trade P&L: -15
    
    Expected:
    * Usable capital = 65.0 (original 80 - 15 loss, cannot go below 0)
    """
    rml = RiskManagementLayer()
    bucket_id = 2
    
    # ✅ FIXED: Removed commented-out code
    rml._load_pools(bucket_id)
    
    # Simulate a losing trade (-15)
    equity_after = 100.0 - 15
    usable = rml.calculate_usable_capital(bucket_id, equity_after)
    
    # ✅ FIXED: Correct pytest.approx() syntax - value on LEFT, approx() on RIGHT
    assert usable == pytest.approx(65.0, rel=1e-3)


def test_volatility_factor_applied():
    """
    Verify that the volatility risk factor is correctly applied to usable capital.
    
    Setup:
    * Initial equity: 100.0 (no P&L change)
    * Initial aggressive pool: 80.0
    * Volatility risk factor: 0.5 (halve usable capital)
    
    Expected:
    * Usable capital = 40.0 (80 * 0.5)
    """
    rml = RiskManagementLayer()
    bucket_id = 3
    
    # ✅ FIXED: Removed commented-out code
    rml._load_pools(bucket_id)
    
    # Set a custom factor via config
    rml.cfg["volatility_risk_factor"] = 0.5
    
    equity_after = 100.0
    usable = rml.calculate_usable_capital(bucket_id, equity_after)
    
    # ✅ FIXED: Correct pytest.approx() syntax - value on LEFT, approx() on RIGHT
    assert usable == pytest.approx(40.0, rel=1e-3)
