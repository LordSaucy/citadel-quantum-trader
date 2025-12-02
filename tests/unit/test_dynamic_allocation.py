import pytest
from src.risk_management_layer import RiskManagementLayer, Config
from src.risk_management_layer import engine, bucket_meta

@pytest.fixture(autouse=True)
def clean_db():
    # truncate tables before each test
    with engine.begin() as conn:
        conn.execute(bucket_meta.delete())
        conn.execute("DELETE FROM trades")
    yield

def test_aggressive_pool_grows_with_profit():
    rml = RiskManagementLayer()
    bucket_id = 1
    start_eq = 100.0
    # initialise meta row (reserve 20%)
    rml._load_pools(bucket_id)   # creates row with aggressive=80, reserve=20

    # Simulate a winning trade (+10)
    equity_after = start_eq + 10
    usable = rml.calculate_usable_capital(bucket_id, equity_after)

    # aggressive pool should have increased by the profit (80 + 10 = 90)
    assert pytest.approx(usable, rel=1e-3) == 90.0

def test_aggressive_pool_shrinks_on_loss():
    rml = RiskManagementLayer()
    bucket_id = 2
    rml._load_pools(bucket_id)   # aggressive=80, reserve=20

    # Simulate a losing trade (-15)
    equity_after = 100.0 - 15
    usable = rml.calculate_usable_capital(bucket_id, equity_after)

    # aggressive pool should be 80 - 15 = 65 (cannot go below 0)
    assert pytest.approx(usable, rel=1e-3) == 65.0

def test_volatility_factor_applied():
    rml = RiskManagementLayer()
    bucket_id = 3
    rml._load_pools(bucket_id)   # aggressive=80
    # set a custom factor via config
    rml.cfg["volatility_risk_factor"] = 0.5   # halve usable capital

    equity_after = 100.0   # no P&L change
    usable = rml.calculate_usable_capital(bucket_id, equity_after)

    # aggressive pool (80) * 0.5 = 40
    assert pytest.approx(usable, rel=1e-3) == 40.0
