def test_rr_is_used():
    from src.advanced_execution_engine import build_order
    order = build_order(entry_price=100.0, risk_amount=1.0)
    assert order["tp"] == 105.0   # 5 × R
    assert order["sl"] == 99.0    # 1 × R loss
