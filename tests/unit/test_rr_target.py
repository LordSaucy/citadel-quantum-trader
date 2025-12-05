import pytest

def test_rr_is_used():
    from src.advanced_execution_engine import build_order
    order = build_order(entry_price=100.0, risk_amount=1.0)

    # 5×R target profit
    assert order["tp"] == pytest.approx(105.0)

    # 1×R stop‑loss
    assert order["sl"] == pytest.approx(99.0)
