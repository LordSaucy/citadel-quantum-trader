import pytest
from src.risk_management_layer import (
    update_equity_window,
    portfolio_atr,
    current_max_dd_allowed,
)

def test_dynamic_dd_shrinks_when_volatility_spikes():
    # Simulate a calm equity curve (5 % draw‑down allowed)
    for equity in [1000, 1010, 1020, 1030, 1040]:
        update_equity_window(equity)

    assert pytest.approx(current_max_dd_allowed(), rel=1e-2) == 0.20  # base cap

    # Inject a volatility spike (equity swings ±10 %)
    for equity in [950, 1080, 920, 1100, 880]:
        update_equity_window(equity)

    # Now the ATR is roughly double → cap should be halved
    assert current_max_dd_allowed() < 0.12
