# tests/test_dynamic_dd.py
import pytest
from src.risk_management.risk_manager import (
    update_equity_window,
    portfolio_atr,
    current_max_dd_allowed,
)


def test_dynamic_dd_shrinks_when_volatility_spikes():
    """
    The DD (draw‑down) cap is a function of the portfolio ATR.
    When volatility spikes, the ATR doubles and the allowed DD should
    shrink proportionally.
    """

    # -----------------------------------------------------------------
    # 1️⃣  Calm equity curve – 5 % draw‑down allowed (base cap = 0.20)
    # -----------------------------------------------------------------
    for equity in [1000, 1010, 1020, 1030, 1040]:
        update_equity_window(equity)

    # The base cap is 0.20 (20 %).  With a calm ATR the function should
    # return something *very close* to that value.
    assert pytest.approx(current_max_dd_allowed(), rel=1e-2) == 0.20

    # -----------------------------------------------------------------
    # 2️⃣  Inject a volatility spike (equity swings ±10 %)
    # -----------------------------------------------------------------
    for equity in [950, 1080, 920, 1100, 880]:
        update_equity_window(equity)

    # After the spike the ATR roughly doubles, so the DD cap should be
    # roughly halved (< 0.12).  We only need to assert the inequality.
    assert current_max_dd_allowed() < 0.12
