# tests/test_risk_management.py
import datetime
from unittest import mock

import pytest

from risk_management import RiskManagementLayer
from src.risk_management_layer import compute_stake, get_trade_counter


# ----------------------------------------------------------------------
# Helper: a fake MT5/IBKR price feed (returns a deterministic price)
# ----------------------------------------------------------------------
def fake_price_feed(*_, **__):
    # Return a flat price series – perfect for deterministic risk checks
    return 1.0800


@pytest.fixture
def rm_layer():
    """Instantiate the layer with a tiny config for testing."""
    cfg = {
        "daily_drawdown_limit_pct": 3.0,
        "weekly_drawdown_limit_pct": 8.0,
        "max_open_positions": 4,
        "monitor_interval_seconds": 0.1,  # fast loop for the test
    }
    return RiskManagementLayer(cfg)


def test_initial_state(rm_layer):
    """A fresh instance should allow trading."""
    can_trade, reason = rm_layer.can_take_new_trade()
    assert can_trade is True
    assert reason == ""


def test_position_limit(rm_layer):
    """When the number of open positions reaches the limit, trading must stop."""
    # Pretend we already have 4 open positions
    with mock.patch("risk_management.mt5.positions_total", return_value=4):
        can_trade, reason = rm_layer.can_take_new_trade()
        assert can_trade is False
        assert "Maximum open positions reached" in reason


def test_daily_drawdown_trigger(rm_layer):
    """A daily draw‑down > limit should activate the kill‑switch."""
    # Mock the account equity to be 5 % below a fabricated high‑water mark
    mock_account = mock.Mock()
    mock_account.equity = 95.0
    mock_account.balance = 100.0

    with mock.patch("risk_management.mt5.account_info", return_value=mock_account):
        # Set a high‑water mark artificially high
        rm_layer.daily_high_water_mark = 100.0
        # Run the monitoring loop once (fast interval)
        rm_layer._check_drawdown_once()  # internal helper we expose for testing
        assert rm_layer.kill_switch_active is True
        assert "Daily draw‑down" in rm_layer.kill_switch_reason


def test_kill_switch_cooldown(rm_layer):
    """After a kill‑switch fires, it stays active for the configured cooldown."""
    # Force activation
    rm_layer.kill_switch_active = True
    rm_layer.kill_switch_reason = "Test reason"
    rm_layer.unlock_time = datetime.datetime.now() + datetime.timedelta(seconds=30)

    # Call the routine that would normally clear it (if time passed)
    rm_layer._maybe_clear_kill_switch()
    # Still active because unlock_time not reached
    assert rm_layer.kill_switch_active is True

    # Fast‑forward time
    with mock.patch("risk_management.datetime") as dt_mock:
        dt_mock.utcnow.return_value = rm_layer.unlock_time + datetime.timedelta(seconds=1)
        rm_layer._maybe_clear_kill_switch()
        assert rm_layer.kill_switch_active is False
        assert rm_layer.kill_switch_reason == ""

def test_compute_stake_basic(monkeypatch):
    # Mock DB calls – we only need the trade index
    monkeypatch.setattr("src.risk_management_layer.get_trade_counter", lambda _: 3)
    # Fake config – risk schedule from config.yaml
    from config_loader import Config
    cfg = Config().settings
    cfg["risk_schedule"] = {1: 1.0, 2: 1.0, 3: 0.6, "default": 0.4}
    # Patch the global config used inside the module
    monkeypatch.setitem("src.risk_management_layer.cfg", cfg, raising=False)

    equity = 10_000.0
    stake = compute_stake(bucket_id=1, equity=equity)
    # Trade index = 4 (3 previous + 1) → default 0.4
    assert pytest.approx(stake, rel=1e-3) == equity * 0.4
