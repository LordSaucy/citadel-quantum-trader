# tests/unit/test_backtest_sanity.py
import pytest
from datetime import datetime, timedelta

import pandas as pd

# Import the class under test
from src.backtest_validator import BacktestValidator

# -----------------------------------------------------------------
# Helper – a *very* simple dummy strategy that never signals
# -----------------------------------------------------------------
def dummy_no_signal(_data):
    """Always returns None – used for the “no‑signal on first bar” test."""
    return None

# -----------------------------------------------------------------
# Helper – a strategy that incorrectly signals on the first bar
# -----------------------------------------------------------------
def dummy_bad_signal(_data):
    """Incorrectly returns a signal on the very first bar."""
    return {
        "symbol": "EURUSD",
        "direction": "BUY",
        "entry_price": 1.1000,
        "stop_loss": 1.0990,
        "take_profit": 1.1010,
        "volume": 1.0
    }

# -----------------------------------------------------------------
# Fixture – a fresh validator for each test
# -----------------------------------------------------------------
@pytest.fixture
def validator():
    return BacktestValidator(initial_balance=10_000, risk_per_trade=2.0)


# -----------------------------------------------------------------
# 1️⃣  Test that a *single* bar yields no signal (sanity check #1)
# -----------------------------------------------------------------
def test_single_bar_returns_none(validator):
    # Build a 1‑minute dataset (only one row)
    single_bar = pd.DataFrame({
        "time":   [pd.Timestamp(datetime.utcnow())],
        "open":   [1.1000],
        "high":   [1.1010],
        "low":    [1.0990],
        "close":  [1.1005],
        "volume": [0]
    })

    # The validator internally calls `_fetch_historical_data`, but we can
    # bypass that by calling the private sanity checker directly.
    # It should *not* raise an AssertionError because we are only testing
    # the “no‑signal on first bar” rule – the data length check is done
    # later when the real fetch runs.
    validator._sanity_checks(
        symbol="EURUSD",
        timeframe=5,                     # any MT5 timeframe constant
        start_date=datetime.utcnow() - timedelta(minutes=1),
        end_date=datetime.utcnow(),
        strategy_function=dummy_no_signal
    )   # should pass silently (no exception)


# -----------------------------------------------------------------
# 2️⃣  Test that the validator rejects a strategy that returns a signal
#     on the first bar (sanity check #2)
# -----------------------------------------------------------------
def test_strategy_returns_signal_on_first_bar_fails(validator):
    with pytest.raises(AssertionError, match="Strategy returned a signal on the first bar"):
        validator._sanity_checks(
            symbol="EURUSD",
            timeframe=5,
            start_date=datetime.utcnow() - timedelta(minutes=1),
            end_date=datetime.utcnow(),
            strategy_function=dummy_bad_signal
        )


# -----------------------------------------------------------------
# 3️⃣  Test that the *full* `run_validation` raises an error when the
#     sanity check fails (pre‑run validation step)
# -----------------------------------------------------------------
def test_run_validation_fails_on_bad_strategy(validator):
    # Use a realistic date range (>= 2 bars) – the internal fetch will
    # succeed because we are not actually hitting MT5 in the unit test.
    # We monkey‑patch `_fetch_historical_data` to return a tiny DataFrame
    # so the method can continue far enough to hit our sanity‑check result.
    def fake_fetch(*_args, **_kwargs):
        # Return two bars – enough for the engine to iterate
        now = datetime.utcnow()
        return pd.DataFrame({
            "time":   [now - timedelta(minutes=1), now],
            "open":   [1.1000, 1.1005],
            "high":   [1.1010, 1.1015],
            "low":    [1.0990, 1.0995],
            "close":  [1.1005, 1.1010],
            "volume": [0, 0]
        })
    # Patch the private fetch method
    validator._fetch_historical_data = fake_fetch

    result = validator.run_validation(
        symbol="EURUSD",
        timeframe=5,
        start_date=datetime.utcnow() - timedelta(days=1),
        end_date=datetime.utcnow(),
        strategy_function=dummy_bad_signal,   # this will fail the sanity check
        min_win_rate=0.0                     # we don't care about win‑rate here
    )

    assert result["success"] is False
    assert "Strategy returned a signal on the first bar" in result["error"]

