# tests/test_confluence_system.py
import datetime
from unittest import mock

import pytest

from ultimate_confluence_system import ultimate_system


# ----------------------------------------------------------------------
# Helper: a deterministic price series (flat + small noise)
# ----------------------------------------------------------------------
def make_price_series(base=1.0800, length=200, noise=0.0001):
    """Return a list of (timestamp, price) tuples."""
    now = datetime.datetime.utcnow()
    series = []
    for i in range(length):
        ts = now - datetime.timedelta(minutes=5 * i)
        price = base + ((i % 5) - 2) * noise  # tiny wiggle
        series.append((ts, price))
    return list(reversed(series))


@pytest.fixture
def mock_mt5_rates():
    """Patch mt5.copy_rates_from_pos to return a synthetic series."""
    series = make_price_series()
    # Convert to the dict format that the original code expects
    rates = [
        {
            "time": int(dt.timestamp()),
            "open": p,
            "high": p,
            "low": p,
            "close": p,
            "tick_volume": 100,
            "spread": 0,
            "real_volume": 0,
        }
        for dt, p in series
    ]

    with mock.patch("ultimate_confluence_system.mt5.copy_rates_from_pos", return_value=rates):
        yield


def test_aoi_detection_success(mock_mt5_rates):
    """With a clean flat series the AOI detector should find a zone."""
    result = ultimate_system.analyze_complete(
        symbol="EURUSD",
        direction="BUY",
        entry_price=1.0800,
    )
    # The AOI should be found because the synthetic series has many
    # identical lows (or highs) that satisfy the 3‑touch rule.
    assert result["aoi_found"] is True
    assert result["aoi"]["price"] == pytest.approx(1.0800, rel=1e-4)
    assert result["aoi"]["touches"] >= 3
    # Since the price equals the AOI price, we are “at AOI”
    assert result["aoi"]["at_aoi"] is True


def test_confluence_weights_affect_score(mock_mt5_rates):
    """Changing a weight should linearly affect the total score."""
    # First run with default weights (all 0.2)
    default = ultimate_system.analyze_complete(
        symbol="EURUSD",
        direction="BUY",
        entry_price=1.0800,
    )
    default_score = default["total_score"]

    # Now temporarily monkey‑patch the weight for MTF structure to 0.5
    with mock.patch.object(ultimate_system, "WEIGHTS", {"mtf_structure": 0.5, "aoi": 0.2,
                                                       "candlestick": 0.1, "smc": 0.2,
                                                       "head_shoulders": 0.2, "traditional_ta": 0.0}):
        boosted = ultimate_system.analyze_complete(
            symbol="EURUSD",
            direction="BUY",
            entry_price=1.0800,
        )
    boosted_score = boosted["total_score"]

    # The boosted score must be higher because the MTF weight grew from 0.2 → 0.5
    assert boosted_score > default_score
    # The difference should be roughly proportional to the weight delta
    delta_expected = (0.5 - 0.2) * default["mtf_structure"]["score"]
    assert boosted_score - default_score == pytest.approx(delta_expected, rel=0.05)
