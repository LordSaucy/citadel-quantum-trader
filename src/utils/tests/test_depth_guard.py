#!/usr/bin/env python3
import pandas as pd
import pytest

from ..depth_guard import depth_guard, price_to_lots


@pytest.fixture
def dummy_dom():
    """A tiny DOM DataFrame with a few price levels."""
    return pd.DataFrame({
        "price":       [1.0798, 1.0800, 1.0802],
        "bid_volume":  [200_000, 150_000, 100_000],   # units (e.g., lots*100k)
        "ask_volume":  [180_000, 120_000,  80_000],
    })


def test_long_trade_passes_when_enough_ask_volume(dummy_dom):
    entry_price = 1.0800
    stake_usd = 500.0                     # we risk $500
    assert depth_guard(entry_price, dummy_dom, stake_usd) is True


def test_long_trade_fails_when_not_enough_ask_volume(dummy_dom):
    entry_price = 1.0802                 # nearest price = 1.0802, ask = 80k
    stake_usd = 500.0
    # With default safety_multiplier=2, we need 2× the lots needed.
    # 500 $ at 1.0802 ≈ 0.46 lots → 0.46 * 100 000 ≈ 46 000 units,
    # but ask volume is only 80 000 → still enough? 2×46k = 92k > 80k → FAIL
    assert depth_guard(entry_price, dummy_dom, stake_usd) is False


def test_short_trade_uses_bid_volume(dummy_dom):
    entry_price = 1.0800
    stake_usd = -300.0                    # negative → short
    # 300 $ risk → ~0.28 lots → 28 000 units needed.
    # bid at 1.0800 = 150 000 → 2×28k = 56k < 150k → PASS
    assert depth_guard(entry_price, dummy_dom, stake_usd) is True


def test_price_to_lots_basic():
    # 1 pip = 0.0001, lot size = 100 000, entry 1.0800, risk $500
    lots = price_to_lots(1.0800, 500.0)
    # Roughly: 500 / (0.0001 * 100_000 / 1.08) ≈ 0.054 lots
    assert pytest.approx(lots, rel=1e-2) == 0.054
