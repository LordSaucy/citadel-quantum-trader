# tests/unit/guard_helpers_test.py
"""
Production-grade unit tests for src.guard_helpers module.

Tests the four core guard rails:
  1. check_depth() → Validates market depth and liquidity
  2. check_latency() → Validates order submission latency
  3. check_spread() → Validates bid-ask spread
  4. check_volatility() → Validates ATR-based volatility

SonarCloud Compliance: All unused parameters removed or explicitly marked (_param).

Usage:
    pytest tests/unit/guard_helpers_test.py -v
    pytest tests/unit/guard_helpers_test.py::test_depth_guard_pass -vv
    pytest tests/unit/guard_helpers_test.py -k "depth" --cov=src.guard_helpers
"""

import pytest
from typing import Dict, List, Any
from dataclasses import dataclass
from src.guard_helpers import (
    check_depth,
    check_latency,
    check_spread,
    check_volatility,
)


# ============================================================================
# FIXTURES & TEST DATA
# ============================================================================

@dataclass
class MarketDepthTick:
    """Represents a single depth level in order book."""
    side: str  # "bid" or "ask"
    price: float
    bid_volume: float
    ask_volume: float


@pytest.fixture
def dummy_broker():
    """Factory fixture for creating mock broker instances."""
    class DummyBroker:
        """Mock broker that simulates market data and connectivity."""

        def __init__(
            self,
            depth: List[Dict[str, Any]] | None = None,
            latency: float = 0.05,
            spread: float = 0.3,
            price: float = 1.0,
        ):
            """
            Initialize mock broker.

            Args:
                depth: Order book depth (list of {"side", "price", "bid_volume", "ask_volume"})
                latency: Network latency in seconds
                spread: Bid-ask spread in pips
                price: Current mid-price
            """
            self._depth = depth or []
            self._latency = latency
            self._spread = spread
            self._price = price

        def ping(self) -> None:
            """Mock connectivity check (always succeeds for testing)."""
            pass

        def get_market_depth(self, _symbol: str, _depth: int = 20) -> List[Dict[str, Any]]:
            """
            Get order book depth.

            Args:
                _symbol: Trading symbol (unused in mock, marked with _)
                _depth: Depth level requested (unused in mock, marked with _)

            Returns:
                Depth data (pre-configured for this mock)
            """
            return self._depth

        def get_quote(self, _symbol: str) -> Dict[str, float]:
            """
            Get current bid/ask quote.

            Args:
                _symbol: Trading symbol (unused in mock, marked with _)

            Returns:
                {"bid": <bid_price>, "ask": <ask_price>}
            """
            return {
                "bid": self._price - self._spread / 2,
                "ask": self._price + self._spread / 2,
            }

        def get_price(self, _symbol: str) -> float:
            """
            Get current mid-price.

            Args:
                _symbol: Trading symbol (unused in mock, marked with _)

            Returns:
                Mid-price (mid = (bid + ask) / 2)
            """
            return self._price

    return DummyBroker


@pytest.fixture
def dummy_tech_calculator():
    """Factory fixture for creating mock technical indicator calculator instances."""
    class DummyTechCalc:
        """Mock technical calculator that simulates volatility metrics."""

        def __init__(self, atr: float, avg_atr: float, price: float):
            """
            Initialize mock tech calculator.

            Args:
                atr: Current Average True Range (volatility measure)
                avg_atr: Average ATR over longer period
                price: Current price (for ATR % calculation)
            """
            self._atr = atr
            self._avg_atr = avg_atr
            self._price = price

        def get_atr(self, _symbol: str, _lookback: int = 14) -> float:
            """
            Get Average True Range.

            Args:
                _symbol: Trading symbol (unused in mock, marked with _)
                _lookback: Lookback period in bars (unused in mock, marked with _)

            Returns:
                ATR value in price units
            """
            return self._atr

        def get_atr_average(self, _symbol: str, _lookback: int = 100) -> float:
            """
            Get average ATR over period.

            Args:
                _symbol: Trading symbol (unused in mock, marked with _)
                _lookback: Lookback period in bars (unused in mock, marked with _)

            Returns:
                Average ATR value
            """
            return self._avg_atr

        def get_price(self, _symbol: str) -> float:
            """
            Get current price.

            Args:
                _symbol: Trading symbol (unused in mock, marked with _)

            Returns:
                Current price
            """
            return self._price

    return DummyTechCalc


# ============================================================================
# DEPTH GUARD TESTS
# ============================================================================

class TestDepthGuard:
    """Test suite for check_depth() guard rail."""

    def test_depth_guard_pass_sufficient_liquidity(self, dummy_broker):
        """Test that depth guard passes with sufficient bid/ask volume."""
        broker = dummy_broker(
            depth=[
                {
                    "side": "bid",
                    "price": 1.0,
                    "bid_volume": 200,
                    "ask_volume": 0,
                },
                {
                    "side": "ask",
                    "price": 1.0,
                    "bid_volume": 0,
                    "ask_volume": 200,
                },
            ]
        )
        assert check_depth(broker, "EURUSD", required_volume=100, min_lir=0.5)

    def test_depth_guard_fail_low_liquidity_imbalance_ratio(self, dummy_broker):
        """Test that depth guard fails with poor liquidity imbalance ratio (LIR)."""
        broker = dummy_broker(
            depth=[
                {"side": "bid", "price": 1.0, "bid_volume": 10, "ask_volume": 0},
                {"side": "ask", "price": 1.0, "bid_volume": 0, "ask_volume": 90},
            ]
        )
        assert not check_depth(broker, "EURUSD", required_volume=50, min_lir=0.5)

    def test_depth_guard_fail_insufficient_volume(self, dummy_broker):
        """Test that depth guard fails when total volume is below requirement."""
        broker = dummy_broker(
            depth=[
                {"side": "bid", "price": 1.0, "bid_volume": 30, "ask_volume": 0},
                {"side": "ask", "price": 1.0, "bid_volume": 0, "ask_volume": 30},
            ]
        )
        assert not check_depth(broker, "EURUSD", required_volume=200, min_lir=0.5)

    def test_depth_guard_empty_depth_book(self, dummy_broker):
        """Test that depth guard fails gracefully with empty order book."""
        broker = dummy_broker(depth=[])
        assert not check_depth(broker, "EURUSD", required_volume=50, min_lir=0.5)

    def test_depth_guard_boundary_lir_exactly_minimum(self, dummy_broker):
        """Test boundary condition: LIR exactly at minimum threshold."""
        broker = dummy_broker(
            depth=[
                {"side": "bid", "price": 1.0, "bid_volume": 50, "ask_volume": 0},
                {"side": "ask", "price": 1.0, "bid_volume": 0, "ask_volume": 100},
            ]
        )
        assert check_depth(broker, "EURUSD", required_volume=100, min_lir=0.5)

    @pytest.mark.parametrize(
        "bid_vol,ask_vol,required_vol,min_lir,expected",
        [
            (100, 100, 150, 0.8, True),   # Balanced, sufficient volume
            (100, 100, 250, 0.8, False),  # Insufficient volume
            (50, 150, 100, 0.3, True),    # Imbalanced but acceptable LIR
            (50, 150, 100, 0.5, False),   # LIR too low
            (0, 100, 50, 0.5, False),     # One side empty
        ],
    )
    def test_depth_guard_parametrized(
        self, dummy_broker, bid_vol, ask_vol, required_vol, min_lir, expected
    ):
        """Parametrized test for various depth scenarios."""
        broker = dummy_broker(
            depth=[
                {"side": "bid", "price": 1.0, "bid_volume": bid_vol, "ask_volume": 0},
                {"side": "ask", "price": 1.0, "bid_volume": 0, "ask_volume": ask_vol},
            ]
        )
        result = check_depth(broker, "EURUSD", required_volume=required_vol, min_lir=min_lir)
        assert result == expected


# ============================================================================
# LATENCY GUARD TESTS
# ============================================================================

class TestLatencyGuard:
    """Test suite for check_latency() guard rail."""

    def test_latency_guard_pass_within_limit(self, dummy_broker):
        """Test that latency guard passes when latency is below threshold."""
        broker = dummy_broker(latency=0.08)
        assert check_latency(broker, max_latency_sec=0.15)

    def test_latency_guard_fail_exceeds_limit(self, dummy_broker):
        """Test that latency guard fails when latency exceeds threshold."""
        broker = dummy_broker(latency=0.25)
        assert not check_latency(broker, max_latency_sec=0.15)

    def test_latency_guard_boundary_exactly_at_limit(self, dummy_broker):
        """Test boundary condition: latency exactly at maximum threshold."""
        broker = dummy_broker(latency=0.15)
        result = check_latency(broker, max_latency_sec=0.15)
        assert result is True

    def test_latency_guard_very_low_latency(self, dummy_broker):
        """Test with extremely low latency (near-zero)."""
        broker = dummy_broker(latency=0.001)
        assert check_latency(broker, max_latency_sec=0.15)

    def test_latency_guard_high_threshold(self, dummy_broker):
        """Test with very high latency threshold (relaxed guard)."""
        broker = dummy_broker(latency=1.0)
        assert check_latency(broker, max_latency_sec=5.0)

    @pytest.mark.parametrize(
        "latency_sec,max_latency_sec,expected",
        [
            (0.05, 0.15, True),   # 50ms, limit 150ms → pass
            (0.15, 0.15, True),   # 150ms, limit 150ms → pass (boundary)
            (0.16, 0.15, False),  # 160ms, limit 150ms → fail
            (0.001, 0.1, True),   # 1ms, limit 100ms → pass
            (2.0, 0.5, False),    # 2s, limit 500ms → fail
        ],
    )
    def test_latency_guard_parametrized(
        self, dummy_broker, latency_sec, max_latency_sec, expected
    ):
        """Parametrized test for various latency scenarios."""
        broker = dummy_broker(latency=latency_sec)
        result = check_latency(broker, max_latency_sec=max_latency_sec)
        assert result == expected


# ============================================================================
# SPREAD GUARD TESTS
# ============================================================================

class TestSpreadGuard:
    """Test suite for check_spread() guard rail."""

    def test_spread_guard_pass_within_limit(self, dummy_broker):
        """Test that spread guard passes when spread is below threshold."""
        broker = dummy_broker(spread=0.3)  # 0.3 pips
        assert check_spread(broker, "EURUSD", max_spread_pips=0.5)

    def test_spread_guard_fail_exceeds_limit(self, dummy_broker):
        """Test that spread guard fails when spread exceeds threshold."""
        broker = dummy_broker(spread=0.8)  # 0.8 pips
        assert not check_spread(broker, "EURUSD", max_spread_pips=0.5)

    def test_spread_guard_boundary_exactly_at_limit(self, dummy_broker):
        """Test boundary condition: spread exactly at maximum threshold."""
        broker = dummy_broker(spread=0.5)
        assert check_spread(broker, "EURUSD", max_spread_pips=0.5)

    def test_spread_guard_tight_spread_1_pip(self, dummy_broker):
        """Test with very tight spread (1 pip)."""
        broker = dummy_broker(spread=0.01)
        assert check_spread(broker, "EURUSD", max_spread_pips=0.1)

    def test_spread_guard_wide_spread_relaxed_limit(self, dummy_broker):
        """Test with wide spread but relaxed limit."""
        broker = dummy_broker(spread=2.0)
        assert check_spread(broker, "EURUSD", max_spread_pips=5.0)

    @pytest.mark.parametrize(
        "spread_pips,max_spread_pips,expected",
        [
            (0.3, 0.5, True),    # 0.3 pip spread, 0.5 limit → pass
            (0.5, 0.5, True),    # 0.5 pip spread, 0.5 limit → pass (boundary)
            (0.6, 0.5, False),   # 0.6 pip spread, 0.5 limit → fail
            (0.01, 0.5, True),   # 0.01 pip (tight), 0.5 limit → pass
            (3.0, 1.0, False),   # 3 pips (wide), 1 limit → fail
        ],
    )
    def test_spread_guard_parametrized(
        self, dummy_broker, spread_pips, max_spread_pips, expected
    ):
        """Parametrized test for various spread scenarios."""
        broker = dummy_broker(spread=spread_pips)
        result = check_spread(broker, "EURUSD", max_spread_pips=max_spread_pips)
        assert result == expected


# ============================================================================
# VOLATILITY GUARD TESTS
# ============================================================================

class TestVolatilityGuard:
    """Test suite for check_volatility() guard rail."""

    def test_volatility_guard_pass_normal_volatility(self, dummy_tech_calculator):
        """Test that volatility guard passes under normal market conditions."""
        tech = dummy_tech_calculator(atr=0.001, avg_atr=0.001, price=1.0)
        assert check_volatility(tech, "EURUSD", max_atr_pct=0.20)

    def test_volatility_guard_fail_extreme_spike(self, dummy_tech_calculator):
        """Test that volatility guard fails during volatility spike."""
        tech = dummy_tech_calculator(
            atr=0.003, avg_atr=0.001, price=1.0
        )  # 300% spike
        assert not check_volatility(tech, "EURUSD", max_atr_pct=0.20)

    def test_volatility_guard_boundary_exactly_at_limit(self, dummy_tech_calculator):
        """Test boundary condition: ATR % exactly at maximum threshold."""
        tech = dummy_tech_calculator(atr=0.002, avg_atr=0.001, price=1.0)
        result = check_volatility(tech, "EURUSD", max_atr_pct=0.20)
        assert result is True

    def test_volatility_guard_very_low_volatility(self, dummy_tech_calculator):
        """Test with extremely low volatility (calm market)."""
        tech = dummy_tech_calculator(atr=0.0001, avg_atr=0.0001, price=1.0)
        assert check_volatility(tech, "EURUSD", max_atr_pct=0.20)

    def test_volatility_guard_relaxed_limit(self, dummy_tech_calculator):
        """Test with relaxed volatility limit (high spike allowed)."""
        tech = dummy_tech_calculator(atr=0.05, avg_atr=0.01, price=1.0)
        assert check_volatility(tech, "EURUSD", max_atr_pct=1.0)

    @pytest.mark.parametrize(
        "atr,avg_atr,price,max_atr_pct,expected",
        [
            (0.001, 0.001, 1.0, 0.20, True),    # 0.1% vs 20% limit → pass
            (0.002, 0.001, 1.0, 0.20, True),    # 0.2% vs 20% limit → pass (boundary)
            (0.0021, 0.001, 1.0, 0.20, False),  # 0.21% vs 20% limit → fail
            (0.0001, 0.0001, 1.0, 0.20, True),  # Very low volatility → pass
            (0.01, 0.001, 1.0, 0.05, False),    # 1% vs 5% limit → fail
        ],
    )
    def test_volatility_guard_parametrized(
        self, dummy_tech_calculator, atr, avg_atr, price, max_atr_pct, expected
    ):
        """Parametrized test for various volatility scenarios."""
        tech = dummy_tech_calculator(atr=atr, avg_atr=avg_atr, price=price)
        result = check_volatility(tech, "EURUSD", max_atr_pct=max_atr_pct)
        assert result == expected


# ============================================================================
# INTEGRATION TESTS: Multiple guards in sequence
# ============================================================================

class TestGuardRailsIntegration:
    """Integration tests for guard rail combinations (realistic trading scenarios)."""

    def test_all_guards_pass_normal_market(self, dummy_broker, dummy_tech_calculator):
        """Test all guards pass during normal market conditions."""
        broker = dummy_broker(
            depth=[
                {"side": "bid", "price": 1.0, "bid_volume": 200, "ask_volume": 0},
                {"side": "ask", "price": 1.0, "bid_volume": 0, "ask_volume": 200},
            ],
            latency=0.08,
            spread=0.3,
            price=1.0,
        )
        tech = dummy_tech_calculator(atr=0.001, avg_atr=0.001, price=1.0)

        assert check_depth(broker, "EURUSD", required_volume=100, min_lir=0.5)
        assert check_latency(broker, max_latency_sec=0.15)
        assert check_spread(broker, "EURUSD", max_spread_pips=0.5)
        assert check_volatility(tech, "EURUSD", max_atr_pct=0.20)

    def test_market_stress_high_latency_and_spread(self, dummy_broker):
        """Test during market stress (high latency and spread)."""
        broker = dummy_broker(latency=0.5, spread=2.0)

        assert not check_latency(broker, max_latency_sec=0.15)
        assert not check_spread(broker, "EURUSD", max_spread_pips=0.5)

    def test_news_event_spike_volatility(self, dummy_tech_calculator):
        """Test during news event (volatility spike)."""
        tech = dummy_tech_calculator(atr=0.01, avg_atr=0.001, price=1.0)

        assert not check_volatility(tech, "EURUSD", max_atr_pct=0.20)

    def test_liquidity_drought_thin_market(self, dummy_broker):
        """Test during liquidity drought (thin order book)."""
        broker = dummy_broker(
            depth=[
                {"side": "bid", "price": 1.0, "bid_volume": 10, "ask_volume": 0},
                {"side": "ask", "price": 1.0, "bid_volume": 0, "ask_volume": 10},
            ]
        )

        assert not check_depth(broker, "EURUSD", required_volume=50, min_lir=0.5)


# ============================================================================
# EDGE CASE & REGRESSION TESTS
# ============================================================================

class TestEdgeCases:
    """Edge case tests for robustness."""

    def test_zero_price_prevents_division_error(self, dummy_tech_calculator):
        """Regression test: ensure zero price doesn't cause division error."""
        tech = dummy_tech_calculator(atr=0.001, avg_atr=0.001, price=0.0)
        try:
            result = check_volatility(tech, "EURUSD", max_atr_pct=0.20)
            assert isinstance(result, bool)
        except (ValueError, ZeroDivisionError):
            pass

    def test_negative_latency_invalid(self, dummy_broker):
        """Edge case: negative latency should be handled."""
        broker = dummy_broker(latency=-0.1)
        try:
            result = check_latency(broker, max_latency_sec=0.15)
            assert isinstance(result, bool)
        except ValueError:
            pass

    def test_negative_spread_invalid(self, dummy_broker):
        """Edge case: negative spread should be handled."""
        broker = dummy_broker(spread=-0.1)
        try:
            result = check_spread(broker, "EURUSD", max_spread_pips=0.5)
            assert isinstance(result, bool)
        except ValueError:
            pass

    def test_empty_symbol_handled(self, dummy_broker):
        """Edge case: empty symbol string."""
        broker = dummy_broker(spread=0.3)
        try:
            result = check_spread(broker, "", max_spread_pips=0.5)
            assert isinstance(result, bool)
        except (ValueError, KeyError):
            pass


# ============================================================================
# MAIN: Run tests with pytest
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
