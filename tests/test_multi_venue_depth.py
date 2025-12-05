#!/usr/bin/env python3
"""
Unit tests for multi-venue market depth aggregation

Tests the VenueManager's ability to aggregate order book depth from
multiple trading venues (MT5, IBKR) and determine if minimum depth
requirements are met.

Author: Lawful Banker
Created: 2024‑11‑26
Version: 2.0 – With Documented Stubs
"""

import pytest
from citadel.trading.venue_manager import VenueManager
from citadel.trading.config_loader import Config


@pytest.fixture
def fake_cfg(monkeypatch):
    """
    Fixture: Minimal multi-venue configuration for testing.
    
    Creates a configuration with two fake trading venues (MT5 and IBKR)
    with contract notional and depth multiplier settings. These values
    control position sizing and liquidity requirements.
    
    Monkeypatches Config.settings so VenueManager loads our test config
    instead of reading from files/environment.
    """
    cfg = {
        "venues": [
            {"name": "fake_mt5", "type": "mt5", "vault_path": "dummy"},
            {"name": "fake_ibkr", "type": "ibkr", "vault_path": "dummy"},
        ],
        "min_depth_multiplier": 2.0,  # Required liquidity = 2 × required_lot
        "contract_notional": 100_000,  # $100k per standard contract
    }
    monkeypatch.setattr(Config, "settings", cfg, raising=False)
    return cfg


class StubAdapter:
    """
    Stub adapter for MT5 and IBKR that simulates market depth responses.
    
    This stub mimics the interface of real trading venue adapters
    (MT5Adapter, IBKRAdapter) without connecting to actual brokers.
    
    Used by monkeypatch to replace real adapter classes during tests.
    """
    
    def __init__(self, *a, **kw) -> None:
        """
        ✅ INTENTIONALLY EMPTY: Mock implementation of adapter initialization
        
        WHY THIS IS EMPTY:
        - This is a test stub that doesn't need initialization logic
        - Real adapters (MT5Adapter, IBKRAdapter) would connect to brokers,
          load credentials, establish WebSocket connections, etc.
        - For testing VenueManager's aggregation logic, we don't need those
        - The stub only needs to provide get_market_depth() method
        - Initialization would be tested in adapter-specific unit tests
        - Using *a, **kw allows flexibility: stub accepts any arguments
          that the real adapter constructor would receive
        
        WHEN THIS WOULD BE IMPLEMENTED:
        - If we needed to test adapter initialization failure handling
        - If we needed to verify VenueManager passes correct parameters
        - Example: `self.venue_type = kw.get('type'); self.config = a[0]`
        
        CURRENT PURPOSE:
        - Silently accept and ignore all initialization parameters
        - Allow VenueManager to instantiate the stub without errors
        - Let the stub proceed to mock get_market_depth() calls
        """
        pass
    
    def get_market_depth(self, symbol: str, depth: int = 20):
        """
        ✅ Mock implementation: Return deterministic market depth data.
        
        This stub returns the same depth for all symbols and venues:
        - 0.5 lot on the bid side (buy pressure)
        - 0.5 lot on the ask side (sell pressure)
        - Both at price 1.0 (simplified for testing)
        
        Returns:
            List[Dict]: Depth data in the expected format
        """
        # Return consistent depth: 0.5 lot on each side for both venues
        return [
            {"price": 1.0, "volume": 0.5, "side": "bid"},
            {"price": 1.0, "volume": 0.5, "side": "ask"},
        ]


def test_aggregate_depth(monkeypatch, fake_cfg):
    """
    ✅ Test: VenueManager.aggregate_depth() sums liquidity from multiple venues.
    
    Workflow:
    1. Monkeypatch MT5Adapter and IBKRAdapter to use StubAdapter
    2. Both stubs return 0.5 lot bid + 0.5 lot ask
    3. VenueManager should aggregate:
       - Total bid volume: 0.5 (MT5) + 0.5 (IBKR) = 1.0 lot
       - Total ask volume: 0.5 (MT5) + 0.5 (IBKR) = 1.0 lot
    4. Assert aggregated depth matches expected totals
    
    Mocking Strategy:
    - Replace real adapters with StubAdapter that returns deterministic data
    - No actual broker connections or WebSocket streams
    - Isolates VenueManager logic from adapter implementation
    """
    # ✅ Replace real adapters with our stub
    monkeypatch.setattr("citadel.trading.mt5_adapter.MT5Adapter", StubAdapter)
    monkeypatch.setattr("citadel.trading.ibkr_adapter.IBKRAdapter", StubAdapter)
    
    # Create VenueManager and aggregate depth for EURUSD
    vm = VenueManager()
    agg = vm.aggregate_depth("EURUSD")
    
    # ✅ FIXED: Use pytest.approx() for floating point comparisons
    # WHY: Floating point arithmetic can have precision issues (e.g., 0.1 + 0.2 != 0.3 in IEEE 754)
    # Using == for float values is unreliable; pytest.approx() uses relative/absolute tolerance
    # Default: relative tolerance 1e-6, absolute tolerance 1e-12
    assert agg["bid_volume"] == pytest.approx(1.0), "Expected 0.5 (MT5) + 0.5 (IBKR) = 1.0 total bid"
    assert agg["ask_volume"] == pytest.approx(1.0), "Expected 0.5 (MT5) + 0.5 (IBKR) = 1.0 total ask"


def test_meets_minimum_depth(monkeypatch, fake_cfg):
    """
    ✅ Test: VenueManager.meets_minimum() validates liquidity requirements.
    
    Scenario:
    - Total available liquidity: 1.0 lot (bid) + 1.0 lot (ask) from aggregation
    - Required position: 0.02 lot
    - Minimum depth multiplier: 2.0 (from config)
    - Required liquidity threshold: 2.0 × 0.02 = 0.04 lot
    - Assert: 1.0 lot available > 0.04 lot required → meets_minimum() = True
    
    Formula:
    - required_depth = min_depth_multiplier × required_lot
    - meets_minimum = (available_bid_volume + available_ask_volume) >= required_depth
    """
    monkeypatch.setattr("citadel.trading.mt5_adapter.MT5Adapter", StubAdapter)
    monkeypatch.setattr("citadel.trading.ibkr_adapter.IBKRAdapter", StubAdapter)
    
    vm = VenueManager()
    agg = vm.aggregate_depth("EURUSD")
    
    # Required position size based on capital
    # Assume trading capital = $2000 → lot = $2000 / $100,000 = 0.02 lot
    required_lot = 0.02
    
    # Check if available liquidity meets minimum requirement
    # Required depth = 2.0 × 0.02 = 0.04 lot
    # Available = 1.0 bid + 1.0 ask = 2.0 lot total → meets requirement
    assert vm.meets_minimum(required_lot, agg), \
        f"Expected meets_minimum=True: required={2.0 * required_lot:.4f}, available={agg['bid_volume'] + agg['ask_volume']:.1f}"


def test_meets_minimum_insufficient_depth(monkeypatch, fake_cfg):
    """
    ✅ Test: VenueManager.meets_minimum() rejects insufficient liquidity.
    
    Scenario:
    - Total available liquidity: 1.0 lot (from aggregation)
    - Required position: 1.0 lot (very large)
    - Minimum depth multiplier: 2.0
    - Required liquidity threshold: 2.0 × 1.0 = 2.0 lot
    - Assert: 1.0 lot available < 2.0 lot required → meets_minimum() = False
    
    This ensures the system rejects trades that don't have sufficient
    liquidity protection.
    """
    monkeypatch.setattr("citadel.trading.mt5_adapter.MT5Adapter", StubAdapter)
    monkeypatch.setattr("citadel.trading.ibkr_adapter.IBKRAdapter", StubAdapter)
    
    vm = VenueManager()
    agg = vm.aggregate_depth("EURUSD")
    
    # Request a very large position
    large_required_lot = 1.0
    
    # Required depth = 2.0 × 1.0 = 2.0 lot
    # Available = 1.0 lot → INSUFFICIENT
    assert not vm.meets_minimum(large_required_lot, agg), \
        f"Expected meets_minimum=False: required={2.0 * large_required_lot:.1f}, available={agg['bid_volume'] + agg['ask_volume']:.1f}"


@pytest.mark.parametrize("symbol,required_lot,should_meet", [
    ("EURUSD", 0.01, True),   # 2 × 0.01 = 0.02 < 1.0 available → True
    ("EURUSD", 0.02, True),   # 2 × 0.02 = 0.04 < 1.0 available → True
    ("EURUSD", 0.50, True),   # 2 × 0.50 = 1.00 = 1.0 available → True
    ("EURUSD", 0.51, False),  # 2 × 0.51 = 1.02 > 1.0 available → False
    ("GBPUSD", 0.25, True),   # Same logic for GBPUSD
    ("GBPUSD", 0.75, False),  # Same logic for GBPUSD
])
def test_meets_minimum_parametrized(monkeypatch, fake_cfg, symbol, required_lot, should_meet):
    """
    ✅ Test: VenueManager.meets_minimum() across multiple scenarios.
    
    Parametrized test that verifies the depth validation logic
    works correctly for various position sizes and symbols.
    
    Test matrix:
    - Symbols: EURUSD, GBPUSD
    - Position sizes: 0.01 → 0.75 lot
    - Each combination validates meets_minimum() behavior
    """
    monkeypatch.setattr("citadel.trading.mt5_adapter.MT5Adapter", StubAdapter)
    monkeypatch.setattr("citadel.trading.ibkr_adapter.IBKRAdapter", StubAdapter)
    
    vm = VenueManager()
    agg = vm.aggregate_depth(symbol)
    
    result = vm.meets_minimum(required_lot, agg)
    expected_msg = f"Symbol={symbol}, required={2.0 * required_lot:.2f}, available={agg['bid_volume'] + agg['ask_volume']:.1f}"
    
    assert result == should_meet, expected_msg


def test_venue_manager_initialization(monkeypatch, fake_cfg):
    """
    ✅ Test: VenueManager initializes with correct venue adapters.
    
    Verifies that VenueManager:
    1. Loads configuration from Config.settings
    2. Instantiates adapters for each venue
    3. Stores venues in internal state
    """
    monkeypatch.setattr("citadel.trading.mt5_adapter.MT5Adapter", StubAdapter)
    monkeypatch.setattr("citadel.trading.ibkr_adapter.IBKRAdapter", StubAdapter)
    
    vm = VenueManager()
    
    # ✅ Verify VenueManager has venues
    assert hasattr(vm, "venues"), "VenueManager should have 'venues' attribute"
    assert len(vm.venues) == 2, "Expected 2 venues (MT5, IBKR)"
    
    # ✅ FIXED: Use pytest.approx() for floating point comparisons
    # WHY: Configuration values like min_depth_multiplier are floats
    # Using == can fail due to floating point precision issues
    # pytest.approx() compares with tolerance (default: 1e-6 relative, 1e-12 absolute)
    assert vm.min_depth_multiplier == pytest.approx(2.0), "Expected min_depth_multiplier=2.0 from config"
    assert vm.contract_notional == pytest.approx(100_000), "Expected contract_notional=100_000 from config"


def test_aggregate_depth_empty_venues(monkeypatch, fake_cfg):
    """
    ✅ Test: VenueManager.aggregate_depth() handles edge cases.
    
    Tests behavior when venues return no depth data or minimal data.
    This ensures robustness in adverse market conditions.
    """
    class EmptyStubAdapter:
        """Adapter stub that returns empty depth."""
        
        def __init__(self, *a, **kw) -> None:
            """
            ✅ INTENTIONALLY EMPTY: Mock initialization
            
            WHY THIS IS EMPTY:
            - Same rationale as StubAdapter.__init__
            - This variant is used to test edge case handling
            """
            pass
        
        def get_market_depth(self, symbol: str, depth: int = 20):
            """Return empty depth data."""
            return []
    
    monkeypatch.setattr("citadel.trading.mt5_adapter.MT5Adapter", EmptyStubAdapter)
    monkeypatch.setattr("citadel.trading.ibkr_adapter.IBKRAdapter", EmptyStubAdapter)
    
    vm = VenueManager()
    agg = vm.aggregate_depth("EURUSD")
    
    # ✅ FIXED: Use pytest.approx() for floating point comparisons
    # WHY: bid_volume and ask_volume are floats
    # Even 0.0 should be compared with tolerance to handle floating point edge cases
    # (e.g., accumulation of tiny rounding errors that result in -1e-15 instead of 0.0)
    assert agg["bid_volume"] == pytest.approx(0.0), "Expected bid_volume=0 for empty depth"
    assert agg["ask_volume"] == pytest.approx(0.0), "Expected ask_volume=0 for empty depth"
    
    # ✅ meets_minimum should return False with no liquidity
    assert not vm.meets_minimum(0.01, agg), "Expected meets_minimum=False with zero liquidity"
