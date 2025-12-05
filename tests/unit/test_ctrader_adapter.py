#!/usr/bin/env python3
"""
Unit tests for CTraderAdapter

Tests the ctrader_adapter module with mocked HTTP responses.
Uses pytest fixtures and monkeypatch to stub out external dependencies.

Author: Lawful Banker
Created: 2024‑11‑26
Version: 2.0 – With Documented Stubs
"""

import pytest
from src.ctrader_adapter import CTraderAdapter


@pytest.fixture
def cfg():
    """
    Fixture: Minimal CTrader configuration for testing.
    
    Provides sandbox API credentials that will be monkeypatched
    during tests to avoid actual API calls.
    """
    return {
        "ctrader": {
            "api_url": "https://sandbox.api.ctrader.com/v2",
            "api_key": "dummy-key"
        }
    }


class DummyResp:
    """
    Stub HTTP response object for mocking requests.Session responses.
    
    This stub mimics the interface of requests.Response without making
    actual network calls. Used by monkeypatch to replace requests.sessions.Session.get().
    """
    
    ok = True  # ✅ Indicates successful HTTP status (200-299)
    
    def raise_for_status(self) -> None:
        """
        ✅ INTENTIONALLY EMPTY: Mock implementation of requests.Response.raise_for_status()
        
        WHY THIS IS EMPTY:
        - This is a test stub that mimics requests.Response interface
        - In real requests library, raise_for_status() raises HTTPError if status >= 400
        - Since self.ok = True (simulating successful response), raising an exception
          would break the test flow
        - The test fixture doesn't need to actually raise anything – we're testing
          the adapter's logic, not the requests library
        - An exception handler in the real code would be tested elsewhere via
          integration tests or error case tests
        
        WHEN THIS WOULD BE IMPLEMENTED:
        - If we needed to test error handling (status_code >= 400), we'd
          create a separate ErrorResp stub class with real error behavior
        - Example: `raise requests.exceptions.HTTPError("404 Not Found")`
        """
        pass


def test_connect_success(monkeypatch, cfg):
    """
    ✅ Test: CTraderAdapter.connect() succeeds with mock HTTP response.
    
    Workflow:
    1. Monkeypatch requests.sessions.Session.get to return DummyResp (200 OK)
    2. Create CTraderAdapter with dummy config
    3. Call adapter.connect() which internally calls requests.get()
    4. Assert connection returns True
    
    Mocking Strategy:
    - We replace the real HTTP GET with our stub so no network call happens
    - DummyResp has ok=True to simulate successful response
    - The adapter should interpret ok=True and return True
    """
    # ✅ Replace requests.Session.get with a lambda that returns DummyResp
    # This prevents actual HTTP calls during testing
    monkeypatch.setattr(
        "requests.sessions.Session.get",
        lambda *a, **kw: DummyResp()
    )
    
    adapter = CTraderAdapter(cfg)
    assert adapter.connect() is True, "Expected connect() to return True with ok=True response"


def test_connect_failure(monkeypatch, cfg):
    """
    ✅ Test: CTraderAdapter.connect() fails with mock error response.
    
    This test verifies that the adapter properly handles failed connections.
    """
    class FailResp:
        """Stub response simulating HTTP error (e.g., 500 Internal Server Error)."""
        ok = False  # ❌ Indicates failed HTTP status
        
        def raise_for_status(self) -> None:
            """
            ✅ INTENTIONALLY EMPTY: Mock implementation of raise_for_status()
            
            WHY THIS IS EMPTY:
            - For this error stub, we're testing the adapter's response to ok=False
            - The adapter should check the `ok` attribute before calling raise_for_status()
            - If the adapter properly handles ok=False, it won't reach this method
            - If the adapter does call this, we'd implement it here to raise an exception
            - Keeping it empty allows the test to verify the adapter checks `ok` first
            """
            pass
    
    monkeypatch.setattr("requests.sessions.Session.get", lambda *a, **kw: FailResp())
    
    adapter = CTraderAdapter(cfg)
    assert adapter.connect() is False, "Expected connect() to return False with ok=False response"


def test_get_account_info(monkeypatch, cfg):
    """
    ✅ Test: CTraderAdapter.get_account_info() with mocked response.
    
    Verifies the adapter can parse account info from a mocked API response.
    """
    class AccountResp:
        """Stub response for account info endpoint."""
        ok = True
        
        def json(self):
            """Return mock account data."""
            return {
                "accountId": "123456",
                "balance": 10000.0,
                "currency": "USD",
                "leverage": 100
            }
        
        def raise_for_status(self) -> None:
            """
            ✅ INTENTIONALLY EMPTY: Same pattern as DummyResp
            
            WHY THIS IS EMPTY:
            - This stub represents a successful (ok=True) response
            - We're testing successful response handling, not error paths
            - The real requests.Response.raise_for_status() only raises on error
            - Since ok=True, nothing is raised
            """
            pass
    
    monkeypatch.setattr("requests.sessions.Session.get", lambda *a, **kw: AccountResp())
    
    adapter = CTraderAdapter(cfg)
    info = adapter.get_account_info()
    
    assert info["accountId"] == "123456"
    # ✅ FIXED: Use pytest.approx() for floating point comparison
    # WHY: Account balance is a float that may have precision issues from API
    # API responses from JSON serialization can have floating point precision errors
    # Default tolerance: 1e-6 relative, 1e-12 absolute
    assert info["balance"] == pytest.approx(10000.0), "Expected balance=10000.0 from API response"
    assert info["leverage"] == 100


def test_get_market_depth(monkeypatch, cfg):
    """
    ✅ Test: CTraderAdapter.get_market_depth() with mocked market data.
    
    Verifies the adapter can fetch and parse order book depth.
    """
    class DepthResp:
        """Stub response for market depth endpoint."""
        ok = True
        
        def json(self):
            """Return mock order book depth data."""
            return {
                "bid": [
                    {"price": 1.0950, "volume": 1.0},
                    {"price": 1.0949, "volume": 2.0},
                ],
                "ask": [
                    {"price": 1.0951, "volume": 1.5},
                    {"price": 1.0952, "volume": 2.5},
                ]
            }
        
        def raise_for_status(self) -> None:
            """
            ✅ INTENTIONALLY EMPTY: Mock implementation
            
            WHY THIS IS EMPTY:
            - This stub represents successful market data retrieval
            - In production, raise_for_status() would raise on HTTP errors
            - Our test provides ok=True and valid market data
            - Error handling is tested in separate test cases
            """
            pass
    
    monkeypatch.setattr("requests.sessions.Session.get", lambda *a, **kw: DepthResp())
    
    adapter = CTraderAdapter(cfg)
    depth = adapter.get_market_depth("EURUSD", depth=20)
    
    assert len(depth["bid"]) == 2
    # ✅ FIXED: Use pytest.approx() for floating point comparisons
    # WHY: Price data is float that may have precision issues from API JSON serialization
    # Order book prices from external APIs can have floating point precision errors
    # Example: 1.0950 might be stored as 1.0949999999999998 in IEEE 754
    # Default tolerance: 1e-6 relative, 1e-12 absolute
    assert depth["bid"][0]["price"] == pytest.approx(1.0950), "Expected bid price=1.0950"
    assert len(depth["ask"]) == 2
    assert depth["ask"][0]["price"] == pytest.approx(1.0951), "Expected ask price=1.0951"


# =====================================================================
# Parametrized Tests – Testing multiple symbols at once
# =====================================================================
@pytest.mark.parametrize("symbol,expected_bid,expected_ask", [
    ("EURUSD", 1.0950, 1.0951),
    ("GBPUSD", 1.2500, 1.2501),
    ("USDJPY", 110.00, 110.01),
])
def test_get_market_depth_multiple_symbols(monkeypatch, cfg, symbol, expected_bid, expected_ask):
    """
    ✅ Test: CTraderAdapter.get_market_depth() works for multiple currency pairs.
    
    Uses parametrized testing to verify the adapter handles different symbols.
    """
    class DepthResp:
        ok = True
        
        def __init__(self, bid, ask):
            self.bid = bid
            self.ask = ask
        
        def json(self):
            return {
                "bid": [{"price": self.bid, "volume": 1.0}],
                "ask": [{"price": self.ask, "volume": 1.0}]
            }
        
        def raise_for_status(self) -> None:
            """
            ✅ INTENTIONALLY EMPTY: Mock implementation
            
            WHY THIS IS EMPTY:
            - Parametrized test fixture; success path only
            - Error handling tested separately
            """
            pass
    
    monkeypatch.setattr(
        "requests.sessions.Session.get",
        lambda *a, **kw: DepthResp(expected_bid, expected_ask)
    )
    
    adapter = CTraderAdapter(cfg)
    depth = adapter.get_market_depth(symbol)
    
    # ✅ FIXED: Use pytest.approx() for floating point comparisons
    # WHY: Parametrized test values are floats that may have precision issues
    # When comparing prices from different symbols and expected values,
    # floating point precision can vary, so tolerance-based comparison is needed
    assert depth["bid"][0]["price"] == pytest.approx(expected_bid), f"Expected bid price={expected_bid} for {symbol}"
    assert depth["ask"][0]["price"] == pytest.approx(expected_ask), f"Expected ask price={expected_ask} for {symbol}"
