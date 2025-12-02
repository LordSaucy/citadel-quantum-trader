import pytest
import asyncio
from src.triangular_arb_executor import execute_triangular_arb, ArbExecutionError
from src.broker_interface import BrokerInterface

class DummyBroker(BrokerInterface):
    async def get_current_price(self, symbol): return 1.2000
    async def get_book_depth(self, symbol, price): return 10.0   # ample depth
    async def submit_order_async(self, symbol, volume, side, price=None): return "TICKET123"
    async def get_order_status(self, ticket): return "filled"
    async def get_last_fill(self, symbol, side): return {"price": 1.2001, "timestamp": 0}
    async def get_spread_at_timestamp(self, symbol, ts): return 0.00002   # 0.2 pips

@pytest.mark.asyncio
async def test_latency_allowance_pass():
    cfg = {
        "arb": {
            "latency_allowance_secs": 0.15,
            "latency_multiplier": 2.0,
            "min_depth_multiplier": 2.0,
            "spread_buffer_pips": 0.5,
            "partial_fill_timeout_secs": 5,
        }
    }
    # monkey‑patch Config loader to return the above dict
    from src import config_loader
    config_loader.Config._instance = None
    config_loader.Config._load = lambda _: setattr(config_loader.Config, "settings", cfg)

    broker = DummyBroker()
    legs = [
        {"symbol": "EURUSD", "side": "buy", "volume": 0.01},
        {"symbol": "USDJPY", "side": "sell", "volume": 0.01},
        {"symbol": "EURJPY", "side": "sell", "volume": 0.01},
    ]
    # Gross profit 1.0 pips > latency requirement (~0.3 pips)
    await execute_triangular_arb(broker, legs, gross_profit_pips=1.0)

@pytest.mark.asyncio
async def test_latency_allowance_fail():
    # Same config, but profit too low
    cfg = {{
        "arb": {
            "latency_allowance_secs": 0.15,
            "latency_multiplier": 2.0,
            "min_depth_multiplier": 2.0,
            "spread_buffer_pips": 0.5,
            "partial_fill_timeout_secs": 5,
        }
    }
    patch_config(monkeypatch, cfg)

    broker = DummyBroker()
    legs = [
        {"symbol": "EURUSD", "side": "buy",  "volume": 0.01},
        {"symbol": "USDJPY", "side": "sell", "volume": 0.01},
        {"symbol": "EURJPY", "side": "sell", "volume": 0.01},
    ]

    # Gross profit 1.0 pips > latency requirement (~0.3 pips)
    await execute_triangular_arb(broker, legs, gross_profit_pips=1.0)


@pytest.mark.asyncio
async def test_latency_allowance_fail(monkeypatch):
    """Profit too low once latency cost is accounted for."""
    cfg = {
        "arb": {
            "latency_allowance_secs": 0.15,
            "latency_multiplier": 2.0,
            "min_depth_multiplier": 2.0,
            "spread_buffer_pips": 0.5,
            "partial_fill_timeout_secs": 5,
        }
    }
    patch_config(monkeypatch, cfg)

    broker = DummyBroker()
    legs = [
        {"symbol": "EURUSD", "side": "buy",  "volume": 0.01},
        {"symbol": "USDJPY", "side": "sell", "volume": 0.01},
        {"symbol": "EURJPY", "side": "sell", "volume": 0.01},
    ]

    # Gross profit 0.2 pips < required ~0.3 pips → should raise
    with pytest.raises(ArbExecutionError, match="latency"):
        await execute_triangular_arb(broker, legs, gross_profit_pips=0.2)


@pytest.mark.asyncio
async def test_depth_check_fail(monkeypatch):
    """Depth is insufficient for at least one leg."""
    cfg = {
        "arb": {
            "latency_allowance_secs": 0.15,
            "latency_multiplier": 2.0,
            "min_depth_multiplier": 2.0,
            "spread_buffer_pips": 0.5,
            "partial_fill_timeout_secs": 5,
        }
    }
    patch_config(monkeypatch, cfg)

    # Provide depth = 0.01 lots, but we request 0.01 lots * 2 = 0.02 required → fail
    broker = DummyBroker(depth=0.01)
    legs = [
        {"symbol": "EURUSD", "side": "buy",  "volume": 0.01},
        {"symbol": "USDJPY", "side": "sell", "volume": 0.01},
        {"symbol": "EURJPY", "side": "sell", "volume": 0.01},
    ]

    with pytest.raises(ArbExecutionError, match="Insufficient depth"):
        await execute_triangular_arb(broker, legs, gross_profit_pips=2.0)


@pytest.mark.asyncio
async def test_partial_fill_guard(monkeypatch):
    """First leg only partially fills → abort the arb."""
    cfg = {
        "arb": {
            "latency_allowance_secs": 0.15,
            "latency_multiplier": 2.0,
            "min_depth_multiplier": 2.0,
            "spread_buffer_pips": 0.5,
            "partial_fill_timeout_secs": 1,   # short timeout for test speed
        }
    }
    patch_config(monkeypatch, cfg)

    # Simulate a *partial* fill status for the first order
    broker = DummyBroker(fill_status="partial")
    legs = [
        {"symbol": "EURUSD", "side": "buy",  "volume": 0.01},
        {"symbol": "USDJPY", "side": "sell", "volume": 0.01},
        {"symbol": "EURJPY", "side": "sell", "volume": 0.01},
    ]

    with pytest.raises(ArbExecutionError, match="not fully filled"):
        await execute_triangular_arb(broker, legs, gross_profit_pips=2.0)


@pytest.mark.asyncio
async def test_spread_adjusted_profit_fail(monkeypatch):
    """Net profit after spread subtraction falls below buffer."""
    cfg = {
        "arb": {
            "latency_allowance_secs": 0.15,
            "latency_multiplier": 2.0,
            "min_depth_multiplier": 2.0,
            "spread_buffer_pips": 0.5,   # require at least 0.5 pips net
            "partial_fill_timeout_secs": 5,
        }
    }
    patch_config(monkeypatch, cfg)

    # Use a realistic spread of 0.3 pips (0.00003) → net profit will be <0.5 pips
    broker = DummyBroker(spread=0.00003)
    legs = [
        {"symbol": "EURUSD", "side": "buy",  "volume": 0.01},
        {"symbol": "USDJPY", "side": "sell", "volume": 0.01},
        {"symbol": "EURJPY", "side": "sell", "volume": 0.01},
    ]

    # Gross profit 0.8 pips – after subtracting 0.3 pips spread per leg (≈0.9 pips total)
    # net profit will be <0.5 pips → should raise
    with pytest.raises(ArbExecutionError, match="Net arb profit"):
        await execute_triangular_arb(broker, legs, gross_profit_pips=0.8)


@pytest.mark.asyncio
async def test_successful_arb(monkeypatch):
    """All buffers pass → arb counted as success."""
    cfg = {
        "arb": {
            "latency_allowance_secs": 0.15,
            "latency_multiplier": 2.0,
            "min_depth_multiplier": 2.0,
            "spread_buffer_pips": 0.5,
            "partial_fill_timeout_secs": 5,
        }
    }
    patch_config(monkeypatch, cfg)

    broker = DummyBroker()
    legs = [
        {"symbol": "EURUSD", "side": "buy",  "volume": 0.01},
        {"symbol": "USDJPY", "side": "sell", "volume": 0.01},
        {"symbol": "EURJPY", "side": "sell", "volume": 0.01},
    ]

    # Gross profit 2.0 pips – comfortably above all thresholds
    await execute_triangular_arb(broker, legs, gross_profit_pips=2.0)

    # If we reach here, the arb succeeded – you can also assert that the
    # Prometheus counter was incremented (requires a fixture that exposes the
    # registry, omitted for brevity).
 
