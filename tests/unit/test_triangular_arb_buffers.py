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
    cfg = {
