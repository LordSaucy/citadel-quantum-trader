# src/execution/broker_router.py
from src.broker.mt5_adapter import MT5Adapter
from src.broker.ibkr_adapter import IBKRAdapter
from src.broker.binance_adapter import BinanceAdapter


BROKER_MAP = {
    "forex": MT5Adapter,
    "metal": MT5Adapter,
    "index": MT5Adapter,
    "stock": IBKRAdapter,
    "future": IBKRAdapter,
    "crypto": BinanceAdapter,
}


def get_broker(asset_class: str):
    cls = BROKER_MAP.get(asset_class.lower())
    if not cls:
        raise ValueError(f"No broker for asset class {asset_class}")
    return cls()
