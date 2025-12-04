# src/data_feed/feed_factory.py
from .mt5_feed import MT5Feed
from .ibkr_feed import IBKRFeed
from .binance_feed import BinanceFeed


ASSET_FEED_MAP = {
    "forex": MT5Feed,
    "metal": MT5Feed,
    "index": MT5Feed,          # many indices are CFD‑style via MT5
    "stock": IBKRFeed,
    "future": IBKRFeed,        # IBKR supports futures
    "crypto": BinanceFeed,
}


def get_feed(asset_class: str):
    cls = ASSET_FEED_MAP.get(asset_class.lower())
    if not cls:
        raise ValueError(f"Unsupported asset class: {asset_class}")
    return cls()
