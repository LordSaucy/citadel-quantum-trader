# src/market_data/ibkr_dom.py
#!/usr/bin/env python3
"""
IBKR DOM feed – maintains a live, thread‑safe pandas DataFrame
containing the top N price levels for any subscribed symbol.
"""

import threading
from typing import Dict
import pandas as pd
from ib_insync import IB, util

# -----------------------------------------------------------------
# Global cache (one per symbol) – protected by a lock
# -----------------------------------------------------------------
_dom_cache: Dict[str, pd.DataFrame] = {}
_cache_lock = threading.RLock()

# -----------------------------------------------------------------
def connect_ibkr(host: str = "127.0.0.1", port: int = 7497, client_id: int = 1) -> IB:
    """Connect to the local IB Gateway / TWS."""
    ib = IB()
    ib.connect(host, port, clientId=client_id)
    return ib

# -----------------------------------------------------------------
def _dom_callback(tick):
    """
    IB‑insync callback – receives a TickPriceBidAsk object.
    We keep only the top N levels (default 20) for each symbol.
    """
    symbol = tick.contract.symbol
    with _cache_lock:
        # Build a tiny DataFrame from the tick (price, bidVol, askVol)
        row = {
            "price":        tick.price,
            "bid_volume":   tick.bidSize,
            "ask_volume":   tick.askSize,
        }
        df = _dom_cache.get(symbol)
        if df is None:
            df = pd.DataFrame([row])
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        # Keep only the most recent N rows (sorted by price distance)
        df = df.sort_values("price").tail(20)
        _dom_cache[symbol] = df

# -----------------------------------------------------------------
def start_dom_subscription(ib: IB, symbol: str, exchange: str = "IDEALPRO"):
    """
    Subscribe to market‑depth updates for a single symbol.
    The callback populates the global cache.
    """
    contract = ib.qualifyContracts(util.create_contract(symbol, exchange))[0]
    ib.reqMktDepth(contract, numRows=20, isSmartDepth=True)
    ib.pendingTickersEvent += lambda _: _dom_callback(_)

# -----------------------------------------------------------------
def get_dom(symbol: str) -> pd.DataFrame:
    """
    Return a **copy** of the latest cached DOM for `symbol`.
    Caller must not modify the returned DataFrame.
    """
    with _cache_lock:
        df = _dom_cache.get(symbol)
        return df.copy() if df is not None else pd.DataFrame()
