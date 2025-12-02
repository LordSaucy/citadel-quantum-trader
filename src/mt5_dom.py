import MetaTrader5 as mt5
import pandas as pd

# -----------------------------------------------------------------
# Initialise the MT5 connection (the same credentials you already
# store in Vault and inject into the container).
# -----------------------------------------------------------------
def init_mt5(login: int, password: str, server: str = "MetaQuotes-Demo"):
    if not mt5.initialize(server=server, login=login, password=password):
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")

# -----------------------------------------------------------------
# Pull the top N levels for a given symbol.
# -----------------------------------------------------------------
def get_dom(symbol: str, depth: int = 20) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
        price, bid_volume, ask_volume
    ordered from best price outward.
    """
    # MT5 returns a list of dictionaries, each with price, volume, type (BID/ASK)
    depth_raw = mt5.market_book_get(symbol, depth)
    if depth_raw is None:
        raise RuntimeError(f"Failed to get market book for {symbol}")

    rows = []
    for level in depth_raw:
        rows.append({
            "price": level.price,
            "bid_volume": level.volume if level.type == mt5.BOOK_TYPE_BID else 0,
            "ask_volume": level.volume if level.type == mt5.BOOK_TYPE_ASK else 0,
        })
    df = pd.DataFrame(rows)
    # Collapse duplicate prices (sometimes MT5 splits a price into several rows)
    df = df.groupby("price", as_index=False).sum()
    return df.sort_values("price", ascending=False)   # best bid on top
