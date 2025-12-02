def get_market_depth(self, symbol: str, depth: int = 30) -> List[Dict]:
    # MT5 provides depth via MarketBookGet(symbol, depth)
    # The API returns a list of tuples (price, volume, type)
    raw = mt5.market_book_get(symbol)
    if not raw:
        return []

    depth_data = []
    for entry in raw[:depth]:
        depth_data.append({
            "price": entry.price,
            "bid_vol": entry.volume if entry.type == mt5.BOOK_TYPE_BUY else 0,
            "ask_vol": entry.volume if entry.type == mt5.BOOK_TYPE_SELL else 0,
        })
    return depth_data

 # existing send_order implementation ...

    def send_order_secondary(self, **kwargs):
        # For MT5 we simply delegate to the IBKR adapter (see below)
        raise NotImplementedError("MT5Adapter cannot act as secondary")

