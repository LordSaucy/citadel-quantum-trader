import MetaTrader5 as mt5

class MT5Adapter(BrokerInterface):
    # ...

    def ping(self):
        if not mt5.initialize():
            raise RuntimeError("MT5 ping failed – cannot connect")

    def get_market_depth(self, symbol, depth=20):
        # MT5 provides a depth‑of‑market request
        depth_info = mt5.market_book_get(symbol)
        if depth_info is None:
            return []
        # Convert to the generic dict format expected by the guard
        result = []
        for level in depth_info:
            result.append({
                "side": "bid" if level.type == mt5.MARKET_BOOK_TYPE_BID else "ask",
                "price": level.price,
                "bid_volume": level.volume if level.type == mt5.MARKET_BOOK_TYPE_BID else 0,
                "ask_volume": level.volume if level.type == mt5.MARKET_BOOK_TYPE_ASK else 0,
            })
        return result[:depth]

    def get_quote(self, symbol):
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"No tick for {symbol}")
        return {"bid": tick.bid, "ask": tick.ask}
 
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

# mt5_adapter.py (add to class MT5Adapter)

async def get_book_depth(self, symbol: str, price: float) -> float:
    # MT5 provides depth via `mt5.market_book_get(symbol)`
    depth = mt5.market_book_get(symbol)
    if not depth:
        return 0.0
    # Sum volumes that are within a tiny epsilon of the target price
    eps = 0.00001
    total = sum(
        entry.volume for entry in depth if abs(entry.price - price) <= eps
    )
    # MT5 volume is in lots already
    return total

async def submit_order_async(self, symbol, volume, side, price=None):
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": mt5.ORDER_TYPE_BUY if side == "buy" else mt5.ORDER_TYPE_SELL,
        "price": price or 0.0,
        "deviation": 10,
        "magic": 123456,
        "comment": "arb‑auto",
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        raise RuntimeError(f"Order failed: {result.comment}")
    # Return the ticket; the caller can poll `mt5.positions_get(ticket=…)`
    return str(result.order)
