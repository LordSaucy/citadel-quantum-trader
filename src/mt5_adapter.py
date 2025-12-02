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

class MT5Adapter:
    def __init__(self, paper_mode: bool = False):
        self.paper_mode = paper_mode
        # Load credentials from Vault (the same code you already have)
        self._load_credentials()
        # If paper_mode, point the MetaTrader5 client at the demo server
        if self.paper_mode:
            self.server = self.creds["server"]   # demo server from Vault
        else:
            self.server = self.creds["server"]   # production server

    # -----------------------------------------------------------------
    # Example: send_order – record latency & slippage metrics
    # -----------------------------------------------------------------
    def send_order(self, order):
        start = time.time()
        # -----------------------------------------------------------------
        # 1️⃣  Call MT5 (or IBKR) – the library automatically talks to the
        #     demo server when `self.server` points there.
        # -----------------------------------------------------------------
        response = self._mt5_send(order)   # existing low‑level call
        elapsed = time.time() - start

        # -----------------------------------------------------------------
        # 2️⃣  Record latency (seconds) – Prometheus gauge
        # -----------------------------------------------------------------
        from prometheus_client import Gauge
        latency_gauge = Gauge('cqt_latency_seconds',
                             'Round‑trip latency of order submission (seconds)',
                             ['symbol'])
        latency_gauge.labels(symbol=order.symbol).set(elapsed)

        # -----------------------------------------------------------------
        # 3️⃣  Detect rejection
        # -----------------------------------------------------------------
        reject_counter = Counter('cqt_reject_total',
                                 'Total number of order rejections',
                                 ['symbol'])
        if not response.success:
            reject_counter.labels(symbol=order.symbol).inc()
            log.warning(f"❌ Order rejected [{order.symbol}] – {response.error}")

        # -----------------------------------------------------------------
        # 4️⃣  Compute slippage (in pips) – compare requested price vs fill price
        # -----------------------------------------------------------------
        slippage_gauge = Gauge('cqt_slippage_pips',
                               'Absolute slippage per filled order (pips)',
                               ['symbol'])
        # For a BUY: slippage = fill_price - requested_price
        # For a SELL: slippage = requested_price - fill_price
        if order.direction.upper() == "BUY":
            slip = (response.fill_price - order.price) / self.pip_size(order.symbol)
        else:
            slip = (order.price - response.fill_price) / self.pip_size(order.symbol)

        slippage_gauge.labels(symbol=order.symbol).set(abs(slip))

        return response

