import MetaTrader5 as mt5
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from prometheus_client import Counter, Gauge

log = logging.getLogger(__name__)


class BrokerInterface(ABC):
    """Abstract base class for broker adapters."""
    
    @abstractmethod
    def ping(self) -> None:
        """Test connection to broker."""
        pass

    @abstractmethod
    def get_market_depth(self, symbol: str, depth: int = 20) -> List[Dict]:
        """Get market depth data."""
        pass

    @abstractmethod
    def get_quote(self, symbol: str) -> Dict:
        """Get current bid/ask quote."""
        pass

    @abstractmethod
    def send_order(self, order: Dict) -> Dict:
        """Send an order to the broker."""
        pass


class MT5Adapter(BrokerInterface):
    """MetaTrader5 broker adapter for CQT."""

    # =====================================================================
    # Prometheus metrics – registered once
    # =====================================================================
    _latency_gauge = Gauge(
        'cqt_latency_seconds',
        'Round‑trip latency of order submission (seconds)',
        ['symbol']
    )
    _reject_counter = Counter(
        'cqt_reject_total',
        'Total number of order rejections',
        ['symbol']
    )
    _slippage_gauge = Gauge(
        'cqt_slippage_pips',
        'Absolute slippage per filled order (pips)',
        ['symbol']
    )

    def __init__(self, paper_mode: bool = False, cfg: Optional[Dict] = None) -> None:
        """
        Initialize MT5 adapter.
        
        Arguments:
            paper_mode: If True, use demo server instead of production.
            cfg: Configuration dictionary (contains credentials, servers, etc.)
        """
        self.paper_mode = paper_mode
        self.cfg = cfg or {}
        self._load_credentials()
        
        # ✅ FIXED: Removed duplicate if/else (both branches identical)
        # Now uses a single conditional assignment
        self.server = self.creds.get("server_demo" if self.paper_mode else "server_prod")

    def _load_credentials(self) -> None:
        """Load MT5 credentials from Vault or config."""
        # Placeholder: implement actual credential loading
        self.creds = {
            "server_prod": "MetaQuotes-Demo",
            "server_demo": "MetaQuotes-Demo",
            "login": self.cfg.get("mt5_login"),
            "password": self.cfg.get("mt5_password"),
        }

    def pip_size(self, symbol: str) -> float:
        """Get the pip size for a given symbol."""
        # For most FX pairs: 0.0001 (4 decimals)
        # For JPY pairs: 0.01 (2 decimals)
        if symbol.upper().endswith("JPY"):
            return 0.01
        return 0.0001

    # =====================================================================
    # Public API – BrokerInterface implementation
    # =====================================================================

    def ping(self) -> None:
        """Test connection to MT5."""
        if not mt5.initialize():
            raise RuntimeError("MT5 ping failed – cannot connect")
        log.info("MT5 ping successful")

    def get_market_depth(self, symbol: str, depth: int = 20) -> List[Dict]:
        """
        Fetch market depth (order book) from MT5.
        
        Returns a list of dicts with bid/ask volumes at each price level.
        """
        depth_info = mt5.market_book_get(symbol)
        if depth_info is None or len(depth_info) == 0:
            return []

        result = []
        for level in depth_info[:depth]:
            result.append({
                "side": "bid" if level.type == mt5.MARKET_BOOK_TYPE_BID else "ask",
                "price": level.price,
                "bid_volume": level.volume if level.type == mt5.MARKET_BOOK_TYPE_BID else 0,
                "ask_volume": level.volume if level.type == mt5.MARKET_BOOK_TYPE_ASK else 0,
            })
        return result

    def get_quote(self, symbol: str) -> Dict:
        """Get current bid/ask quote for a symbol."""
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"No tick data available for {symbol}")
        return {"bid": tick.bid, "ask": tick.ask}

    def send_order(self, order: Dict) -> Dict:
        """
        Send an order to MT5.
        
        Measures latency and slippage, records metrics.
        
        Arguments:
            order: Dict with keys {symbol, direction, volume, price, sl, tp}
        
        Returns:
            Dict with {success, order_id, fill_price, error}
        """
        start_ts = time.time()
        symbol = order.get("symbol")
        direction = order.get("direction", "BUY").upper()
        volume = order.get("volume")
        price = order.get("price")

        # ✅ FIXED: Moved logic into this single method (removed unreachable stub)
        try:
            # Build MT5 order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL,
                "price": price or 0.0,
                "deviation": 10,
                "magic": 123456,
                "comment": "cqt-auto",
            }

            # Send to MT5
            result = mt5.order_send(request)
            elapsed = time.time() - start_ts

            # Record latency
            self._latency_gauge.labels(symbol=symbol).set(elapsed)

            # Check for rejection
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self._reject_counter.labels(symbol=symbol).inc()
                log.warning(
                    f"MT5 order rejected [{symbol}]: {result.comment} "
                    f"(retcode={result.retcode})"
                )
                return {
                    "success": False,
                    "order_id": None,
                    "fill_price": None,
                    "error": result.comment,
                }

            # Compute slippage (in pips)
            fill_price = result.price if hasattr(result, 'price') else price
            if fill_price is not None and price is not None:
                if direction == "BUY":
                    slippage_pips = (fill_price - price) / self.pip_size(symbol)
                else:  # SELL
                    slippage_pips = (price - fill_price) / self.pip_size(symbol)
                self._slippage_gauge.labels(symbol=symbol).set(abs(slippage_pips))
            else:
                slippage_pips = None

            log.info(
                f"MT5 order sent: {direction} {volume} {symbol} "
                f"@ {price} (latency={elapsed:.3f}s, slippage={slippage_pips})"
            )

            return {
                "success": True,
                "order_id": str(result.order),
                "fill_price": fill_price,
                "error": None,
            }

        except Exception as exc:
            log.exception(f"MT5 order submission failed for {symbol}")
            self._reject_counter.labels(symbol=symbol).inc()
            return {
                "success": False,
                "order_id": None,
                "fill_price": None,
                "error": str(exc),
            }

    # =====================================================================
    # Helper methods (synchronous)
    # =====================================================================

    def get_book_depth(self, symbol: str, price: float) -> float:
        """
        Get the total volume available at or near a specific price level.
        
        ✅ FIXED: Removed `async` keyword (was unused – no await operations)
        """
        depth = mt5.market_book_get(symbol)
        if not depth:
            return 0.0

        # Sum volumes that are within a tiny epsilon of the target price
        eps = 0.00001
        total = sum(
            entry.volume for entry in depth if abs(entry.price - price) <= eps
        )
        # MT5 volume is in lots already
        return float(total)

    def submit_order(
        self,
        symbol: str,
        volume: float,
        side: str,
        price: Optional[float] = None,
    ) -> str:
        """
        Submit an order synchronously and return the order ticket.
        
        ✅ FIXED: Removed `async` keyword (was unused – no await operations)
        """
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": mt5.ORDER_TYPE_BUY if side.upper() == "BUY" else mt5.ORDER_TYPE_SELL,
            "price": price or 0.0,
            "deviation": 10,
            "magic": 123456,
            "comment": "arb-auto",
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            raise RuntimeError(f"Order failed: {result.comment} (retcode={result.retcode})")

        # Return the ticket; the caller can poll `mt5.positions_get(ticket=…)`
        return str(result.order)

    # =====================================================================
    # Utility methods
    # =====================================================================

    def cancel_all(self) -> int:
        """Cancel all open orders. Returns count of cancelled orders."""
        positions = mt5.positions_get()
        if not positions:
            return 0

        cancelled = 0
        for pos in positions:
            request = {
                "action": mt5.TRADE_ACTION_CLOSE,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY,
            }
            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                cancelled += 1
        return cancelled

    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get the current open position for a symbol."""
        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            return None

        pos = positions[0]
        return {
            "symbol": pos.symbol,
            "volume": pos.volume,
            "entry_price": pos.price_open,
            "current_price": pos.price_current,
            "profit_loss": pos.profit,
            "direction": "BUY" if pos.type == 0 else "SELL",
        }
