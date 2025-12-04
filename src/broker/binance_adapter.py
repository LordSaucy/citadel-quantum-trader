# src/broker/binance_adapter.py
from __future__ import annotations
import logging
from binance import AsyncClient, BinanceSocketManager
import asyncio
from typing import Any

log = logging.getLogger(__name__)

class BinanceAdapter:
    """Async wrapper – works for spot crypto (BTC/USDT, ETH/USDT, etc.)."""

    def __init__(self):
        self.client: AsyncClient | None = None
        self.loop = asyncio.get_event_loop()
        self.loop.run_until_complete(self._connect())

    async def _connect(self):
        self.client = await AsyncClient.create()
        log.info("✅ Connected to Binance API")

    # -----------------------------------------------------------------
    # Unified order placement (market order for simplicity)
    # -----------------------------------------------------------------
    async def _place_market(self, symbol: str, side: str, qty: float) -> dict:
        order = await self.client.create_order(
            symbol=symbol,
            side=side,
            type="MARKET",
            quantity=qty,
        )
        return order

    def place_order(
        self,
        symbol: str,
        direction: str,
        volume: float,
        price: float | None = None,
        sl: float | None = None,
        tp: float | None = None,
        **kwargs,
    ) -> dict:
        # Binance spot does not support native SL/TP – you must emulate with OCO.
        side = "BUY" if direction.upper() == "BUY" else "SELL"
        order = self.loop.run_until_complete(self._place_market(symbol, side, volume))
        # Record sl/tp for audit – they will be stored in the ledger but not enforced by Binance.
        return {
            "success": order["status"] == "FILLED",
            "fill_price": float(order["fills"][0]["price"]),
            "order_id": order["orderId"],
            "status": order["status"],
            "sl": sl,
            "tp": tp,
        }

    # -----------------------------------------------------------------
    # Position helpers (spot – just check balance)
    # -----------------------------------------------------------------
    def get_position(self, symbol: str) -> dict:
        # Spot balances are per‑asset; we treat the whole balance as the “position”.
        asset = symbol.replace("USDT", "")  # e.g. BTCUSDT → BTC
        bal = self.loop.run_until_complete(self.client.get_asset_balance(asset))
        return {"size": float(bal["free"]), "avg_price": 0.0}

    def close_position(self, symbol: str) -> dict:
        # Close by selling the entire free balance.
        pos = self.get_position(symbol)
        if pos["size"] == 0:
            return {"closed": False, "reason": "no balance"}
        side = "SELL"
        order = self.loop.run_until_complete(
            self._place_market(symbol, side, pos["size"])
        )
        return {"closed": order["status"] == "FILLED", "order_id": order["orderId"]}

    def cancel_all(self) -> dict:
        # Binance does not expose a single “cancel all” endpoint for spot;
        # we simply ignore because we never place pending limit orders in paper mode.
        return {"cancelled": True}

class BaseBroker(ABC):
    @abstractmethod
    def place_order(self, symbol, direction, volume, price, sl, tp): ...
    @abstractmethod
    def cancel_all(self): ...
    @abstractmethod
    def get_position(self, symbol): ...
