# src/broker/ibkr_adapter.py
from __future__ import annotations
from abc import ABC, abstractmethod
from ib_insync import IB, Contract, util
import logging
from typing import Any

log = logging.getLogger(__name__)

class IBKRAdapter:
    """Thin wrapper around ib_insync – works for stocks, futures, and CFDs."""
    def __init__(self):
        self.ib = IB()
        # Assumes a local IB Gateway/TWS is running (Docker‑compose can spin one up)
        self.ib.connect("127.0.0.1", 7497, clientId=42)
        log.info("✅ Connected to IBKR gateway")

    # -----------------------------------------------------------------
    # Helper – build a generic contract from a symbol string
    # -----------------------------------------------------------------
    @staticmethod
    def _contract(symbol: str) -> Contract:
        # Very basic heuristic – you can extend with a lookup table.
        if symbol.endswith("_FUT"):
            # Futures – strip suffix, assume CME front month
            base = symbol.replace("_FUT", "")
            return Contract(symbol=base, secType="FUT", exchange="CME", currency="USD")
        elif symbol.isupper() and len(symbol) <= 5:
            # Assume equity
            return Contract(symbol=symbol, secType="STK", exchange="SMART", currency="USD")
        else:
            # Default to CFD (used for many indices/metals)
            return Contract(symbol=symbol, secType="CFD", exchange="SMART", currency="USD")

    # -----------------------------------------------------------------
    # Unified order placement
    # -----------------------------------------------------------------
    def place_order(
        self,
        symbol: str,
        direction: str,
        volume: float,
        price: float,
        sl: float,
        tp: float,
        tif: str = "GTC",
    ) -> dict:
        contract = self._contract(symbol)
        action = "BUY" if direction.upper() == "BUY" else "SELL"
        order = util.limitOrder(action, volume, price, tif=tif, outsideRth=False)

        # Attach attached stop‑loss / take‑profit as bracket orders
        if sl is not None:
            sl_order = util.stopOrder("SELL" if action == "BUY" else "BUY",
                                      volume, sl, tif=tif)
            order = util.bracketOrder(order, sl_order, None)  # TP will be added next
        if tp is not None:
            tp_order = util.limitOrder("SELL" if action == "BUY" else "BUY",
                                      volume, tp, tif=tif)
            if hasattr(order, "children"):
                order.children.append(tp_order)
            else:
                order = util.bracketOrder(order, None, tp_order)

        trade = self.ib.placeOrder(contract, order)
        self.ib.sleep(0.1)  # give IB a moment to respond
        fill_price = trade.fills[-1].price if trade.fills else None
        return {
            "success": trade.isDone(),
            "fill_price": fill_price,
            "order_id": trade.order.permId,
            "status": trade.orderStatus.status,
        }

    # -----------------------------------------------------------------
    # Position helpers
    # -----------------------------------------------------------------
    def get_position(self, symbol: str) -> dict:
        contract = self._contract(symbol)
        pos = self.ib.positions()
        for p in pos:
            if p.contract.conId == contract.conId:
                return {"size": p.position, "avg_price": p.avgCost}
        return {"size": 0, "avg_price": 0.0}

    def close_position(self, symbol: str) -> dict:
        pos = self.get_position(symbol)
        if pos["size"] == 0:
            return {"closed": False, "reason": "no open position"}
        contract = self._contract(symbol)
        action = "SELL" if pos["size"] > 0 else "BUY"
        order = util.marketOrder(action, abs(pos["size"]))
        trade = self.ib.placeOrder(contract, order)
        self.ib.sleep(0.1)
        return {"closed": trade.isDone(), "order_id": trade.order.permId}

    def cancel_all(self) -> dict:
        self.ib.cancelOrders()
        return {"cancelled": True}

class BaseBroker(ABC):
    @abstractmethod
    def place_order(self, symbol, direction, volume, price, sl, tp): ...
    @abstractmethod
    def cancel_all(self): ...
    @abstractmethod
    def get_position(self, symbol): ...
