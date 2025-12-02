# src/venue_manager.py
import logging
from typing import List, Tuple
from .broker_interface import BrokerInterface
from .mt5_adapter import MT5Adapter
from .ibkr_adapter import IBKRAdapter
from .config_loader import Config

log = logging.getLogger(__name__)

class VenueManager:
    """Handles depth aggregation across multiple venues."""

    def __init__(self):
        cfg = Config().settings
        self.min_multiplier = cfg.get("min_depth_multiplier", 2.0)

        # Build concrete broker objects from the config
        self.venues: List[Tuple[str, BrokerInterface]] = []
        for v in cfg.get("venues", []):
            typ = v["type"].lower()
            if typ == "mt5":
                adapter = MT5Adapter(v)          # passes the whole dict (incl. vault_path)
            elif typ == "ibkr":
                adapter = IBKRAdapter(v)
            else:
                raise ValueError(f"Unsupported venue type: {typ}")
            self.venues.append((v["name"], adapter))

        if not self.venues:
            raise RuntimeError("No venues configured – Multi‑Venue Sync cannot run.")

    async def aggregate_depth(self, symbol: str, depth: int = 20) -> dict:
        """
        Queries every venue for depth and returns a dict:
        {
            "bid_volume": total_bid_qty,
            "ask_volume": total_ask_qty,
            "depth_ok": bool   # True if both sides >= min_multiplier * required_lot
        }
        """
        total_bid = 0.0
        total_ask = 0.0

        # NOTE: we use the synchronous Docker‑SDK in the existing codebase,
        # so we keep this method synchronous for now.
        for name, broker in self.venues:
            try:
                depth_book = broker.get_market_depth(symbol, depth=depth)
                # depth_book is a list of dicts: {"price":…, "volume":…, "side":"bid"/"ask"}
                for lvl in depth_book:
                    if lvl["side"] == "bid":
                        total_bid += lvl["volume"]
                    else:
                        total_ask += lvl["volume"]
            except Exception as exc:
                log.warning(f"Venue {name} depth fetch failed: {exc}")

        return {"bid_volume": total_bid, "ask_volume": total_ask}

    def meets_minimum(self, required_lot: float, agg: dict) -> bool:
        """True if BOTH bid and ask side have at least multiplier × lot."""
        min_needed = self.min_multiplier * required_lot
        return agg["bid_volume"] >= min_needed and agg["ask_volume"] >= min_needed
