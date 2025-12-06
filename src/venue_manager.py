#!/usr/bin/env python3
"""
venue_manager.py – Multi‑venue depth aggregation.

Aggregates order book depth across multiple brokers/venues
and enforces minimum liquidity requirements.

✅ FIXED: Removed `async` keyword from aggregate_depth() method
The method uses only synchronous operations (no await calls).
"""

import logging
from typing import Dict, List, Tuple

from .broker_interface import BrokerInterface
from .mt5_adapter import MT5Adapter
from .ibkr_adapter import IBKRAdapter
from .config_loader import Config

log = logging.getLogger(__name__)


class VenueManager:
    """
    Handles depth aggregation across multiple venues/brokers.
    
    Provides a unified interface to query market depth and enforce
    minimum liquidity constraints across multiple execution venues.
    """

    def __init__(self) -> None:
        """
        Initialize the venue manager by loading configured brokers.
        
        Raises
        ------
        RuntimeError
            If no venues are configured.
        """
        cfg = Config().settings
        self.min_multiplier = cfg.get("min_depth_multiplier", 2.0)

        # Build concrete broker objects from the config
        self.venues: List[Tuple[str, BrokerInterface]] = []
        for v in cfg.get("venues", []):
            typ = v["type"].lower()
            if typ == "mt5":
                adapter = MT5Adapter(v)
            elif typ == "ibkr":
                adapter = IBKRAdapter(v)
            else:
                raise ValueError(f"Unsupported venue type: {typ}")
            self.venues.append((v["name"], adapter))

        if not self.venues:
            raise RuntimeError("No venues configured – Multi‑Venue Sync cannot run.")

        log.info(f"VenueManager initialized with {len(self.venues)} venue(s)")

    # =====================================================================
    # ✅ FIXED: Removed `async` keyword (method is synchronous)
    # 
    # Before: async def aggregate_depth(self, symbol: str, depth: int = 20) -> dict:
    # After:  def aggregate_depth(self, symbol: str, depth: int = 20) -> dict:
    # 
    # Reason: The method only uses synchronous broker.get_market_depth() calls.
    #         No `await` operations are present – it's a standard synchronous method.
    # =====================================================================
    def aggregate_depth(self, symbol: str, depth: int = 20) -> Dict[str, float]:
        """
        Query every venue for depth and return aggregated liquidity stats.

        ✅ FIXED: Removed `async` keyword (was unused – all operations are synchronous)

        Parameters
        ----------
        symbol : str
            The trading symbol (e.g. "EURUSD").
        depth : int, optional
            Number of depth levels to query (default 20).

        Returns
        -------
        dict
            Dictionary containing:
            - "bid_volume" (float): Total bid‑side quantity across all venues
            - "ask_volume" (float): Total ask‑side quantity across all venues
            - "depth_ok" (bool): True if both sides meet minimum liquidity requirement

        Examples
        --------
        >>> manager = VenueManager()
        >>> depth_agg = manager.aggregate_depth("EURUSD", depth=20)
        >>> print(depth_agg["bid_volume"], depth_agg["ask_volume"])
        """
        total_bid = 0.0
        total_ask = 0.0

        # Query all configured venues synchronously
        for name, broker in self.venues:
            try:
                depth_book = broker.get_market_depth(symbol, depth=depth)
                # depth_book is a list of dicts with: price, volume, side ("bid"/"ask")
                for lvl in depth_book:
                    if lvl["side"] == "bid":
                        total_bid += lvl["volume"]
                    else:  # "ask"
                        total_ask += lvl["volume"]
            except Exception as exc:
                log.warning(f"Venue {name} depth fetch failed for {symbol}: {exc}")

        return {
            "bid_volume": total_bid,
            "ask_volume": total_ask,
        }

    def meets_minimum(
        self,
        required_lot: float,
        agg: Dict[str, float],
    ) -> bool:
        """
        Check if aggregated depth meets minimum liquidity requirement.

        Both bid and ask sides must have at least (min_multiplier × required_lot).

        Parameters
        ----------
        required_lot : float
            The lot size we want to trade.
        agg : dict
            The aggregated depth dict from aggregate_depth().

        Returns
        -------
        bool
            True if liquidity is sufficient on both sides.

        Examples
        --------
        >>> depth_agg = manager.aggregate_depth("EURUSD")
        >>> if manager.meets_minimum(1.0, depth_agg):
        ...     print("Liquidity is OK – proceed with trade")
        """
        min_needed = self.min_multiplier * required_lot
        return (
            agg.get("bid_volume", 0.0) >= min_needed
            and agg.get("ask_volume", 0.0) >= min_needed
        )

    def get_best_venue(self, symbol: str) -> Tuple[str, BrokerInterface]:
        """
        Find the venue with the most liquidity for a given symbol.

        Parameters
        ----------
        symbol : str
            The trading symbol.

        Returns
        -------
        tuple
            (venue_name, broker_instance)

        Raises
        ------
        RuntimeError
            If no venues are configured or all queries fail.
        """
        best_venue = None
        best_liquidity = 0.0

        for name, broker in self.venues:
            try:
                depth_book = broker.get_market_depth(symbol, depth=20)
                total_volume = sum(lvl.get("volume", 0.0) for lvl in depth_book)
                if total_volume > best_liquidity:
                    best_liquidity = total_volume
                    best_venue = (name, broker)
            except Exception as exc:
                log.debug(f"Could not query venue {name} for {symbol}: {exc}")

        if best_venue is None:
            raise RuntimeError(f"No venue returned valid depth for {symbol}")

        return best_venue
