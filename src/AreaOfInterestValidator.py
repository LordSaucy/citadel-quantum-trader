#!/usr/bin/env python3
"""
Area of Interest (AOI) Validator

Finds support / resistance zones with a minimum of 3 “touches”.
Validates that the zone lies inside the structure bounds supplied by the
Multi‑Timeframe Structure module and that the current price is *at* the AOI
(≤ 0.2 % distance).

AOI Rules
---------
1. Minimum 3 touches required.
2. Must be inside the HH/HL (bullish) or LH/LL (bearish) bounds.
3. More touches → higher strength (5+ touches = 100 % strength).
4. Price must be AT the AOI (≤ 0.2 % distance) to trade.
"""

# ----------------------------------------------------------------------
# Standard library
# ----------------------------------------------------------------------
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from .aoi_controller import aoi_controller   # <-- NEW import


# ----------------------------------------------------------------------
# Third‑party
# ----------------------------------------------------------------------
import MetaTrader5 as mt5
import numpy as np

# ----------------------------------------------------------------------
# Logging configuration (uses the global logger of the application)
# ----------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass
class AreaOfInterest:
    """Support or resistance zone with multiple touches."""
    price: float                     # Representative price of the zone
    touches: int                     # Number of swing points inside the zone
    type: str                        # "SUPPORT" or "RESISTANCE"
    first_touch: datetime            # Timestamp of the earliest touch
    last_touch: datetime             # Timestamp of the latest touch
    strength: float                  # 0‑1 based on touch count (5+ touches = 1.0)
    price_range: Tuple[float, float] # (low, high) of the clustered zone


@dataclass
class AOIAnalysis:
    """Result of a full AOI analysis."""
    aoi_found: bool
    aoi: Optional[AreaOfInterest]
    at_aoi: bool                     # True if current price is within AT_AOI_TOLERANCE
    distance_from_aoi: float         # Relative distance (|price‑aoi| / aoi)
    aoi_score: float                 # 0‑100 composite score
    reason: str                      # Human‑readable explanation when aoi_found=False


# ----------------------------------------------------------------------
# Main validator class
# ----------------------------------------------------------------------
class AreaOfInterestValidator:
    """
    Finds and validates Areas of Interest (support / resistance zones).

    The algorithm:
        1️⃣  Scan the last N candles (default 1500 @ H4) for swing highs / lows.
        2️⃣  Keep only those that lie inside the supplied structure bounds.
        3️⃣  Cluster swing points that are within PRICE_TOLERANCE (≈ 0.3 %).
        4️⃣  Keep clusters with at least MIN_TOUCHES points.
        5️⃣  Choose the cluster closest to the current price.
        6️⃣  Compute a score based on touch count and proximity.
        7️⃣  Decide whether the price is “at” the AOI (≤ AT_AOI_TOLERANCE).
    """

    # ------------------------------------------------------------------
    # Tunable constants (feel free to expose them via a config file later)
    # ------------------------------------------------------------------
    @property
    def MIN_TOUCHES(self) -> int:
        return int(aoi_controller.get("min_touches"))

    @property
    def PRICE_TOLERANCE(self) -> float:
        return aoi_controller.get("price_tolerance")

    @property
    def AT_AOI_TOLERANCE(self) -> float:
        return aoi_controller.get("at_aoi_tolerance")

    # ------------------------------------------------------------------
    def __init__(self) -> None:
        """Initialise the validator."""
        self.aoi_cache = {}
        logger.info("Area of Interest Validator initialised")

    # ------------------------------------------------------------------
    # Cache handling (optional – speeds up repeated calls on the same symbol)
    # ------------------------------------------------------------------
    def _load_cache(self) -> None:
        if self._cache_path.is_file():
            try:
                with open(self._cache_path, "r") as f:
                    raw = json.load(f)
                # Re‑hydrate objects (lightweight – only needed fields)
                for sym, data in raw.items():
                    if data["aoi"] is not None:
                        aoi_data = data["aoi"]
                        aoi = AreaOfInterest(
                            price=aoi_data["price"],
                            touches=aoi_data["touches"],
                            type=aoi_data["type"],
                            first_touch=datetime.fromisoformat(aoi_data["first_touch"]),
                            last_touch=datetime.fromisoformat(aoi_data["last_touch"]),
                            strength=aoi_data["strength"],
                            price_range=tuple(aoi_data["price_range"]),
                        )
                    else:
                        aoi = None
                    self._cache[sym] = AOIAnalysis(
                        aoi_found=data["aoi_found"],
                        aoi=aoi,
                        at_aoi=data["at_aoi"],
                        distance_from_aoi=data["distance_from_aoi"],
                        aoi_score=data["aoi_score"],
                        reason=data["reason"],
                    )
                logger.info("AOI cache loaded from disk")
            except Exception as exc:   # pragma: no cover
                logger.error(f"Failed to load AOI cache: {exc}")

    def _persist_cache(self) -> None:
        try:
            serialisable = {}
            for sym, analysis in self._cache.items():
                serialisable[sym] = {
                    "aoi_found": analysis.aoi_found,
                    "aoi": None if analysis.aoi is None else {
                        "price": analysis.aoi.price,
                        "touches": analysis.aoi.touches,
                        "type": analysis.aoi.type,
                        "first_touch": analysis.aoi.first_touch.isoformat(),
                        "last_touch": analysis.aoi.last_touch.isoformat(),
                        "strength": analysis.aoi.strength,
                        "price_range": list(analysis.aoi.price_range),
                    },
                    "at_aoi": analysis.at_aoi,
                    "distance_from_aoi": analysis.distance_from_aoi,
                    "aoi_score": analysis.aoi_score,
                    "reason": analysis.reason,
                }
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._cache_path, "w") as f:
                json.dump(serialisable, f, indent=2)
        except Exception as exc:   # pragma: no cover
            logger.error(f"Failed to persist AOI cache: {exc}")

    # ------------------------------------------------------------------
    # Public entry point – the method used by the rest of the bot
    # ------------------------------------------------------------------
    def find_area_of_interest(
        self,
        symbol: str,
        direction: str,
        structure_bounds: Dict,
        current_price: float,
        regime: str,
    ) -> AOIAnalysis:
        """
        Locate a valid Area‑of‑Interest for ``symbol`` in ``direction``.

        Args:
            symbol: Trading symbol (e.g. "EURUSD").
            direction: "BUY" or "SELL".
            structure_bounds: Dict with optional ``lower_bound`` and ``upper_bound``
                              supplied by the Multi‑Timeframe Structure module.
            current_price: Latest market price (mid‑quote).
            regime: Current market regime – "BULLISH" or "BEARISH".

        Returns:
            AOIAnalysis – a rich result object.
        """
        cache_key = f"{symbol}:{direction}:{regime}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        logger.info(f"Searching for {direction} AOI on {symbol} @ {current_price:.5f}")

        # ------------------------------------------------------------------
        # 1️⃣  Regime‑direction sanity check
        # ------------------------------------------------------------------
        if direction == "BUY" and regime != "BULLISH":
            return self._no_aoi_result(
                current_price,
                f"Cannot BUY in {regime} regime",
            )
        if direction == "SELL" and regime != "BEARISH":
            return self._no_aoi_result(
                current_price,
                f"Cannot SELL in {regime} regime",
            )

        # ------------------------------------------------------------------
        # 2️⃣  Pull recent H4 candles (enough history for reliable swing detection)
        # ------------------------------------------------------------------
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H4, 0, 1500)
        if rates is None or len(rates) < 100:
            return self._no_aoi_result(current_price, "Insufficient market data")

        # ------------------------------------------------------------------
        # 3️⃣  Resolve structure bounds (fallback to recent extremes if missing)
        # ------------------------------------------------------------------
        lower_bound = structure_bounds.get("lower_bound")
        upper_bound = structure_bounds.get("upper_bound")

        recent_slice = rates[-200:]                     # last ~200 H4 bars ≈ 33 days
        if lower_bound is None:
            lower_bound = float(np.min(recent_slice["low"]))
        if upper_bound is None:
            upper_bound = float(np.max(recent_slice["high"]))

        # ------------------------------------------------------------------
        # 4️⃣  Find candidate AOI (support for BUY, resistance for SELL)
        # ------------------------------------------------------------------
        if direction == "BUY":
            aoi = self._find_support_zone(rates, lower_bound, upper_bound, current_price)
        else:
            aoi = self._find_resistance_zone(rates, lower_bound, upper_bound, current_price)

        if aoi is None:
            result = self._no_aoi_result(
                current_price,
                f"No {direction} AOI found with ≥{self.MIN_TOUCHES} touches",
            )
            self._cache[cache_key] = result
            self._persist_cache()
            return result

        # ------------------------------------------------------------------
        # 5️⃣  Proximity & scoring
        # ------------------------------------------------------------------
        at_aoi = self._is_at_aoi(aoi, current_price)
        distance = abs(current_price - aoi.price) / aoi.price
        aoi_score = self._calculate_aoi_score(aoi, at_aoi, distance)

        result = AOIAnalysis(
            aoi_found=True,
            aoi=aoi,
            at_aoi=at_aoi,
            distance_from_aoi=distance,
            aoi_score=aoi_score,
            reason=(
                f"{aoi.type} with {aoi.touches} touches "
                f"(strength {aoi.strength:.0%})"
            ),
        )

        # Cache the result for fast subsequent calls
        self._cache[cache_key] = result
        self._persist_cache()
        return result

    # ------------------------------------------------------------------
    # 6️⃣  Support‑zone finder (BUY side)
    # ------------------------------------------------------------------
    def _find_support_zone(
        self,
        rates: np.ndarray,
        lower_bound: float,
        upper_bound: float,
        current_price: float,
    ) -> Optional[AreaOfInterest]:
        """Locate a support zone (≥ MIN_TOUCHES) inside the supplied bounds."""
        swing_lows: List[Dict] = []

        # Scan for swing lows that respect the bounds
        for i in range(10, len(rates) - 5):
            low = float(rates[i]["low"])
            if not (lower_bound <= low <= upper_bound):
                continue

            window = rates[i - 5 : i + 6]               # 11‑bar window
            if low == float(np.min(window["low"])):     # genuine swing low
                swing_lows.append(
                    {
                        "price": low,
                        "time": datetime.fromtimestamp(rates[i]["time"]),
                        "index": i,
                    }
                )

        if len(swing_lows) < self.MIN_TOUCHES:
            logger.debug(
                f"Support scan: only {len(swing_lows)} swing lows (need {self.MIN_TOUCHES})"
            )
            return None

        # Cluster nearby lows into zones
        zones = self._cluster_price_levels(swing_lows, self.PRICE_TOLERANCE)

        # Pick the best zone close to the current price
        return self._select_best_zone(zones, current_price, "SUPPORT")

    # ------------------------------------------------------------------
    # 7️⃣  Resistance‑zone finder (SELL side)
    # ------------------------------------------------------------------
    def _find_resistance_zone(
        self,
        rates: np.ndarray,
        lower_bound: float,
        upper_bound: float,
        current_price: float,
    ) -> Optional[AreaOfInterest]:
        """Locate a resistance zone (≥ MIN_TOUCHES) inside the supplied bounds."""
        swing_highs: List[Dict] = []

        for i in range(10, len(rates) - 5):
            high = float(rates[i]["high"])
            if not (lower_bound <= high <= upper_bound):
                continue

            window = rates[i - 5 : i + 6]
            if high == float(np.max(window["high"])):   # genuine swing high
                swing_highs.append(
                    {
                        "price": high,
                        "time": datetime.fromtimestamp(rates[i]["time"]),
                        "index": i,
                    }
                )

        if len(swing_highs) < self.MIN_TOUCHES:
            logger.debug(
                f"Resistance scan: only {len(swing_highs)} swing highs (need {self.MIN_TOUCHES})"
            )
            return None

        zones = self._cluster_price_levels(swing_highs, self.PRICE_TOLERANCE)
        return self._select_best_zone(zones, current_price, "RESISTANCE")

    # ------------------------------------------------------------------
    # 8️⃣  Helper – cluster price levels that are within ``tolerance``
    # ------------------------------------------------------------------
    def _cluster_price_levels(
        self, levels: List[Dict], tolerance: float
    ) -> Dict[float, List[Dict]]:
        """
        Group price points that lie within ``tolerance`` (relative) of each other.

        Returns a dict ``{zone_price: [touch_dict, …]}`` where ``zone_price`` is
        the average price of the points belonging to that zone.
        """
        if not levels:
            return {}

        zones: Dict[float, List[Dict]] = {}

        for lvl in levels:
            price = lvl["price"]
            found = False
            for zone_price in list(zones.keys()):
                if abs(price - zone_price) / zone_price <= tolerance:
                    zones[zone_price].append(lvl)
                    # Re‑average the zone price
                    avg = np.mean([p["price"] for p in zones[zone_price]])
                    if avg != zone_price:
                        zones[avg] = zones.pop(zone_price)
                    found = True
                    break
            if not found:
                zones[price] = [lvl]

        return zones

    # ------------------------------------------------------------------
    # 9️⃣  Helper – pick the best zone (most touches & closest to price)
    # ------------------------------------------------------------------
    def _select_best_zone(
        self,
        zones: Dict[float, List[Dict]],
        current_price: float,
        zone_type: str,
    ) -> Optional[AreaOfInterest]:
        """
        From the clustered zones choose the one that:

        * has at least ``MIN_TOUCHES`` points,
        * is within 2 % of the current price,
        * maximises a simple score = touches × (1 − distance).

        Returns an ``AreaOfInterest`` instance or ``None``.
        """
        candidates: List[Dict] = []

        for zone_price, touches in zones.items():
            if len(touches) < self.MIN_TOUCHES:
                continue

            distance = abs(current_price - zone_price) / zone_price
            if distance > 0.02:               # ignore far‑away zones
                continue

            # Simple heuristic: more touches + closer price = higher score
            score = len(touches) * (1 - distance * 10)
            candidates.append(
                {
                    "price": zone_price,
                    "touches": touches,
                    "score": score,
                    "distance": distance,
                }
            )

        if not candidates:
            return None

        best = max(candidates, key=lambda x: x["score"])

        # Derive the price range of the zone (min / max of constituent points)
        prices = [t["price"] for t in best["touches"]]
        price_range = (min(prices), max(prices))

        # Strength: 5+ touches → 1.0, otherwise linear scaling
        strength = min(1.0, best["touches"] / 5.0)

        return AreaOfInterest(
            price=best["price"],
            touches=len(best["touches"]),
            type=zone_type,
            first_touch=min(t["time"] for t in best["touches"]),
            last_touch=max(t["time"] for t in best["touches"]),
            strength=strength,
            price_range=price_range,
        )

    # ------------------------------------------------------------------
    # 10️⃣  Helper – is the current price “at” the AOI?
    # ------------------------------------------------------------------
    def _is_at_aoi(self, aoi: AreaOfInterest, current_price: float) -> bool:
        """True if the price lies within AT_AOI_TOLERANCE of the zone price."""
        distance = abs(current_price - aoi.price) / aoi.price
        return distance <= self.AT_AOI_TOLERANCE

   # ------------------------------------------------------------------
    # 11️⃣  Scoring routine – combines touches and proximity
    # ------------------------------------------------------------------
    def _calculate_aoi_score(
        self,
        aoi: AreaOfInterest,
        at_aoi: bool,
        distance: float,
    ) -> float:
        """
        Compute a 0‑100 AOI score.

        Factors:
        • Touch count – more touches = higher base score (3 touches → 60,
          5+ touches → 100).
        • Proximity – the closer the entry price to the AOI, the higher the
          proximity component.
        • “At AOI” bonus – if the price is inside the AT_AOI_TOLERANCE the
          proximity component is maximal (100).

        The final score is a weighted average:
            60 % touch‑score  +  40 % proximity‑score
        """
        # ----- Touch‑score -------------------------------------------------
        touch_score = min(100, aoi.touches * 20)   # 3 touches → 60, 5+ → 100

        # ----- Proximity‑score --------------------------------------------
        if at_aoi:
            proximity_score = 100
        elif distance < 0.005:          # within 0.5 %
            proximity_score = 90
        elif distance < 0.01:           # within 1 %
            proximity_score = 80
        else:
            proximity_score = 70

        # ----- Weighted combination -----------------------------------------
        final_score = touch_score * 0.6 + proximity_score * 0.4
        return final_score

    # ------------------------------------------------------------------
    # 12️⃣  Helper – default result when no AOI is found
    # ------------------------------------------------------------------
    def _no_aoi_result(self, current_price: float, reason: str) -> AOIAnalysis:
        """Return a uniform “no AOI” result."""
        return AOIAnalysis(
            aoi_found=False,
            aoi=None,
            at_aoi=False,
            distance_from_aoi=999.0,
            aoi_score=0.0,
            reason=reason,
        )

# ----------------------------------------------------------------------
# Global singleton – import this from ``src/main.py`` and use it directly
# ----------------------------------------------------------------------
aoi_validator = AreaOfInterestValidator()

   # ------------------------------------------------------------------
    # 13️⃣  Utility – clear the in‑memory cache (and optionally the file)
    # ------------------------------------------------------------------
    def clear_cache(self, purge_file: bool = False) -> None:
        """
        Empty the internal AOI cache.  If ``purge_file`` is True the persisted
        JSON file on disk is also removed.

        This can be handy during back‑testing or when you want to force a
        fresh scan after a structural market change.
        """
        self._cache.clear()
        if purge_file and self._cache_path.is_file():
            try:
                self._cache_path.unlink()
                logger.info("AOI cache file removed from disk")
            except Exception as exc:   # pragma: no cover
                logger.error(f"Failed to delete AOI cache file: {exc}")

    # ------------------------------------------------------------------
    # 14️⃣  Representation helpers (nice printing for debugging)
    # ------------------------------------------------------------------
    def __repr__(self) -> str:   # pragma: no cover
        return f"<AreaOfInterestValidator cache_size={len(self._cache)}>"

# ----------------------------------------------------------------------
# Public symbols for ``from src.aoi_validator import *``
# ----------------------------------------------------------------------
__all__ = [
    "AreaOfInterest",
    "AOIAnalysis",
    "AreaOfInterestValidator",
    "aoi_validator",
]

# ----------------------------------------------------------------------
# Global singleton – import this from ``src/main.py`` and use it directly
# ----------------------------------------------------------------------
aoi_validator = AreaOfInterestValidator()



# ----------------------------------------------------------------------
# aoi_validator.py  (only the top part is shown – insert after imports)
# ----------------------------------------------------------------------
from .aoi_controller import aoi_controller   # <-- NEW import

class AreaOfInterestValidator:
    """
    Finds and validates Areas of Interest (support/resistance zones).
    """

    # ------------------------------------------------------------------
    # Previously hard‑coded constants – now delegated to the controller
    # ------------------------------------------------------------------
    @property
    def MIN_TOUCHES(self) -> int:
        return int(aoi_controller.get("min_touches"))

    @property
    def PRICE_TOLERANCE(self) -> float:
        return aoi_controller.get("price_tolerance")

    @property
    def AT_AOI_TOLERANCE(self) -> float:
        return aoi_controller.get("at_aoi_tolerance")

    # ------------------------------------------------------------------
    def __init__(self) -> None:
        """Initialise the validator."""
        self.aoi_cache = {}
        logger.info("Area of Interest Validator initialised")
