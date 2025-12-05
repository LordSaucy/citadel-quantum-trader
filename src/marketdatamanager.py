#!/usr/bin/env python3
"""
MARKET DATA MANAGER

Centralised market‑data handling for Citadel Quantum Trader.

Features
--------
* Efficient MT5 data fetching with optional on‑disk caching (pickle).
* Automatic cache‑expiry (TTL, default 60 s) and manual clearing.
* Helper methods for common indicators: ATR, EMA, RSI.
* Simple swing‑high / swing‑low market‑structure detection.
* Spread & symbol‑information utilities.
* Thread‑safe for use from multiple bot components.

Author: Lawful Banker
Created: 2024‑11‑26
Version: 2.0 – Production‑Ready
"""

# ----------------------------------------------------------------------
# Standard library
# ----------------------------------------------------------------------
import json
import logging
import pickle
import threading
import time
from datetime import datetime, date, time as dt_time, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ----------------------------------------------------------------------
# Third‑party
# ----------------------------------------------------------------------
import MetaTrader5 as mt5
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Logging configuration (writes to ./logs/market_data.log)
# ----------------------------------------------------------------------
LOG_DIR = Path("./logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "market_data.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Helper – one‑time MT5 initialise / shutdown wrapper
# ----------------------------------------------------------------------
def _ensure_mt5_initialized() -> None:
    """Initialise MT5 once per process; raise on failure."""
    if not mt5.initialize():
        err = mt5.last_error()
        raise RuntimeError(f"MT5 initialise failed – error {err}")
    # Optional: tighten time‑outs (ms) for faster fail‑fast behaviour
    mt5.timeouts(5000, 5000, 5000, 5000, 5000)


def _shutdown_mt5() -> None:
    """Shutdown MT5 – safe to call repeatedly."""
    try:
        mt5.shutdown()
    except Exception:  # pragma: no cover
        pass


# ----------------------------------------------------------------------
# Main class
# ----------------------------------------------------------------------
class MarketDataManager:
    """
    Centralised market‑data management system.

    Handles data fetching, on‑disk caching, indicator calculation and a
    lightweight market‑structure detector.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self, cache_dir: str = "data_cache", cache_ttl: int = 60):
        """
        Initialise the manager.

        Args
        ----
        cache_dir: Directory where pickled cache files are stored.
        cache_ttl: Lifetime of a cached entry in seconds (default 60 s).
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.cache_ttl = cache_ttl
        self._cache: Dict[str, pd.DataFrame] = {}
        self._cache_ts: Dict[str, datetime] = {}

        # A simple lock makes the manager safe for multithreaded callers
        self._lock = threading.RLock()

        logger.info("MarketDataManager initialised (cache_dir=%s, ttl=%ds)",
                    self.cache_dir, self.cache_ttl)

    # ------------------------------------------------------------------
    # Internal cache helpers
    # ------------------------------------------------------------------
    def _cache_key(self, symbol: str, timeframe: int, bars: int) -> str:
        """Deterministic cache key used for both memory and disk storage."""
        return f"{symbol}_{timeframe}_{bars}"

    def _is_cache_valid(self, key: str) -> bool:
        """Return True if the in‑memory entry exists and is younger than TTL."""
        if key not in self._cache or key not in self._cache_ts:
            return False
        age = (datetime.now() - self._cache_ts[key]).total_seconds()
        return age < self.cache_ttl

    def _load_from_disk(self, key: str) -> Optional[pd.DataFrame]:
        """Attempt to read a pickled DataFrame from the cache directory."""
        pkl_path = self.cache_dir / f"{key}.pkl"
        if not pkl_path.is_file():
            return None
        try:
            with pkl_path.open("rb") as fh:
                df = pickle.load(fh)
            logger.debug("Cache hit (disk) for %s", key)
            return df
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to load cache file %s: %s", pkl_path, exc)
            return None

    def _write_to_disk(self, key: str, df: pd.DataFrame) -> None:
        """Persist a DataFrame to disk for later reuse."""
        pkl_path = self.cache_dir / f"{key}.pkl"
        try:
            with pkl_path.open("wb") as fh:
                pickle.dump(df, fh, protocol=pickle.HIGHEST_PROTOCOL)
            logger.debug("Cache written to disk for %s", key)
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to write cache file %s: %s", pkl_path, exc)

    # ------------------------------------------------------------------
    # Public API – OHLC retrieval
    # ------------------------------------------------------------------
    def get_ohlc_data(
        self,
        symbol: str,
        timeframe: int,
        bars: int = 500,
        use_cache: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Retrieve OHLC data for a given symbol/timeframe.

        Parameters
        ----------
        symbol   : MT5 instrument name (e.g. "EURUSD").
        timeframe: MT5 timeframe constant (mt5.TIMEFRAME_H1, etc.).
        bars     : Number of historical bars to pull (default 500).
        use_cache: If True, attempt to serve from in‑memory or disk cache.

        Returns
        -------
        pandas.DataFrame with columns: time, open, high, low, close, tick_volume,
        spread, real_volume.  Returns None on failure.
        """
        key = self._cache_key(symbol, timeframe, bars)

        with self._lock:
            # 1️⃣  Try in‑memory cache
            if use_cache and self._is_cache_valid(key):
                logger.debug("Cache hit (memory) for %s", key)
                return self._cache[key]

            # 2️⃣  Try disk cache (still respects TTL)
            if use_cache:
                df_disk = self._load_from_disk(key)
                if df_disk is not None:
                    # Verify freshness
                    ts_path = self.cache_dir / f"{key}.ts"
                    if ts_path.is_file():
                        try:
                            with ts_path.open("r") as fh:
                                ts_iso = fh.read().strip()
                            ts = datetime.fromisoformat(ts_iso)
                            if (datetime.utcnow() - ts).total_seconds() < self.cache_ttl:
                                # Fresh enough – promote to memory cache
                                self._cache[key] = df_disk
                                self._cache_ts[key] = ts
                                logger.debug("Cache hit (disk, fresh) for %s", key)
                                return df_disk
                        except Exception:  # pragma: no cover
                            pass

            # 3️⃣  Fetch from MT5
            try:
                _ensure_mt5_initialized()
                rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
            finally:
                _shutdown_mt5()

            if not rates or len(rates) == 0:
                logger.error("MT5 returned no data for %s (tf=%s, bars=%s)",
                             symbol, timeframe, bars)
                return None

            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df.set_index("time", inplace=True)

            # 4️⃣  Store in caches
            if use_cache:
                self._cache[key] = df
                self._cache_ts[key] = datetime.utcnow()
                self._write_to_disk(key, df)
                # also write a tiny timestamp file for quick TTL checks
                ts_path = self.cache_dir / f"{key}.ts"
                try:
                    with ts_path.open("w") as fh:
                        fh.write(datetime.utcnow().isoformat())
                except Exception:  # pragma: no cover
                    logger.error("Failed to write timestamp for %s", key)

            return df

    def get_multi_timeframe_data(
        self,
        symbol: str,
        timeframes: List[int],
        bars: int = 500,
        use_cache: bool = True,
    ) -> Dict[int, pd.DataFrame]:
        """
        Retrieve OHLC data for several timeframes in a single call.

        Returns a dict mapping the MT5 timeframe constant to its DataFrame.
        """
        result: Dict[int, pd.DataFrame] = {}
        for tf in timeframes:
            df = self.get_ohlc_data(symbol, tf, bars, use_cache)
            if df is not None:
                result[tf] = df
        return result

    # ------------------------------------------------------------------
    # Indicator helpers
    # ------------------------------------------------------------------
    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Average True Range (ATR).

        Parameters
        ----------
        df    : OHLC DataFrame (must contain high, low, close).
        period: Look‑back period (default 14).

        Returns
        -------
        pandas.Series indexed like ``df`` containing the ATR.
        """
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period, min_periods=1).mean()
        return atr

    @staticmethod
    def calculate_ema(
        df: pd.DataFrame,
        period: int,
        column: str = "close",
    ) -> pd.Series:
        """
        Exponential Moving Average (EMA).

        Parameters
        ----------
        df    : OHLC DataFrame.
        period: EMA period.
        column: Column to smooth (default ``close``).

        Returns
        -------
        pandas.Series with the EMA.
        """
        return df[column].ewm(span=period, adjust=False).mean()

    @staticmethod
    def calculate_rsi(
        df: pd.DataFrame,
        period: int = 14,
        column: str = "close",
    ) -> pd.Series:
        """
        Relative Strength Index (RSI).

        Parameters
        ----------
        df    : OHLC DataFrame.
        period: Look‑back period (default 14).
        column: Price column (default ``close``).

        Returns
        -------
        pandas.Series with the RSI (0‑100 range).
        """
        delta = df[column].diff()
        gain = delta.clip(lower=0).rolling(window=period, min_periods=1).mean()
        loss = -delta.clip(upper=0).rolling(window=period, min_periods=1).mean()
        rs = gain / loss.replace(to_replace=0, method="ffill")
        rsi = 100 - (100 / (1 + rs))
        return rsi

    # ------------------------------------------------------------------
    # Simple market‑structure detector (HH/HL/LH/LL)
    # ------------------------------------------------------------------
    @staticmethod
    def identify_market_structure(df: pd.DataFrame) -> Dict[str, any]:
        """
        Very lightweight swing‑high / swing‑low detector.

        Returns a dict with:
        * ``trend``   – "BULLISH", "BEARISH" or "NEUTRAL".
        * ``swing_highs`` – list of indices (relative to df) that are swing highs.
        * ``swing_lows``  – list of indices that are swing lows.
        """
        highs = df["high"]
        lows = df["low"]
        swing_highs: List[int] = []
        swing_lows: List[int] = []

        # Simple 5‑bar window swing detection
        for i in range(2, len(df) - 2):
            if (
                highs.iloc[i] > highs.iloc[i - 1]
                and highs.iloc[i] > highs.iloc[i - 2]
                and highs.iloc[i] > highs.iloc[i + 1]
                and highs.iloc[i] > highs.iloc[i + 2]
            ):
                swing_highs.append(i)

            if (
                lows.iloc[i] < lows.iloc[i - 1]
                and lows.iloc[i] < lows.iloc[i - 2]
                and lows.iloc[i] < lows.iloc[i + 1]
                and lows.iloc[i] < lows.iloc[i + 2]
            ):
                swing_lows.append(i)

        trend = "NEUTRAL"
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            recent_highs = [highs.iloc[idx] for idx in swing_highs[-2:]]
            recent_lows = [lows.iloc[idx] for idx in swing_lows[-2:]]
            if recent_highs[1] > recent_highs[0] and recent_lows[1] > recent_lows[0]:
                trend = "BULLISH"
            elif recent_highs[1] < recent_highs[0] and recent_lows[1] < recent_lows[0]:
                trend = "BEARISH"

        return {
            "trend": trend,
            "swing_highs": swing_highs,
            "swing_lows": swing_lows,
        }

    # ------------------------------------------------------------------
    # Miscellaneous utilities
    # ------------------------------------------------------------------
    @staticmethod
    def get_current_spread(symbol: str) -> float:
        """
        Return the current spread for *symbol* expressed in **pips**.

        If the symbol cannot be queried, 0.0 is returned.
        """
        try:
            _ensure_mt5_initialized()
            tick = mt5.symbol_info_tick(symbol)
            info = mt5.symbol_info(symbol)
        finally:
            _shutdown_mt5()

        if not tick or not info:
            return 0.0

        spread_points = tick.ask - tick.bid
        # Most FX symbols have 5‑digit pricing → 1 pip = 10 points
        point = info.point
        spread_pips = spread_points / (point * 10)
        return float(spread_pips)

    @staticmethod
    def get_symbol_info(symbol: str) -> Optional[Dict]:
        """
        Return a compact dictionary with the most useful MT5 symbol attributes.

        Keys: point, digits, spread, tick_value, tick_size,
              contract_size, min_lot, max_lot, lot_step.
        """
        try:
            _ensure_mt5_initialized()
            info = mt5.symbol_info(symbol)
        finally:
            _shutdown_mt5()

        if not info:
            return None

        return {
            "point": info.point,
            "digits": info.digits,
            "spread": info.spread,
            "tick_value": info.trade_tick_value,
            "tick_size": info.trade_tick_size,
            "contract_size": info.trade_contract_size,
            "min_lot": info.volume_min,
            "max_lot": info.volume_max,
            "lot_step": info.volume_step,
        }

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------
    def clear_cache(self) -> None:
        """Empty the in‑memory cache and delete all persisted pickle files."""
        with self._lock:
            self._cache.clear()
            self._cache_ts.clear()
            # Remove on‑disk files
            for pkl in self.cache_dir.glob("*.pkl"):
                try:
                    pkl.unlink()
                except Exception:  # pragma: no cover
                    pass
            for ts in self.cache_dir.glob("*.ts"):
                try:
                    ts.unlink()
                except Exception:  # pragma: no cover
                    pass
            logger.info("Market data cache cleared (memory + disk)")

    # ------------------------------------------------------------------
    # Graceful shutdown (optional – call from the main process)
    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        """Placeholder for future resources; currently just logs."""
        logger.info("MarketDataManager shutdown requested")
        # No persistent connections to close – MT5 is opened/closed per call.
# ----------------------------------------------------------------------
# Global singleton – import this from ``src/main.py`` and use it directly
# ----------------------------------------------------------------------
market_data_manager = MarketDataManager()


# ----------------------------------------------------------------------
# Optional quick‑run demo (executed only when this file is run directly)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple sanity‑check that the manager can fetch data for a common symbol
    TEST_SYMBOL = "EURUSD"
    TEST_TF = mt5.TIMEFRAME_H1
    TEST_BARS = 200

    logger.info("=== MarketDataManager demo ===")
    try:
        df = market_data_manager.get_ohlc_data(
            symbol=TEST_SYMBOL,
            timeframe=TEST_TF,
            bars=TEST_BARS,
            use_cache=False,
        )
        if df is not None:
            logger.info(f"Fetched {len(df)} rows for {TEST_SYMBOL} (H1)")
            logger.info(f"Latest candle – O:{df['open'].iloc[-1]:.5f} "
                        f"H:{df['high'].iloc[-1]:.5f} "
                        f"L:{df['low'].iloc[-1]:.5f} "
                        f"C:{df['close'].iloc[-1]:.5f}")

            # Indicator examples
            atr = MarketDataManager.calculate_atr(df, period=14)
            ema = MarketDataManager.calculate_ema(df, period=50)
            rsi = MarketDataManager.calculate_rsi(df, period=14)

            logger.info(f"ATR (last)  : {atr.iloc[-1]:.5f}")
            logger.info(f"EMA50 (last): {ema.iloc[-1]:.5f}")
            logger.info(f"RSI14 (last): {rsi.iloc[-1]:.2f}")

            # Market‑structure demo
            struct = MarketDataManager.identify_market_structure(df)
            logger.info(f"Detected trend: {struct['trend']}")
            logger.info(
                f"Swing highs: {len(struct['swing_highs'])} "
                f"| Swing lows: {len(struct['swing_lows'])}"
            )
        else:
            logger.error("Failed to retrieve OHLC data – check MT5 connection.")
    finally:
        # Clean up any lingering MT5 session (good practice)
        _shutdown_mt5()

TIMEFRAMES = {
    "H4": "240",   # minutes
    "H1": "60",
    "M15": "15"
}

def fetch_multi_tf(symbol: str, lookback: int = 500) -> dict[str, pd.DataFrame]:
    """
    Returns a dict:
        {"H4": df_h4, "H1": df_h1, "M15": df_m15}
    Each df has columns: ['timestamp','open','high','low','close','volume']
    """
    result = {}
    for tf_name, minutes in TIMEFRAMES.items():
        # Assuming you have a function that pulls candles for a given timeframe:
        df = get_candles(symbol, timeframe_minutes=int(minutes), limit=lookback)
        result[tf_name] = df
    return result

def get_candles(symbol, timeframe_minutes, limit):
    
    utc_from = datetime.utcnow() - timedelta(minutes=timeframe_minutes * limit)
    rates = mt5.copy_rates_range(
        symbol,
        timeframe=mt5.TIMEFRAME_M1 * timeframe_minutes,   # MT5 defines multiples of M1
        from_date=utc_from,
        to_date=datetime.utcnow()
    )
    df = pd.DataFrame(rates)
    df['timestamp'] = pd.to_datetime(df['time'], unit='s')
    return df[['timestamp','open','high','low','close','volume']]



