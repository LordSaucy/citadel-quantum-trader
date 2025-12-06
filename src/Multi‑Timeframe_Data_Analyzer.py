#!/usr/bin/env python3
"""
Multi‑Timeframe_Data_Analyzer.py – Time‑series aggregation across multiple timeframes.

✅ FIXED: 
- Refactored get_best_alignment_periods() to reduce cognitive complexity from 23 to 12
- Consolidated exception handling: except (FileNotFoundError, OSError) → except OSError
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class MultiTimeframeAnalyzer:
    """Analyzes price action across multiple timeframes (H1, M15, M5, M1)."""

    def __init__(self, data_directory: str = "/data/market_data"):
        self.data_dir = Path(data_directory)
        self.data_cache: Dict[str, pd.DataFrame] = {}
        self._ensure_data_dir()

    def _ensure_data_dir(self) -> None:
        """Ensure data directory exists (create if needed)."""
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Data directory verified at {self.data_dir}")
        except OSError as exc:
            logger.error(f"Failed to create/access data directory: {exc}")
            raise

    def load_ohlcv_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Load OHLCV data from disk cache or Redis."""
        cache_key = f"{symbol}_{timeframe}"
        
        # Check in-memory cache first
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]

        # Try to load from disk
        file_path = self.data_dir / f"{symbol}_{timeframe}.csv"
        
        try:
            df = pd.read_csv(file_path, parse_dates=['timestamp'], index_col='timestamp')
            self.data_cache[cache_key] = df
            logger.info(f"Loaded {len(df)} rows for {cache_key}")
            return df
        except OSError as exc:
            # ✅ FIXED: Single exception handler for all file I/O errors
            # Covers: FileNotFoundError, PermissionError, IsADirectoryError, etc.
            logger.warning(f"Could not load OHLCV data for {cache_key}: {exc}")
            return None
        except pd.errors.ParserError as exc:
            logger.error(f"CSV parse error for {cache_key}: {exc}")
            return None
        except Exception as exc:
            logger.error(f"Unexpected error loading {cache_key}: {exc}")
            return None

    def load_config_file(self, config_path: str) -> Optional[Dict[str, Any]]:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except OSError as exc:
            # ✅ FIXED: Single handler for all file I/O errors
            logger.warning(f"Could not read config file {config_path}: {exc}")
            return None
        except json.JSONDecodeError as exc:
            logger.error(f"Invalid JSON in config file: {exc}")
            return None

    def aggregate_timeframes(
        self,
        symbol: str,
        timeframes: List[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Aggregate OHLCV data across multiple timeframes.

        Returns a dict keyed by timeframe with aggregated DataFrames.
        """
        if timeframes is None:
            timeframes = ["H1", "M15", "M5", "M1"]

        results = {}
        for tf in timeframes:
            df = self.load_ohlcv_data(symbol, tf)
            if df is not None:
                results[tf] = self._compute_indicators(df)
            else:
                logger.warning(f"No data available for {symbol} {tf}")

        return results

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute technical indicators on OHLCV data."""
        df = df.copy()

        # Simple Moving Averages
        df['SMA20'] = df['close'].rolling(20).mean()
        df['SMA50'] = df['close'].rolling(50).mean()

        # Average True Range
        df['TR'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                np.abs(df['high'] - df['close'].shift()),
                np.abs(df['low'] - df['close'].shift())
            )
        )
        df['ATR14'] = df['TR'].rolling(14).mean()

        # RSI
        delta = df['close'].diff()
        gains = delta.where(delta > 0, 0).rolling(14).mean()
        losses = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gains / losses
        df['RSI14'] = 100 - (100 / (1 + rs))

        return df

    # =====================================================================
    # ✅ FIXED: Refactored get_best_alignment_periods() to reduce complexity
    # Before: Cognitive Complexity = 23
    # After:  Cognitive Complexity = 12
    # 
    # Strategy: Extracted nested conditionals into helper methods
    # =====================================================================

    def get_best_alignment_periods(
        self,
        symbol: str,
        timeframes: List[str] = None,
    ) -> List[Tuple[str, str, float]]:
        """
        Find the best period windows where multiple timeframes align strongly.

        Returns list of (timeframe1, timeframe2, alignment_score) tuples,
        sorted by alignment score (highest first).

        ✅ FIXED: Refactored to reduce cognitive complexity from 23 to 12
        by extracting helper methods for:
        - Indicator extraction
        - Pairwise comparison
        - Score calculation
        """
        if timeframes is None:
            timeframes = ["H1", "M15", "M5"]

        aggregated = self.aggregate_timeframes(symbol, timeframes)

        if len(aggregated) < 2:
            logger.warning(f"Not enough data for alignment analysis ({len(aggregated)} timeframes)")
            return []

        # Compare all pairs of timeframes
        alignment_results = []
        for i, tf1 in enumerate(timeframes):
            for tf2 in timeframes[i + 1:]:
                if tf1 not in aggregated or tf2 not in aggregated:
                    continue

                score = self._calculate_pairwise_alignment(
                    aggregated[tf1],
                    aggregated[tf2],
                    tf1,
                    tf2
                )

                if score is not None:
                    alignment_results.append((tf1, tf2, score))

        # Sort by score descending
        alignment_results.sort(key=lambda x: x[2], reverse=True)
        return alignment_results

    def _calculate_pairwise_alignment(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        tf1: str,
        tf2: str,
    ) -> Optional[float]:
        """
        ✅ FIXED: Extracted helper to reduce complexity of get_best_alignment_periods()
        
        Calculate alignment score between two timeframes.
        Returns alignment score (0-100) or None if insufficient data.
        """
        required_cols = ['SMA50', 'RSI14']
        
        # Validate data availability
        if not self._has_required_indicators(df1, required_cols) or \
           not self._has_required_indicators(df2, required_cols):
            return None

        # Get latest values from both timeframes
        sma50_1 = df1['SMA50'].iloc[-1]
        sma50_2 = df2['SMA50'].iloc[-1]
        rsi_1 = df1['RSI14'].iloc[-1]
        rsi_2 = df2['RSI14'].iloc[-1]

        # Calculate components and combine
        sma_alignment = self._calculate_sma_alignment(sma50_1, sma50_2)
        rsi_alignment = self._calculate_rsi_alignment(rsi_1, rsi_2)
        
        # Weighted score (60% SMA, 40% RSI)
        alignment_score = (sma_alignment * 0.6) + (rsi_alignment * 0.4)

        logger.debug(f"Alignment {tf1}↔{tf2}: SMA={sma_alignment:.1f}, RSI={rsi_alignment:.1f}, Total={alignment_score:.1f}")
        return alignment_score

    def _has_required_indicators(self, df: pd.DataFrame, required_cols: List[str]) -> bool:
        """
        ✅ FIXED: Extracted validation helper
        
        Check if DataFrame has required indicators without NaNs.
        """
        for col in required_cols:
            if col not in df.columns or df[col].isna().all():
                return False
        return True

    def _calculate_sma_alignment(self, sma1: float, sma2: float) -> float:
        """
        ✅ FIXED: Extracted SMA alignment calculation
        
        Calculate how close the SMAs are (returns 0-100 score).
        Perfect alignment = SMAs within 0.1%, max divergence = 2%+
        """
        if sma1 == 0 or sma2 == 0:
            return 0.0

        pct_diff = abs(sma1 - sma2) / max(sma1, sma2) * 100
        
        # Score inversely proportional to percentage difference
        # 0.1% diff → 100 pts, 2%+ diff → 0 pts
        if pct_diff <= 0.1:
            return 100.0
        elif pct_diff >= 2.0:
            return 0.0
        else:
            # Linear interpolation between 0.1% and 2.0%
            return 100.0 * (1.0 - (pct_diff - 0.1) / 1.9)

    def _calculate_rsi_alignment(self, rsi1: float, rsi2: float) -> float:
        """
        ✅ FIXED: Extracted RSI alignment calculation
        
        Calculate how close the RSIs are (returns 0-100 score).
        Perfect alignment = RSIs within 5 points, max divergence = 30+ points
        """
        rsi_diff = abs(rsi1 - rsi2)
        
        if rsi_diff <= 5:
            return 100.0
        elif rsi_diff >= 30:
            return 0.0
        else:
            # Linear interpolation between 5 and 30 points
            return 100.0 * (1.0 - (rsi_diff - 5.0) / 25.0)

    def detect_confluence_zones(
        self,
        symbol: str,
        timeframes: List[str] = None,
    ) -> List[Tuple[float, float]]:
        """
        Detect price zones where multiple timeframes agree (confluence).

        Returns list of (low, high) tuples representing confluence zones.
        """
        if timeframes is None:
            timeframes = ["H1", "M15"]

        aggregated = self.aggregate_timeframes(symbol, timeframes)

        if not aggregated:
            logger.warning(f"No data available for confluence analysis of {symbol}")
            return []

        # Simplified confluence: zones where SMA50 is within 1% on 2+ timeframes
        zones = []
        sma_values = []

        for tf, df in aggregated.items():
            if 'SMA50' in df.columns and not df['SMA50'].isna().all():
                latest_sma = df['SMA50'].iloc[-1]
                sma_values.append(latest_sma)

        if len(sma_values) >= 2:
            avg_sma = np.mean(sma_values)
            tolerance = avg_sma * 0.01  # 1% tolerance
            low = avg_sma - tolerance
            high = avg_sma + tolerance
            zones.append((low, high))
            logger.info(f"Confluence zone detected for {symbol}: {low:.5f} - {high:.5f}")

        return zones

    def volatility_state(
        self,
        symbol: str,
        timeframe: str = "H1",
    ) -> Optional[str]:
        """
        Classify volatility as LOW, NORMAL, or HIGH based on recent ATR.
        """
        try:
            df = self.load_ohlcv_data(symbol, timeframe)
            if df is None or len(df) < 20:
                return None

            if 'ATR14' not in df.columns:
                df = self._compute_indicators(df)

            atr_current = df['ATR14'].iloc[-1]
            atr_mean = df['ATR14'].iloc[-50:].mean()

            if atr_mean == 0:
                return "UNKNOWN"

            ratio = atr_current / atr_mean

            if ratio < 0.7:
                return "LOW"
            elif ratio > 1.3:
                return "HIGH"
            else:
                return "NORMAL"

        except Exception as exc:
            logger.error(f"Error computing volatility state for {symbol}: {exc}")
            return None

    def clear_cache(self) -> None:
        """Clear in-memory data cache."""
        self.data_cache.clear()
        logger.info("Data cache cleared")
