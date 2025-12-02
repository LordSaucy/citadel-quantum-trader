#!/usr/bin/env python3
"""
Candlestick Statistics Analyzer

Analyzes MT5‑generated candlestick statistics and provides optimal
Stop‑Loss / Take‑Profit recommendations.  The analyser is completely
self‑contained, persists its configuration, and can be queried from the
trading engine in real time.

Features
--------
* Loads the CSV file produced by the MQL5 “CandlestickAnalysisReport.mq5”.
* Parses overall, hourly, daily and volatility‑segmented statistics.
* Calculates a dynamic SL distance (pips) based on average range,
  volatility state and a safety buffer.
* Returns a full SL/TP dictionary (prices, pips, R:R, confidence, etc.).
* Offers helper methods for:
    – Best trading hours
    – Position‑size volatility adjustment
    – Human‑readable summary
    – JSON export for downstream consumption
* Persists the raw CSV location and can be re‑loaded on demand.
"""

# ----------------------------------------------------------------------
# Standard library
# ----------------------------------------------------------------------
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# ----------------------------------------------------------------------
# Third‑party
# ----------------------------------------------------------------------
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Main analyser class
# ----------------------------------------------------------------------
class CandlestickAnalyzer:
    """
    Core analyser – loads the CSV, parses it into convenient dictionaries
    and provides SL/TP recommendations based on the current hour and
    volatility regime.
    """

    # ------------------------------------------------------------------
    def __init__(self, csv_path: str = "candlestick_stats.csv"):
        """
        Initialise the analyser.

        Args:
            csv_path: Relative or absolute path to the CSV file that the
                      MT5 script writes to.  If the file is not found the
                      analyser will fall back to a safe default.
        """
        self.csv_path = csv_path
        self.df: Optional[pd.DataFrame] = None

        # Parsed containers
        self.overall_stats: Optional[Dict] = None
        self.hourly_stats: Dict[str, Dict] = {}
        self.daily_stats: Dict[str, Dict] = {}
        self.volatility_stats: Dict[str, Dict] = {}

        # Resolve the MT5 “Files” folder (used when the CSV lives there)
        self.mt5_files_path = self._find_mt5_files_path()

        logger.info("📊 Candlestick Analyzer initialised")
        logger.info(f"   CSV path: {self.csv_path}")

    # ------------------------------------------------------------------
    # 0️⃣  Locate the MT5 “Files” directory (Windows only)
    # ------------------------------------------------------------------
    def _find_mt5_files_path(self) -> Optional[str]:
        """Search common locations for the MT5 “Files” folder."""
        possible_paths = [
            os.path.join(os.environ.get("APPDATA", ""), "MetaQuotes", "Terminal"),
            r"C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal",
            r"C:\Program Files\MetaTrader 5\MQL5\Files",
        ]

        for base_path in possible_paths:
            if not os.path.isdir(base_path):
                continue

            # Terminal directories contain a random GUID folder – dive in
            if "Terminal" in base_path:
                for folder in os.listdir(base_path):
                    candidate = os.path.join(base_path, folder, "MQL5", "Files")
                    if os.path.isdir(candidate):
                        return candidate
            else:
                return base_path

        return None

    # ------------------------------------------------------------------
    # 1️⃣  Locate the CSV file (multiple fallback locations)
    # ------------------------------------------------------------------
    def _find_csv_file(self) -> Optional[str]:
        """Return the absolute path to the CSV, or None if not found."""
        # 1️⃣  Absolute / relative path as‑given
        if os.path.isfile(self.csv_path):
            return self.csv_path

        # 2️⃣  Inside the MT5 “Files” folder
        if self.mt5_files_path:
            candidate = os.path.join(self.mt5_files_path, self.csv_path)
            if os.path.isfile(candidate):
                return candidate

        # 3️⃣  Current working directory (useful when running locally)
        cwd_candidate = os.path.join(os.getcwd(), self.csv_path)
        if os.path.isfile(cwd_candidate):
            return cwd_candidate

        return None

    # ------------------------------------------------------------------
    # 2️⃣  Load the CSV into a pandas DataFrame
    # ------------------------------------------------------------------
    def load_statistics(self) -> bool:
        """
        Load candlestick statistics from the CSV file.

        Returns:
            True  – CSV successfully loaded and parsed
            False – CSV not found or parsing error
        """
        csv_file = self._find_csv_file()
        if not csv_file:
            logger.error(f"❌ Candlestick stats CSV not found: {self.csv_path}")
            return False

        try:
            self.df = pd.read_csv(csv_file)
            self._parse_statistics()
            logger.info("✅ Loaded candlestick statistics")
            logger.info(f"   Categories: {len(self.df['category'].unique())}")
            logger.info(f"   Total rows: {len(self.df)}")
            return True
        except Exception as exc:   # pragma: no cover
            logger.error(f"❌ Error loading statistics: {exc}")
            return False

    # ------------------------------------------------------------------
    # 3️⃣  Split the raw DataFrame into handy dictionaries
    # ------------------------------------------------------------------
    def _parse_statistics(self) -> None:
        """Populate overall, hourly, daily and volatility dictionaries."""
        if self.df is None:
            return

        # ----- Overall ----------------------------------------------------
        overall = self.df[self.df["category"] == "OVERALL"]
        if not overall.empty:
            self.overall_stats = overall.iloc[0].to_dict()

        # ----- Hourly -----------------------------------------------------
        hourly = self.df[self.df["category"] == "HOUR"]
        for _, row in hourly.iterrows():
            self.hourly_stats[row["subcategory"]] = row.to_dict()

        # ----- Daily ------------------------------------------------------
        daily = self.df[self.df["category"] == "DAY"]
        for _, row in daily.iterrows():
            self.daily_stats[row["subcategory"]] = row.to_dict()

        # ----- Volatility -------------------------------------------------
        vol = self.df[self.df["category"] == "VOLATILITY"]
        for _, row in vol.iterrows():
            self.volatility_stats[row["subcategory"]] = row.to_dict()

    # ------------------------------------------------------------------
    # 4️⃣  Helper – pick the most appropriate stats bucket
    # ------------------------------------------------------------------
    def _get_applicable_stats(self, hour: int, volatility_state: str) -> Dict:
        """
        Return the statistics dict that best matches the current context.

        Preference order:
            1️⃣  Volatility bucket (LOW / NORMAL / HIGH)
            2️⃣  Hourly bucket (e.g. “13:00‑13:59”)
            3️⃣  Overall stats (fallback)
        """
        # 1️⃣  Volatility
        if volatility_state.upper() in self.volatility_stats:
            return self.volatility_stats[volatility_state.upper()]

        # 2️⃣  Hourly – format must match exactly the CSV subcategory
        hour_key = f"{hour:02d}:00-{hour:02d}:59"
        if hour_key in self.hourly_stats:
            return self.hourly_stats[hour_key]

        # 3️⃣  Overall fallback
        return self.overall_stats or {}

    # ------------------------------------------------------------------
    # 5️⃣  Core SL distance calculation (pips)
    # ------------------------------------------------------------------
    def _calculate_sl_distance(self, stats: Dict, volatility_state: str) -> float:
        """
        Derive a sensible SL distance (in pips) from the supplied stats.

        Strategy
        --------
        * Base SL = average total range * 1.2  (20 % safety buffer)
        * Adjust for volatility:
            – HIGH   → × 1.5  (wider SL)
            – LOW    → × 0.8  (tighter SL)
            – NORMAL → × 1.0
        * Enforce a minimum of 15 pips.
        """
        avg_range = stats.get("total_range_avg_pips", 30)
        base_sl = avg_range * 1.2

        # Volatility multiplier
        if volatility_state.upper() == "HIGH":
            mult = 1.5
        elif volatility_state.upper() == "LOW":
            mult = 0.8
        else:
            mult = 1.0

        sl = base_sl * mult
        sl = max(sl, 15)          # never tighter than 15 pips
        return round(sl, 1)

    # ------------------------------------------------------------------
    # 6️⃣  Public API – optimal SL/TP for a given trade
    # ------------------------------------------------------------------
    def get_optimal_sl_tp(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        current_hour: Optional[int] = None,
        volatility_state: str = "NORMAL",
        risk_reward_ratio: float = 2.0,
    ) -> Dict:
        """
        Calculate optimal SL/TP based on the loaded statistics.

        Args:
            symbol: Instrument ticker (e.g. “EURUSD”)
            direction: “BUY” or “SELL”
            entry_price: Desired entry price
            current_hour: Hour of day (0‑23).  If omitted, uses now().
            volatility_state: “LOW”, “NORMAL” or “HIGH”.
            risk_reward_ratio: Desired R:R (default 2 : 1).

        Returns:
            dict with keys:
                - sl_price, tp_price (price levels)
                - sl_pips, tp_pips (distance in pips)
                - risk_reward_ratio
                - based_on (human readable description)
                - avg_range_pips (for diagnostics)
                - confidence (HIGH / MODERATE / LOW)
        """
        if self.overall_stats is None:
            logger.warning("⚠️ No statistics loaded – falling back to defaults")
            return self._default_sl_tp(entry_price, direction)

        # Resolve hour if not supplied
        if current_hour is None:
            current_hour = datetime.now().hour

        # Grab the most relevant stats bucket
        stats = self._get_applicable_stats(current_hour, volatility_state)

        # Compute SL distance (pips) and derive TP
        sl_pips = self._calculate_sl_distance(stats, volatility_state)
        tp_pips = sl_pips * risk_reward_ratio

        # Pip size depends on the instrument (JPY pairs have 0.01)
        pip_size = 0.0001
        if "JPY" in symbol.upper():
            pip_size = 0.01

        if direction.upper() == "BUY":
            sl_price = entry_price - sl_pips * pip_size
            tp_price = entry_price + tp_pips * pip_size
        else:   # SELL
            sl_price = entry_price + sl_pips * pip_size
            tp_price = entry_price - tp_pips * pip_size

        return {
            "sl_price": round(sl_price, 5),
            "tp_price": round(tp_price, 5),
            "sl_pips": sl_pips,
            "tp_pips": tp_pips,
            "risk_reward_ratio": risk_reward_ratio,
            "based_on": f"{volatility_state.title()} volatility, hour {current_hour}",
            "avg_range_pips": stats.get("total_range_avg_pips", 0),
            "confidence": "HIGH" if len(self.hourly_stats) > 20 else "MODERATE",
        }

    # ------------------------------------------------------------------
    # 7️⃣  Fallback SL/TP when no statistics are available
    # ------------------------------------------------------------------
    def _default_sl_tp(self, entry_price: float, direction: str) -> Dict:
        """Very safe default – 30 pips SL, 60 pips TP (1 : 2)."""
        logger.warning("⚠️ Using default SL/TP (no statistics)")
        sl_pips = 30
        tp_pips = 60
        pip_size = 0.0001
        if direction.upper() == "SELL" and "JPY" in str(entry_price):
            pip_size = 0.01

        if direction.upper() == "BUY":
            sl_price = entry_price - sl_pips * pip_size
            tp_price = entry_price + tp_pips * pip_size
        else:
            sl_price = entry_price + sl_pips * pip_size
            tp_price = entry_price - tp_pips * pip_size

        return {
            "sl_price": round(sl_price, 5),
            "tp_price": round(tp_price, 5),
            "sl_pips": sl_pips,
            "tp_pips": tp_pips,
            "risk_reward_ratio": 2.0,
            "based_on": "DEFAULT",
            "confidence": "LOW",
        }

    # ------------------------------------------------------------------
    # 8️⃣  Utility – best trading hours (based on average range & sample size)
    # ------------------------------------------------------------------
    def get_best_trading_hours(self, min_sample_size: int = 50) -> List[int]:
        """
        Return a list of the top‑performing hours (0‑23) ordered by
        “quality” = avg_range * (samples/100).  Hours with fewer than
        ``min_sample_size`` candles are ignored.
        """
        if not self.hourly_stats:
            # Default to typical European/US session hours
            return list(range(8, 18))

        ranked = []
        for hour_str, stats in self.hourly_stats.items():
            total_candles = stats.get("total_candles", 0)
            if total_candles < min_sample_size:
                continue

            avg_range = stats.get("total_range_avg_pips", 0)
            hour = int(hour_str.split(":")[0])          # “13:00‑13:59” → 13
            quality = avg_range * (total_candles / 100)
            ranked.append({"hour": hour, "quality": quality})

        ranked.sort(key=lambda x: x["quality"], reverse=True)
        return [item["hour"] for item in ranked[:12]]   # top 12 hours

    # ------------------------------------------------------------------
    # 9️⃣  Volatility‑based position‑size adjustment
    # ------------------------------------------------------------------
    def get_volatility_adjustment(self, volatility_state: str) -> float:
        """
        Return a multiplier that can be applied to the raw position size.

        * HIGH volatility → 0.5 (reduce size)
        * LOW  volatility → 1.2 (increase size)
        * NORMAL          → 1.0
        """
        if not self.overall_stats:
            return 1.0

        state = volatility_state.upper()
        if state not in self.volatility_stats:
            return 1.0

        stats = self.volatility_stats[state]
        avg_range = stats.get("total_range_avg_pips", 0)
        overall_avg = self.overall_stats.get("total_range_avg_pips", 50)

        if avg_range > overall_avg * 1.5:
            return 0.5
        if avg_range < overall_avg * 0.7:
            return 1.2
        return 1.0

    # ------------------------------------------------------------------
    # 10️⃣  Human‑readable summary (good for logs / Grafana panels)
    # ------------------------------------------------------------------
    def get_statistics_summary(self) -> str:
        """Return a pretty‑printed multi‑section summary."""
        if self.overall_stats is None:
            return "No statistics loaded"

        s = "\n"
        s += "═" * 70 + "\n"
        s += "       CANDLESTICK STATISTICS SUMMARY\n"
        s += "═" * 70 + "\n"

        # Overall
        s += "\n📊 Overall Statistics:\n"
        s += f"   Total candles : {int(self.overall_stats.get('total_candles', 0))}\n"
        s += f"   Bullish %     : {self.overall_stats.get('bullish_pct', 0):.1f}%\n"
        s += f"   Avg body (p)  : {self.overall_stats.get('body_avg_pips', 0):.1f} pips\n"
        s += f"   Avg range (p) : {self.overall_stats.get('total_range_avg_pips', 0):.1f} pips\n"
        s += f"   Max range (p) : {self.overall_stats.get('total_range_max_pips', 0):.1f} pips\n"

        # Hourly
        s += "\n⏰ Hourly Statistics:\n"
        s += f"   Hours loaded : {len(self.hourly_stats)}\n"

        # Daily
        s += "\n📅 Daily Statistics:\n"
        s += f"   Days loaded  : {len(self.daily_stats)}\n"

        # Volatility
        s += "\n📊 Volatility Statistics:\n"
        for vol, stats in self.volatility_stats.items():
            s += f"   {vol}: {stats.get('total_range_avg_pips', 0):.1f} pips avg\n"

        # Recommendations
        s += "\n💡 Recommendations:\n"
        s += f"   Conservative SL : {self.overall_stats.get('recommended_sl_pips', 0):.1f} pips\n"
        s += f"   Suggested TP (1:2) : {self.overall_stats.get('recommended_tp_pips', 0):.1f} pips\n"

        s += "\n" + "═" * 70 + "\n"
        return s

    # ------------------------------------------------------------------
    # 11️⃣  Export a ready‑to‑consume JSON file (e.g. for UI or downstream bots)
    # ------------------------------------------------------------------
    def export_recommendations(self, output_path: str = "sl_tp_recommendations.json") -> bool:
        """
       except Exception as e:   # pragma: no cover
            logger.error(f"❌ Export failed: {e}")
            return False

# ----------------------------------------------------------------------
# Global singleton – import this from ``src/main.py`` and use it directly
# ----------------------------------------------------------------------
candlestick_analyzer = CandlestickAnalyzer()

# ----------------------------------------------------------------------
# Simple sanity‑check / demo when the module is executed directly
# ----------------------------------------------------------------------
def _demo() -> None:
    """
    Run a quick demonstration:
    * Load the CSV (if present)
    * Print a summary
    * Show SL/TP for a few example symbols / directions
    * List the best trading hours
    * Export the JSON recommendation file
    """
    logger.info("=" * 70)
    logger.info("      CANDLESTICK ANALYZER – DEMO RUN")
    logger.info("=" * 70)

    if not candlestick_analyzer.load_statistics():
        logger.error("❌ Unable to load candlestick statistics – aborting demo.")
        logger.info(
            "💡 To generate the CSV, run the MQL5 script "
            "`CandlestickAnalysisReport.mq5` inside MetaTrader 5."
        )
        return

    # ------------------------------------------------------------------
    # 1️⃣  Print a human‑readable summary
    # ------------------------------------------------------------------
    logger.info("\n📊 Statistics Summary:")
    logger.info(candlestick_analyzer.get_statistics_summary())

    # ------------------------------------------------------------------
    # 2️⃣  Show a few SL/TP calculations
    # ------------------------------------------------------------------
    logger.info("\n🎯 Sample SL/TP calculations")
    logger.info("-" * 60)

    sample_trades = [
        {"symbol": "EURUSD", "direction": "BUY", "entry": 1.0850, "vol": "NORMAL"},
        {"symbol": "GBPUSD", "direction": "SELL", "entry": 1.2650, "vol": "HIGH"},
        {"symbol": "USDJPY", "direction": "BUY", "entry": 150.25, "vol": "LOW"},
    ]

    for tr in sample_trades:
        res = candlestick_analyzer.get_optimal_sl_tp(
            symbol=tr["symbol"],
            direction=tr["direction"],
            entry_price=tr["entry"],
            volatility_state=tr["vol"],
        )
        logger.info(
            f"\n{tr['symbol']} {tr['direction']} @ {tr['entry']:.5f} "
            f"[{tr['vol']} vol]"
        )
        logger.info(
            f"  ➜ SL: {res['sl_price']:.5f} ({res['sl_pips']:.1f} pips) | "
            f"TP: {res['tp_price']:.5f} ({res['tp_pips']:.1f} pips) | "
            f"R:R = 1:{res['risk_reward_ratio']:.1f}"
        )
        logger.info(f"  ➜ Confidence: {res['confidence']} (based on {res['based_on']})")

    # ------------------------------------------------------------------
    # 3️⃣  Best trading hours (according to historic range & sample size)
    # ------------------------------------------------------------------
    best_hours = candlestick_analyzer.get_best_trading_hours()
    logger.info("\n⏰ Best Trading Hours (0‑23):")
    logger.info("-" * 60)
    logger.info(", ".join(f"{h:02d}:00" for h in best_hours))

    # ------------------------------------------------------------------
    # 4️⃣  Export the full recommendation bundle to JSON
    # ------------------------------------------------------------------
    logger.info("\n💾 Exporting JSON recommendation file …")
    if candlestick_analyzer.export_recommendations():
        logger.info("✅ Export successful → sl_tp_recommendations.json")
    else:
        logger.error("❌ Export failed")

    logger.info("\n" + "=" * 70 + "\nDemo finished.\n")


if __name__ == "__main__":
    # When the module is invoked directly, run the demo.
    # In production the bot will import ``candlestick_analyzer`` and call
    # ``get_optimal_sl_tp`` as needed.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _demo()
