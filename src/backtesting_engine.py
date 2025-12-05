#!/usr/bin/env python3
"""
Backtesting Engine

Test trading strategy on historical data

Features:
- Historical data loading from MT5
- Trade simulation
- Performance metrics calculation
- Equity curve generation
- Export results
- Multiple timeframe support
"""

# ----------------------------------------------------------------------
# Standard library
# ----------------------------------------------------------------------
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from threading import Event, Thread
from time import sleep
from typing import Dict, List, Optional, Tuple

# ----------------------------------------------------------------------
# Third‑party
# ----------------------------------------------------------------------
import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from prometheus_client import Gauge  # already a dependency of the project

# ----------------------------------------------------------------------
# Logging configuration (uses the global logger of the application)
# ----------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# DATA CLASSES
# ----------------------------------------------------------------------
@dataclass
class BacktestTrade:
    """Record of a single simulated trade."""
    entry_time: datetime
    exit_time: datetime
    symbol: str
    direction: str                     # "BUY" or "SELL"
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    lot_size: float
    profit: float
    r_multiple: float
    was_news_blocked: bool
    volatility_state: str
    confluence_score: int
    exit_reason: str


@dataclass
class BacktestResults:
    """Container for the complete back‑test output."""
    start_date: datetime
    end_date: datetime
    initial_balance: float
    final_balance: float
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[Dict] = field(default_factory=list)   # [{'time':…, 'balance':…}, …]
    statistics: Dict = field(default_factory=dict)


# ----------------------------------------------------------------------
# BACKTEST ENGINE
# ----------------------------------------------------------------------
class BacktestEngine:
    """
    Backtesting engine to validate strategy performance on historical MT5 data.
    """

    # ------------------------------------------------------------------
    # Prometheus gauges (global – one per engine instance)
    # ------------------------------------------------------------------
    _gauge_trades_processed = Gauge(
        "backtest_trades_processed",
        "Number of trades processed during backtest",
    )
    _gauge_current_equity = Gauge(
        "backtest_current_equity",
        "Current equity during backtest execution",
    )
    _gauge_win_rate = Gauge(
        "backtest_win_rate",
        "Running win‑rate (%) during backtest",
    )
    _gauge_sharpe = Gauge(
        "backtest_sharpe_ratio",
        "Running Sharpe ratio during backtest",
    )

    # ------------------------------------------------------------------
    def __init__(self, initial_balance: float = 10_000.0):
        """
        Initialise the engine.

        Args:
            initial_balance: Starting capital for the simulation.
        """
        self.initial_balance = float(initial_balance)
        self.current_balance = self.initial_balance

        self.trades: List[BacktestTrade] = []
        self.equity_curve: List[Dict] = []

        # Internal bookkeeping for live metrics
        self._processed = 0

        logger.info(f"📊 BacktestEngine initialised with ${self.initial_balance:,.2f}")

    # ------------------------------------------------------------------
    # DATA LOADING
    # ------------------------------------------------------------------
    def load_historical_data(
        self,
        symbol: str,
        timeframe: int,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Load OHLCV data from MT5.

        Returns a DataFrame indexed by time with additional indicator columns.
        """
        try:
            if not mt5.initialize():
                logger.error("MT5 initialisation failed")
                return pd.DataFrame()

            rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)

            if rates is None or len(rates) == 0:
                logger.error(f"No historical data for {symbol}")
                return pd.DataFrame()

            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df = self._add_indicators(df)

            logger.info(f"✅ Loaded {len(df)} bars for {symbol}")
            return df

        except Exception as exc:   # pragma: no cover
            logger.exception(f"Error loading data for {symbol}")
            return pd.DataFrame()

    # ------------------------------------------------------------------
    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate EMA, ATR, RSI and attach them to the frame."""
        # EMA
        df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
        df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()

        # ATR (14‑period)
        df["high_low"] = df["high"] - df["low"]
        df["high_close"] = (df["high"] - df["close"].shift()).abs()
        df["low_close"] = (df["low"] - df["close"].shift()).abs()
        df["tr"] = df[["high_low", "high_close", "low_close"]].max(axis=1)
        df["atr"] = df["tr"].rolling(window=14).mean()

        # RSI (14‑period)
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

        return df

    # ------------------------------------------------------------------
    # SIGNAL GENERATION
    # ------------------------------------------------------------------
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Produce simple confluence signals.

        Adds boolean columns `long_signal` / `short_signal` and SL/TP levels.
        """
        # Trend detection
        df["trend_up"] = (df["ema20"] > df["ema50"]) & (df["close"] > df["ema20"])
        df["trend_down"] = (df["ema20"] < df["ema50"]) & (df["close"] < df["ema20"])

        # Entry signals
        df["long_signal"] = df["trend_up"] & (df["rsi"] > 50) & (df["close"] > df["ema20"])
        df["short_signal"] = df["trend_down"] & (df["rsi"] < 50) & (df["close"] < df["ema20"])

        # Stop‑loss / take‑profit (ATR‑based)
        df["long_sl"] = df["close"] - (df["atr"] * 1.5)
        df["short_sl"] = df["close"] + (df["atr"] * 1.5)

        df["long_tp"] = df["close"] + (df["atr"] * 1.5 * 3)   # 3R
        df["short_tp"] = df["close"] - (df["atr"] * 1.5 * 3)

        return df

    # ------------------------------------------------------------------
    # TRADE SIMULATION
    # ------------------------------------------------------------------
    def simulate_trade(
        self,
        entry_idx: int,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        data: pd.DataFrame,
    ) -> Optional[BacktestTrade]:
        """
        Walk forward from `entry_idx` until SL or TP is hit.

        Returns a `BacktestTrade` instance or `None` if the trade is still open.
        """
        for i in range(entry_idx + 1, len(data)):
            bar = data.iloc[i]

            if direction == "BUY":
                # Stop‑loss hit?
                if bar["low"] <= stop_loss:
                    exit_price = stop_loss
                    exit_reason = "SL_HIT"
                # Take‑profit hit?
                elif bar["high"] >= take_profit:
                    exit_price = take_profit
                    exit_reason = "TP_HIT"
                else:
                    continue
                risk = entry_price - stop_loss
                profit_pips = exit_price - entry_price
                r_multiple = profit_pips / risk if risk > 0 else 0

            else:  # SELL
                if bar["high"] >= stop_loss:
                    exit_price = stop_loss
                    exit_reason = "SL_HIT"
                elif bar["low"] <= take_profit:
                    exit_price = take_profit
                    exit_reason = "TP_HIT"
                else:
                    continue
                risk = stop_loss - entry_price
                profit_pips = entry_price - exit_price
                r_multiple = profit_pips / risk if risk > 0 else 0

            # Simplified monetary profit (fixed $100 risk per trade)
            lot_size = 0.1
            profit = r_multiple * 100

            trade = BacktestTrade(
                entry_time=data.iloc[entry_idx]["time"],
                exit_time=bar["time"],
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                exit_price=exit_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                lot_size=lot_size,
                profit=profit,
                r_multiple=r_multiple,
                was_news_blocked=False,
                volatility_state="NORMAL",
                confluence_score=3,
                exit_reason=exit_reason,
            )
            return trade

        # No exit condition reached – trade remains open
        return None

    # ------------------------------------------------------------------
    # MAIN BACKTEST LOOP
    # ------------------------------------------------------------------
    def run_backtest(
        self,
        symbol: str,
        timeframe: int,
        start_date: datetime,
        end_date: datetime,
    ) -> BacktestResults:
        """
        Execute a full back‑test and return a `BacktestResults` object.
        """
        logger.info("=" * 80)
        logger.info("📊 STARTING BACKTEST")
        logger.info("=" * 80)
        logger.info(f"Symbol: {symbol}")
        logger.info(f"Period: {start_date.date()} → {end_date.date()}")
        logger.info(f"Initial Balance: ${self.initial_balance:,.2f}")
        logger.info("=" * 80 + "\n")

        # ------------------------------------------------------------------
        # 1️⃣ Load data
        # ------------------------------------------------------------------
        df = self.load_historical_data(symbol, timeframe, start_date, end_date)
        if df.empty:
            logger.error("No data loaded – aborting backtest")
            return self._empty_results(start_date, end_date)

        # ------------------------------------------------------------------
        # 2️⃣ Generate signals
        # ------------------------------------------------------------------
        df = self.generate_signals(df)

        # ------------------------------------------------------------------
        # 3️⃣ Simulate trades
        # ------------------------------------------------------------------
        self.trades.clear()
        self.equity_curve.clear()
        self.current_balance = self.initial_balance
        self._processed = 0

        # Record initial equity point
        self.equity_curve.append(
            {"time": df.iloc[0]["time"], "balance": self.current_balance}
        )
        self._gauge_current_equity.set(self.current_balance)

        # Iterate over bars (skip first 50 to allow indicators to stabilise)
        for idx in range(50, len(df)):
            row = df.iloc[idx]

            # LONG signal
            if row["long_signal"]:
                trade = self.simulate_trade(
                    entry_idx=idx,
                    symbol=symbol,
                    direction="BUY",
                    entry_price=row["close"],
                    stop_loss=row["long_sl"],
                    take_profit=row["long_tp"],
                    data=df,
                )
                if trade:
                    self._record_trade(trade)

            # SHORT signal
            elif row["short_signal"]:
                trade = self.simulate_trade(
                    entry_idx=idx,
                    symbol=symbol,
                    direction="SELL",
                    entry_price=row["close"],
                    stop_loss=row["short_sl"],
                    take_profit=row["short_tp"],
                    data=df,
                )
                if trade:
                    self._record_trade(trade)

        # ------------------------------------------------------------------
        # 4️⃣ Compute statistics
        # ------------------------------------------------------------------
        stats = self._calculate_statistics()

        # ------------------------------------------------------------------
        # 5️⃣ Assemble results
        # ------------------------------------------------------------------
        results = BacktestResults(
            start_date=start_date,
            end_date=end_date,
            initial_balance=self.initial_balance,
            final_balance=self.current_balance,
            trades=self.trades,
            equity_curve=self.equity_curve,
            statistics=stats,
        )

        # ------------------------------------------------------------------
        # 6️⃣ Logging / reporting
        # ------------------------------------------------------------------
        self._log_results(stats)

        return results

    # ------------------------------------------------------------------
    # Helper – record a finished trade and update equity / metrics
    # ------------------------------------------------------------------
    def _record_trade(self, trade: BacktestTrade) -> None:
        self.trades.append(trade)
        self.current_balance += trade.profit
        self._processed += 1

        # Equity curve point
        self.equity_curve.append(
            {"time": trade.exit_time, "balance": self.current_balance}
        )

        # Update live Prometheus metrics
        self._gauge_trades_processed.inc()
        self._gauge_current_equity.set(self.current_balance)

        # Re‑calculate temporary win‑rate & Sharpe for live dashboards
        tmp_stats = self._calculate_statistics()
        self._gauge_win_rate.set(tmp_stats["win_rate"])
        self._gauge_sharpe.set(tmp_stats["sharpe_ratio"])

        logger.info(
            f"Trade #{len(self.trades)}: {trade.direction} "
            f"P/L ${trade.profit:+,.2f} ({trade.r_multiple:+.2f}R) "
            f"Equity ${self.current_balance:,.2f}"
        )

    # ------------------------------------------------------------------
    # STATISTICS
    # ------------------------------------------------------------------
    def _calculate_statistics(self) -> Dict:
        """Compute performance metrics from the collected trades."""
        if not self.trades:
            return self._empty_stats()

        total_trades = len(self.trades)
        winners = [t for t in self.trades if t.profit > 0]
        losers = [t for t in self.trades if t.profit <= 0]

        win_rate = (len(winners) / total_trades) * 100 if total_trades else 0

        total_profit = sum(t.profit for t in self.trades)
        total_wins = sum(t.profit for t in winners) if winners else 0
        total_losses = abs(sum(t.profit for t in losers)) if losers else 0

        avg_win = total_wins / len(winners) if winners else 0
        avg_loss = total_losses / len(losers) if losers else 0

        profit_factor = (total_wins / total_losses) if total_losses else 0
        avg_r = sum(t.r_multiple for t in self.trades) / total_trades

        # ------------------- Drawdown -------------------
        peak = self.initial_balance
        max_dd = max_dd_pct = 0
        for pt in self.equity_curve:
            bal = pt["balance"]
            if bal > peak:
                peak = bal
            dd = peak - bal
            dd_pct = (dd / peak) * 100 if peak else 0
            if dd > max_dd:
                max_dd = dd
                max_dd_pct = dd_pct

        # ------------------- Sharpe (simplified) -------------------
        returns = []
        for i in range(1, len(self.equity_curve)):
            prev = self.equity_curve[i - 1]["balance"]
            cur = self.equity_curve[i]["balance"]
            if prev != 0:
                returns.append((cur - prev) / prev)

        if returns:
            avg_ret = np.mean(returns)
            std_ret = np.std(returns)
            sharpe = (avg_ret / std_ret) * np.sqrt(252) if std_ret else 0
        else:
            sharpe = 0

        return {
            "total_trades": total_trades,
            "winners": len(winners),
            "losers": len(losers),
            "win_rate": win_rate,
            "total_profit": total_profit,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "avg_r_multiple": avg_r,
            "max_drawdown": max_dd,
            "max_drawdown_pct": max_dd_pct,
            "sharpe_ratio": sharpe,
            "final_balance": self.current_balance,
            "return_pct": ((self.current_balance - self.initial_balance) / self.initial_balance) * 100,
        }

    # ------------------------------------------------------------------
    def _empty_stats(self) -> Dict:
        """Return a zeroed statistics dict."""
        return {
            "total_trades": 0,
            "winners": 0,
            "losers": 0,
            "win_rate": 0,
            "total_profit": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "profit_factor": 0,
            "avg_r_multiple": 0,
            "max_drawdown": 0,
            "max_drawdown_pct": 0,
            "sharpe_ratio": 0,
            "final_balance": self.initial_balance,
            "return_pct": 0,
        }

    # ------------------------------------------------------------------
    def _empty_results(self, start_date: datetime, end_date: datetime) -> BacktestResults:
        """Create a placeholder result when data could not be loaded."""
        return BacktestResults(
            start_date=start_date,
            end_date=end_date,
            initial_balance=self.initial_balance,
            final_balance=self.initial_balance,
		           trades=[],
            equity_curve=[],
            statistics=self._empty_stats()
        )

    # ------------------------------------------------------------------
    # LOGGING / REPORTING
    # ------------------------------------------------------------------
    def _log_results(self, stats: Dict) -> None:
        """Pretty‑print the back‑test summary to the logger."""
        logger.info("\n" + "=" * 80)
        logger.info("📊 BACKTEST SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total Trades          : {stats['total_trades']}")
        logger.info(f"Winners / Losers     : {stats['winners']} / {stats['losers']}")
        logger.info(f"Win Rate             : {stats['win_rate']:.2f}%")
        logger.info(f"Profit Factor        : {stats['profit_factor']:.2f}")
        logger.info(f"Avg R‑Multiple       : {stats['avg_r_multiple']:.2f}R")
        logger.info(f"Total Profit         : ${stats['total_profit']:+,.2f}")
        logger.info(f"Return on Capital    : {stats['return_pct']:+.2f}%")
        logger.info(f"Max Drawdown         : ${stats['max_drawdown']:.2f} "
                    f"({stats['max_drawdown_pct']:.2f}%)")
        logger.info(f"Sharpe Ratio         : {stats['sharpe_ratio']:.2f}")
        logger.info("=" * 80 + "\n")

    # ------------------------------------------------------------------
    def export_results(self, filename: str = "backtest_results.json") -> None:
        """
        Dump the full back‑test output (trades, equity curve, stats) to a
        JSON file for external analysis or archival.
        """
        results = {
            "statistics": self._calculate_statistics(),
            "trades": [
                {
                    "entry_time": t.entry_time.isoformat(),
                    "exit_time": t.exit_time.isoformat(),
                    "symbol": t.symbol,
                    "direction": t.direction,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "stop_loss": t.stop_loss,
                    "take_profit": t.take_profit,
                    "lot_size": t.lot_size,
                    "profit": t.profit,
                    "r_multiple": t.r_multiple,
                    "exit_reason": t.exit_reason,
                }
                for t in self.trades
            ],
            "equity_curve": [
                {"time": point["time"].isoformat(), "balance": point["balance"]}
                for point in self.equity_curve
            ],
        }

        try:
            with open(filename, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"✅ Backtest results exported to {filename}")
        except Exception as exc:   # pragma: no cover
            logger.error(f"Failed to export results: {exc}")

# ----------------------------------------------------------------------
# GLOBAL ENGINE INSTANCE (convenient for one‑off scripts)
# ----------------------------------------------------------------------
backtest_engine = BacktestEngine()

# ----------------------------------------------------------------------
# QUICK DEMO / SELF‑TEST (run with `python -m src.backtesting_engine`)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple sanity‑check run – 90‑day EURUSD H1 back‑test
    if not mt5.initialize():
        logger.error("❌ MT5 initialisation failed – aborting demo")
        raise SystemExit(1)

    engine = BacktestEngine(initial_balance=10_000)

    now = datetime.now()
    start = now - timedelta(days=90)
    end = now

    results = engine.run_backtest(
        symbol="EURUSD",
        timeframe=mt5.TIMEFRAME_H1,
        start_date=start,
        end_date=end,
    )

    # Export for later inspection
    engine.export_results("eurusd_backtest_90d.json")

    mt5.shutdown()

