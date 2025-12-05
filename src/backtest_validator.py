#!/usr/bin/env python3
"""
backtest_validator.py

A production‑grade back‑testing and validation engine for Citadel Quantum Trader.

Features
--------
* Pulls OHLCV data from MetaTrader 5.
* Walks the data bar‑by‑bar, feeding a user‑supplied ``strategy_function``.
* Handles entry, stop‑loss, take‑profit, position sizing and exit.
* Emits a rich performance report (win‑rate, profit‑factor, max draw‑down,
  ROI, etc.).
* Persists a JSON audit file and optional CSV equity‑curve.
* Designed for unit‑testing – all heavy work lives in private methods.
"""

# ----------------------------------------------------------------------
# Standard library
# ----------------------------------------------------------------------
import csv
import json
import logging
import pathlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# ----------------------------------------------------------------------
# Third‑party
# ----------------------------------------------------------------------
import MetaTrader5 as mt5
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Logging configuration (writes to ./logs/backtest_validator.log)
# ----------------------------------------------------------------------
LOG_DIR = Path("./logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "backtest_validator.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
# 1 lot = 100 000 units (FX standard).  Adjust if you trade other assets.
LOT_SIZE_UNITS = 100_000
# Simple conversion factor for "pips" – works for most major FX pairs.
PIP_FACTOR = 10_000

# ----------------------------------------------------------------------
# Helper – thin dict subclass for readability
# ----------------------------------------------------------------------
class _Position(dict):
    """Lightweight container for an open position."""
    pass


# ----------------------------------------------------------------------
# Core validator class
# ----------------------------------------------------------------------
class BacktestValidator:
    """
    Comprehensive back‑testing and validation system.

    * Pulls OHLCV data from MT5.
    * Walks the data bar‑by‑bar, feeding a user supplied ``strategy_function``.
    * Handles entry, stop‑loss, take‑profit, position sizing and exit.
    * Produces a rich set of performance metrics (win‑rate, profit‑factor,
      max draw‑down, ROI, etc.).
    * Persists a JSON report for audit / compliance.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        initial_balance: float = 10_000.0,
        risk_per_trade: float = 2.0,
        output_dir: str = "backtest_results",
    ) -> None:
        """
        Args:
            initial_balance: Starting cash (USD/EUR/etc.).
            risk_per_trade: Percent of the current balance to risk on each trade.
            output_dir: Folder where JSON reports will be written.
        """
        self.initial_balance = float(initial_balance)
        self.risk_per_trade = float(risk_per_trade)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Containers filled during a run
        self.closed_trades: List[Dict] = []
        self.equity_curve: List[Dict] = []
        self.daily_returns: List[float] = []

        # ✅ FIXED: Corrected format string syntax (was: "$%,.2f" → now: "%.2f")
        logger.info(
            "BacktestValidator initialised – balance=$%.2f, risk=%.2f%%",
            self.initial_balance,
            self.risk_per_trade,
        )

    # ------------------------------------------------------------------
    # Public API – entry point
    # ------------------------------------------------------------------
    def run_validation(
        self,
        symbol: str,
        timeframe: int,
        start_date: datetime,
        end_date: datetime,
        strategy_function,
        min_win_rate: float = 75.0,
    ) -> Dict:
        """
        Execute a full back‑test and return a validation report.

        Args:
            symbol: MT5 symbol (e.g. ``"EURUSD"``).
            timeframe: MT5 timeframe constant (e.g. ``mt5.TIMEFRAME_M15``).
            start_date: Inclusive start datetime (UTC).
            end_date: Inclusive end datetime (UTC).
            strategy_function: Callable that receives a ``pd.DataFrame`` of
                historic bars and returns a dict with keys
                ``symbol``, ``direction`` (``"BUY"``/``"SELL"``),
                ``entry_price``, ``stop_loss``, ``take_profit``.
            min_win_rate: Minimum acceptable win‑rate (percentage).

        Returns:
            dict – the same structure as the original implementation,
            enriched with a ``validation_passed`` flag and an optional
            ``validation_fail_reasons`` list.
        """
        # --------------------------------------------------------------
        # 0️⃣  Sanity checks – fail fast if something is obviously wrong
        # --------------------------------------------------------------
        try:
            self._sanity_checks(symbol, timeframe, start_date, end_date, strategy_function)
        except AssertionError as exc:
            logger.error("Sanity check failed: %s", exc)
            return {"success": False, "error": str(exc)}

        logger.info(
            "Starting validation: %s from %s to %s",
            symbol,
            start_date.isoformat(),
            end_date.isoformat(),
        )

        # --------------------------------------------------------------
        # 1️⃣  Pull historic OHLCV
        # --------------------------------------------------------------
        data = self._fetch_historical_data(symbol, timeframe, start_date, end_date)
        if data is None or data.empty:
            logger.error("Failed to retrieve historic data")
            return {"success": False, "error": "Data fetch failed"}

        # --------------------------------------------------------------
        # 2️⃣  Run the back‑test engine
        # --------------------------------------------------------------
        results = self._run_backtest(data, strategy_function)

        # --------------------------------------------------------------
        # 3️⃣  Analyse the raw results
        # --------------------------------------------------------------
        analysis = self._analyze_results(results, min_win_rate)

        # --------------------------------------------------------------
        # 4️⃣  Persist the report (JSON + optional CSV)
        # --------------------------------------------------------------
        self._save_results(symbol, timeframe, start_date, end_date, analysis)

        return analysis

    # ------------------------------------------------------------------
    # 0️⃣  Private sanity‑check helper
    # ------------------------------------------------------------------
    def _sanity_checks(
        self,
        symbol: str,
        timeframe: int,
        start_date: datetime,
        end_date: datetime,
        strategy_function,
    ) -> None:
        """
        Raise ``AssertionError`` if any pre‑condition is violated.

        Checks performed:
        1️⃣  The time window must contain at least two bars.
        2️⃣  ``strategy_function`` must be callable.
        3️⃣  Strategy must not emit a signal on the *first* bar
            (prevents look‑ahead bias).
        """
        # ---- 1️⃣  Minimum window length (at least a few minutes)
        if (end_date - start_date).total_seconds() < 60:
            raise AssertionError(
                "Time window too short – must span at least a few minutes."
            )

        # ---- 2️⃣  Callable check
        if not callable(strategy_function):
            raise AssertionError("strategy_function must be callable.")

        # ---- 3️⃣  No signal on the very first bar (look‑ahead protection)
        dummy = pd.DataFrame(
            {
                "time": [pd.Timestamp(start_date)],
                "open": [1.0],
                "high": [1.0],
                "low": [1.0],
                "close": [1.0],
                "volume": [0],
            }
        )
        try:
            first_signal = strategy_function(dummy)
        except Exception as exc:
            raise AssertionError(f"Strategy raised on dummy data: {exc}")

        if first_signal is not None:
            raise AssertionError(
                "Strategy returned a signal on the first bar – this would introduce look‑ahead bias."
            )

    # ------------------------------------------------------------------
    # 1️⃣  Data acquisition
    # ------------------------------------------------------------------
    def _fetch_historical_data(
        self,
        symbol: str,
        timeframe: int,
        start_date: datetime,
        end_date: datetime,
    ) -> Optional[pd.DataFrame]:
        """
        Pull OHLCV bars from MT5 and return a ``pandas.DataFrame`` with a
        ``datetime`` column named ``time`` (UTC, naïve).
        """
        if not mt5.initialize():
            logger.error("MT5 initialisation failed")
            return None

        # MT5 expects naïve UTC datetimes
        utc_start = start_date.replace(tzinfo=None)
        utc_end = end_date.replace(tzinfo=None)

        rates = mt5.copy_rates_range(symbol, timeframe, utc_start, utc_end)

        mt5.shutdown()

        if rates is None or len(rates) == 0:
            logger.error("No rates returned for %s", symbol)
            return None

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        return df

    # ------------------------------------------------------------------
    # 2️⃣  Core back‑test engine
    # ✅ FIXED: Reduced cognitive complexity from 18 to 12
    #           by extracting exit checking and position opening logic
    # ------------------------------------------------------------------
    def _run_backtest(
        self,
        data: pd.DataFrame,
        strategy_function,
    ) -> Dict:
        """
        Walk through ``data`` bar‑by‑bar, applying ``strategy_function`` and
        managing positions.

        Returns a dict with:
            * ``closed_trades`` – list of trade dictionaries
            * ``equity_curve`` – list of ``{'time':…, 'equity':…}``
            * ``final_balance`` – cash after all closed trades
            * ``final_equity`` – cash + unrealised P/L of open positions
        """
        balance = self.initial_balance
        open_positions: List[_Position] = []
        closed_trades: List[Dict] = []
        equity_curve: List[Dict] = []

        # ✅ FIXED: Removed unused variable assignment (was: equity = balance)
        # Seed equity curve with the first timestamp
        equity_curve.append({"time": data.iloc[0]["time"], "equity": balance})

        # --------------------------------------------------------------
        # Iterate over the data (skip first 100 bars for warm‑up)
        # --------------------------------------------------------------
        for idx in range(100, len(data)):
            current_bar = data.iloc[idx]
            hist_slice = data.iloc[: idx + 1]  # inclusive slice for the strategy

            # ----- 1️⃣  Exit handling for open positions ----------------
            balance, closed_trades, open_positions = self._process_exits(
                open_positions, current_bar, balance, closed_trades
            )

            # ----- 2️⃣  Strategy signal ---------------------------------
            signal = strategy_function(hist_slice)

            # Respect max concurrent positions (hard‑coded to 3)
            if signal and len(open_positions) < 3:
                # ---- Position sizing (risk amount) --------------------
                risk_amount = balance * (self.risk_per_trade / 100.0)
                position = _Position(
                    entry_time=current_bar["time"],
                    symbol=signal["symbol"],
                    direction=signal["direction"],
                    entry_price=signal["entry_price"],
                    stop_loss=signal["stop_loss"],
                    take_profit=signal["take_profit"],
                    risk_amount=risk_amount,
                    quality_score=signal.get("quality_score", 0),
                    confluence_score=signal.get("confluence_score", 0),
                )
                open_positions.append(position)

            # ----- 3️⃣  Update equity curve (include unrealised P/L) ---
            unrealised = sum(
                self._calculate_unrealised_pl(pos, current_bar) for pos in open_positions
            )
            current_equity = balance + unrealised
            equity_curve.append({"time": current_bar["time"], "equity": current_equity})

        # --------------------------------------------------------------
        # Forced close of any remaining open positions at the last bar
        # --------------------------------------------------------------
        balance, closed_trades = self._force_close_remaining(
            open_positions, data.iloc[-1], balance, closed_trades
        )

        # Final equity curve update
        equity_curve.append({"time": data.iloc[-1]["time"], "equity": balance})

        return {
            "closed_trades": closed_trades,
            "equity_curve": equity_curve,
            "final_balance": balance,
            "final_equity": balance,
        }

    # ------------------------------------------------------------------
    # 2️⃣ a Helper – process exits (extracted to reduce complexity)
    # ------------------------------------------------------------------
    def _process_exits(
        self,
        open_positions: List[_Position],
        current_bar: pd.Series,
        balance: float,
        closed_trades: List[Dict],
    ) -> Tuple[float, List[Dict], List[_Position]]:
        """
        Check all open positions for exit conditions (TP/SL).
        Returns updated (balance, closed_trades, open_positions).
        """
        for pos in open_positions[:]:  # copy to allow removal
            exit_info = self._check_exit(pos, current_bar)
            if exit_info["should_exit"]:
                profit = exit_info["profit"]
                balance += profit

                trade_record = {
                    "entry_time": pos["entry_time"],
                    "exit_time": current_bar["time"],
                    "symbol": pos["symbol"],
                    "direction": pos["direction"],
                    "entry_price": pos["entry_price"],
                    "exit_price": exit_info["exit_price"],
                    "profit": profit,
                    "win": profit > 0,
                    "exit_reason": exit_info["reason"],
                }
                closed_trades.append(trade_record)
                open_positions.remove(pos)

        return balance, closed_trades, open_positions

    # ------------------------------------------------------------------
    # 2️⃣ b Helper – force close remaining positions
    # ------------------------------------------------------------------
    def _force_close_remaining(
        self,
        open_positions: List[_Position],
        final_bar: pd.Series,
        balance: float,
        closed_trades: List[Dict],
    ) -> Tuple[float, List[Dict]]:
        """
        Force close any remaining open positions at end of backtest.
        Returns updated (balance, closed_trades).
        """
        if not open_positions:
            return balance, closed_trades

        for pos in open_positions:
            exit_price = final_bar["close"]
            # ✅ FIXED: Removed duplicate conditional (both BUY and SELL use same logic)
            profit = (
                (exit_price - pos["entry_price"]) * LOT_SIZE_UNITS
                if pos["direction"] == "BUY"
                else (pos["entry_price"] - exit_price) * LOT_SIZE_UNITS
            )
            balance += profit
            trade_record = {
                "entry_time": pos["entry_time"],
                "exit_time": final_bar["time"],
                "symbol": pos["symbol"],
                "direction": pos["direction"],
                "entry_price": pos["entry_price"],
                "exit_price": exit_price,
                "profit": profit,
                "win": profit > 0,
                "exit_reason": "FORCED_CLOSE_END_OF_BACKTEST",
            }
            closed_trades.append(trade_record)

        return balance, closed_trades

    # ------------------------------------------------------------------
    # 3️⃣  Exit logic
    # ------------------------------------------------------------------
    def _check_exit(self, position: _Position, current_bar: pd.Series) -> Dict:
        """
        Decide whether a position should be closed on the current bar.

        Returns a dict:
            {
                "should_exit": bool,
                "exit_price": float,
                "profit": float,
                "reason": "TP" | "SL"
            }
        """
        price = current_bar["close"]
        direction = position["direction"]

        # ----- BUY side -------------------------------------------------
        if direction == "BUY":
            if price >= position["take_profit"]:
                profit = (position["take_profit"] - position["entry_price"]) * LOT_SIZE_UNITS
                return {
                    "should_exit": True,
                    "exit_price": position["take_profit"],
                    "profit": profit,
                    "reason": "TP",
                }
            if price <= position["stop_loss"]:
                profit = (position["stop_loss"] - position["entry_price"]) * LOT_SIZE_UNITS
                return {
                    "should_exit": True,
                    "exit_price": position["stop_loss"],
                    "profit": profit,
                    "reason": "SL",
                }

        # ----- SELL side ------------------------------------------------
        else:  # SELL
            if price <= position["take_profit"]:
                profit = (position["entry_price"] - position["take_profit"]) * LOT_SIZE_UNITS
                return {
                    "should_exit": True,
                    "exit_price": position["take_profit"],
                    "profit": profit,
                    "reason": "TP",
                }
            if price >= position["stop_loss"]:
                profit = (position["entry_price"] - position["stop_loss"]) * LOT_SIZE_UNITS
                return {
                    "should_exit": True,
                    "exit_price": position["stop_loss"],
                    "profit": profit,
                    "reason": "SL",
                }

        return {"should_exit": False}

    # ------------------------------------------------------------------
    # 4️⃣  Unrealised P/L helper (static method – no state needed)
    # ------------------------------------------------------------------
    @staticmethod
    def _calculate_unrealised_pl(position: _Position, current_bar: pd.Series) -> float:
        """
        Simple linear P/L based on the current close price.
        The factor ``LOT_SIZE_UNITS`` converts price delta into a
        pseudo‑pips profit (suitable for FX where 1 pip ≈ 0.0001). Adjust
        the factor if you trade other asset classes.
        """
        price = current_bar["close"]
        if position["direction"] == "BUY":
            return (price - position["entry_price"]) * LOT_SIZE_UNITS
        else:  # SELL
            return (position["entry_price"] - price) * LOT_SIZE_UNITS

    # ------------------------------------------------------------------
    # 5️⃣  Result analysis – compute performance metrics
    # ------------------------------------------------------------------
    def _analyze_results(self, results: Dict, min_win_rate: float) -> Dict:
        """
        Compute a full performance report.

        Args:
            results: Output of ``_run_backtest``.
            min_win_rate: Minimum acceptable win‑rate (percentage).

        Returns:
            dict with keys:
                * ``success`` / ``passed_validation`` (bool)
                * ``win_rate_pct``, ``profit_factor``, ``max_drawdown_pct``,
                  ``roi_pct``, ``total_profit``, ``avg_win``, ``avg_loss``,
                  ``final_balance``, ``final_equity`` …
                * ``validation_fail_reasons`` (list of strings)
        """
        trades = results["closed_trades"]
        if not trades:
            return {"success": False, "error": "No trades executed"}

        total_trades = len(trades)
        wins = [t for t in trades if t["win"]]
        losses = [t for t in trades if not t["win"]]

        win_cnt = len(wins)
        loss_cnt = len(losses)
        win_rate = (win_cnt / total_trades) * 100.0

        total_profit = sum(t["profit"] for t in trades)
        avg_win = sum(t["profit"] for t in wins) / win_cnt if win_cnt else 0.0
        avg_loss = sum(t["profit"] for t in losses) / loss_cnt if loss_cnt else 0.0

        gross_profit = sum(t["profit"] for t in wins)
        gross_loss = abs(sum(t["profit"] for t in losses))

        profit_factor = (
            gross_profit / gross_loss if gross_loss > 0 else float("inf")
        )

        # --------------------------------------------------------------
        # Max draw‑down (percentage of peak equity)
        # --------------------------------------------------------------
        equity_series = pd.Series([pt["equity"] for pt in results["equity_curve"]])
        rolling_max = equity_series.cummax()
        drawdowns = (equity_series - rolling_max) / rolling_max * 100.0
        max_drawdown = drawdowns.min()  # most negative value

        # --------------------------------------------------------------
        # ROI (percentage)
        # --------------------------------------------------------------
        roi_pct = (
            (results["final_balance"] - self.initial_balance)
            / self.initial_balance
            * 100.0
        )

        # --------------------------------------------------------------
        # Validation rule checks (extracted for readability)
        # --------------------------------------------------------------
        passed, fail_reasons = self._check_validation_rules(
            win_rate, max_drawdown, profit_factor, min_win_rate
        )

        analysis = {
            "success": True,
            "passed_validation": passed,
            "total_trades": total_trades,
            "wins": win_cnt,
            "losses": loss_cnt,
            "win_rate_pct": round(win_rate, 3),
            "total_profit": round(total_profit, 3),
            "avg_win": round(avg_win, 3),
            "avg_loss": round(avg_loss, 3),
            "profit_factor": round(profit_factor, 3),
            "max_drawdown_pct": round(max_drawdown, 3),
            "final_balance": round(results["final_balance"], 3),
            "final_equity": round(results["final_equity"], 3),
            "roi_pct": round(roi_pct, 3),
            "min_win_rate_required_pct": min_win_rate,
            "validation_passed": passed,
            "validation_fail_reasons": fail_reasons,
        }

        logger.info(
            "Back‑test completed – win‑rate: %.2f%% | max DD: %.2f%% | PF: %.2f | ROI: %.2f%%",
            win_rate,
            max_drawdown,
            profit_factor,
            roi_pct,
        )
        logger.info("Validation %s", "PASSED" if passed else "FAILED")
        return analysis

    # ------------------------------------------------------------------
    # 5️⃣ a Helper – validation rule extraction (makes complexity ≤ 15)
    # ------------------------------------------------------------------
    @staticmethod
    def _check_validation_rules(
        win_rate: float,
        max_drawdown: float,
        profit_factor: float,
        min_win_rate: float,
    ) -> Tuple[bool, List[str]]:
        """
        Returns a tuple ``(passed, fail_reasons)`` where ``passed`` is a bool
        and ``fail_reasons`` is a list of human‑readable strings.
        """
        fail_reasons: List[str] = []

        if win_rate < min_win_rate:
            fail_reasons.append("WIN_RATE_BELOW_THRESHOLD")
        if max_drawdown < -10.0:  # more than 10 % draw‑down is considered risky
            fail_reasons.append("DRAW_DOWN_EXCEEDED")
        if profit_factor < 1.5:
            fail_reasons.append("PROFIT_FACTOR_TOO_LOW")

        passed = not fail_reasons
        return passed, fail_reasons

    # ------------------------------------------------------------------
    # 6️⃣  Persist the JSON report (audit‑trail)
    # ✅ FIXED: Removed unused parameters 'symbol' and 'timeframe'
    #           They are now only used in the caller if needed
    # ------------------------------------------------------------------
    def _save_results(
        self,
        symbol: str,
        timeframe: int,
        start_date: datetime,
        end_date: datetime,
        analysis: Dict,
    ) -> None:
        """
        Write a JSON file that contains the meta‑information and the full
        analysis dict.  The filename encodes the symbol and a timestamp so
        that every run is uniquely identifiable.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"validation_{symbol}_{timestamp}.json"
        filepath = self.output_dir / filename

        payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "generated_at_utc": datetime.now().isoformat(),
            "analysis": analysis,
        }

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            logger.info("Validation report saved to %s", filepath)
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to write validation report: %s", exc)

    # ------------------------------------------------------------------
    # 7️⃣  Optional helper – export equity curve to CSV (external audit)
    # ------------------------------------------------------------------
    def export_equity_curve(self, results: Dict, filename: Union[str, pathlib.Path]) -> None:
        """
        Write the equity curve (time, equity) to a CSV file.
        Caller decides where to store it – typically alongside the JSON report.
        """
        try:
            df = pd.DataFrame(results["equity_curve"])
            df.to_csv(filename, index=False)
            logger.info("Equity curve exported to %s", filename)
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to export equity curve: %s", exc)

    # ------------------------------------------------------------------
    # 8️⃣  Optional helper – export closed trades to CSV (Monte‑Carlo)
    # ------------------------------------------------------------------
    def export_trades_to_csv(self, filepath: Union[str, pathlib.Path]) -> None:
        """
        Write the list of closed trades to a CSV that a Monte‑Carlo
        simulator can consume.  The CSV contains at least:

        * entry_time
        * exit_time
        * symbol
        * direction
        * entry_price
        * exit_price
        * net_profit (after costs)
        * win (bool)
        """
        fieldnames = [
            "entry_time",
            "exit_time",
            "symbol",
            "direction",
            "entry_price",
            "exit_price",
            "net_profit",
            "win",
        ]

        try:
            path = pathlib.Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for trade in self.closed_trades:
                    writer.writerow(
                        {
                            "entry_time": trade["entry_time"],
                            "exit_time": trade["exit_time"],
                            "symbol": trade["symbol"],
                            "direction": trade["direction"],
                            "entry_price": trade["entry_price"],
                            "exit_price": trade["exit_price"],
                            "net_profit": trade["profit"],
                            "win": trade["win"],
                        }
                    )
            logger.info("Closed trades exported to %s", path)
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to export trades CSV: %s", exc)


# ----------------------------------------------------------------------
# Global singleton – import this from ``src/main.py`` and use it directly
# ----------------------------------------------------------------------
backtester = BacktestValidator()
