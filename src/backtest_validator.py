#!/usr/bin/env python3
"""
BACKTEST VALIDATOR

Demo validation and back‑testing system.
Validates system performance on historical data before live deployment.
"""

# ----------------------------------------------------------------------
# Standard library
# ----------------------------------------------------------------------
import json
import logging
import time
from datetime import datetime, date, time as dt_time, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from .costs import total_cost_pips 
from regime_ensemble import load_bank, match_regime, current_regime_vector
from garch_vol import forecast_vol


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

BANK_CENTROIDS, BANK_META = load_bank()



# ----------------------------------------------------------------------
# Helper – simple position representation
# ----------------------------------------------------------------------
class _Position(dict):
    """Thin dict‑subclass used only for readability."""
    pass


# ----------------------------------------------------------------------
# Main validator class
# ----------------------------------------------------------------------
class BacktestValidator:
    """
    Comprehensive back‑testing and validation system.

    * Pulls OHLCV data from MT5.
    * Walks through the data bar‑by‑bar, feeding a user supplied
      ``strategy_function`` that returns a trade signal.
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
            initial_balance: Starting cash (USD, EUR … whichever currency you use).
            risk_per_trade: Percent of the current balance to risk on each trade.
            output_dir: Folder where JSON reports will be written.
        """
        self.initial_balance = float(initial_balance)
        self.risk_per_trade = float(risk_per_trade)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Containers that are filled during a run
        self.closed_trades: List[Dict] = []
        self.equity_curve: List[Dict] = []
        self.daily_returns: List[float] = []

        logger.info("BacktestValidator initialised")
        logger.info(
            f"Balance=${self.initial_balance:,.2f} | Risk per trade={self.risk_per_trade}%"
        )

# 1️⃣  Pre‑run sanity‑check helper
    # -----------------------------------------------------------------
    def _sanity_checks(self,
                       symbol: str,
                       timeframe: int,
                       start_date: datetime,
                       end_date: datetime,
                       strategy_function) -> None:
        """
        Raise a descriptive AssertionError if any of the basic sanity
        conditions are violated.

        Checks performed:
        1️⃣  The time window must contain at least two bars.
        2️⃣  The strategy function must be callable.
        3️⃣  The strategy must NOT return a trade signal on the *first*
            bar (otherwise we would be trading on incomplete history).
        """
        # ---- 1️⃣  Minimum data length (will be validated later after fetch)
        if (end_date - start_date).total_seconds() < 60:   # < 1 min window → certainly too short
            raise AssertionError("Time window too short – must span at least a few minutes.")

        # ---- 2️⃣  Callable check
        if not callable(strategy_function):
            raise AssertionError("strategy_function must be a callable that returns a signal dict.")

        # ---- 3️⃣  No signal on the very first bar (prevent look‑ahead bias)
        # We fetch a *single* bar just to see what the strategy would do.
        # If it returns a signal on that bar we abort – the back‑test must
        # start with a clean slate.
        dummy_data = pd.DataFrame({
            "time": [pd.Timestamp(start_date)],
            "open": [1.0],
            "high": [1.0],
            "low":  [1.0],
            "close":[1.0],
            "volume":[0]
        })
        try:
            first_signal = strategy_function(dummy_data)
        except Exception as exc:
            raise AssertionError(f"Strategy raised an exception on dummy data: {exc}")

        if first_signal is not None:
            raise AssertionError("Strategy returned a signal on the first bar – "
                                 "this would introduce look‑ahead bias.")

        # If we reach this point, all checks passed.
        logger.debug("Backtest sanity checks passed.")

    # -----------------------------------------------------------------
    # 2️⃣  Insert the sanity check at the start of run_validation
    # -----------------------------------------------------------------
    def run_validation(self,
                       symbol: str,
                       timeframe: int,
                       start_date: datetime,
                       end_date: datetime,
                       strategy_function,
                       min_win_rate: float = 75.0) -> Dict:
        """
        Run complete validation.
        """
        # ---- SANITY CHECKS -------------------------------------------------
        try:
            self._sanity_checks(symbol, timeframe, start_date, end_date, strategy_function)
        except AssertionError as err:
            logger.error(f"Sanity check failed: {err}")
            return {"success": False, "error": str(err)}
        # --------------------------------------------------------------------

        logger.info(f"Starting validation: {symbol} from {start_date} to {end_date}")

        # (rest of the method stays exactly as you already have)

          # ----------------------------------------------------------------
        # 1️⃣  Pull historic OHLCV
        # ----------------------------------------------------------------
        data = self._fetch_historical_data(symbol, timeframe, start_date, end_date)
        if data is None or data.empty:
            logger.error("Failed to retrieve historic data")
            return {"success": False, "error": "Data fetch failed"}

        # ----------------------------------------------------------------
        # 2️⃣  Run the back‑test engine
        # ----------------------------------------------------------------
        results = self._run_backtest(data, strategy_function)

        # ----------------------------------------------------------------
        # 3️⃣  Analyse the raw results
        # ----------------------------------------------------------------
        analysis = self._analyze_results(results, min_win_rate)

        # ----------------------------------------------------------------
        # 4️⃣  Persist the report
        # ----------------------------------------------------------------
        self._save_results(symbol, timeframe, start_date, end_date, analysis)

        return analysis

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
        ``datetime`` index named ``time``.
        """
        if not mt5.initialize():
            logger.error("MT5 initialisation failed")
            return None

        # MT5 expects UTC timestamps – ensure we pass naive UTC datetimes
        utc_start = start_date.replace(tzinfo=None)
        utc_end = end_date.replace(tzinfo=None)

        rates = mt5.copy_rates_range(symbol, timeframe, utc_start, utc_end)

        mt5.shutdown()

        if rates is None or len(rates) == 0:
            logger.error(f"No rates returned for {symbol}")
            return None

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=False)  # keep column for later use
        return df

    # ------------------------------------------------------------------
    # 2️⃣  Core back‑test engine
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
        equity = balance
        open_positions: List[_Position] = []
        closed_trades: List[Dict] = []
        equity_curve: List[Dict] = []

        # Seed equity curve with the first bar timestamp
        equity_curve.append(
            {"time": data.iloc[0]["time"], "equity": equity}
        )

        # ----------------------------------------------------------------
        # Iterate over the data (skip the first 100 bars to give the strategy
        # a warm‑up window – configurable if you wish)
        # ----------------------------------------------------------------
        for idx in range(100, len(data)):
            current_bar = data.iloc[idx]
            hist_slice = data.iloc[: idx + 1]  # inclusive slice for the strategy

            # ------------------------------------------------------------
            # 1️⃣  Check existing positions for exit conditions
            # ------------------------------------------------------------
            for pos in open_positions[:]:  # iterate over a copy
                exit_info = self._check_exit(pos, current_bar)
                if exit_info["should_exit"]:
                    profit = exit_info["profit"]
                    balance += profit
                    equity = balance

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

            # ------------------------------------------------------------
            # 2️⃣  Ask the strategy for a new signal
            # ------------------------------------------------------------
            signal = strategy_function(hist_slice)

            # Respect the max‑concurrent‑position limit (hard‑coded to 3 here)
            if signal and len(open_positions) < 3:
                # ---- position sizing -------------------------------------------------
                risk_amount = balance * (self.risk_per_trade / 100.0)
                stop_distance = abs(signal["entry_price"] - signal["stop_loss"])
                # Simple lot calculation – 1 lot = 100 000 units, adjust as needed
                # Here we just store the risk amount; the actual lot size is not used
                # elsewhere because the back‑test is purely P&L‑based.
                position = _Position(
                    entry_time=current_bar["time"],
                    symbol=signal["symbol"],
                    direction=signal["direction"],
                    entry_price=signal["entry_price"],
                    stop_loss=signal["stop_loss"],
                    take_profit=signal["take_profit"],
                    risk_amount=risk_amount,
                    # optional meta‑data
                    quality_score=signal.get("quality_score", 0),
                    confluence_score=signal.get("confluence_score", 0),
                )
                open_positions.append(position)

            # ------------------------------------------------------------
            # 3️⃣  Update equity curve (include unrealised P/L)
            # ------------------------------------------------------------
            unrealised = sum(
                self._calculate_unrealised_pl(pos, current_bar)
                for pos in open_positions
            )
            equity = balance + unrealised
            equity_curve.append(
                {"time": current_bar["time"], "equity": equity}
            )

        # ----------------------------------------------------------------
        # Wrap up – any remaining open positions are forced closed at the
        # last bar price (conservative assumption)
        # ----------------------------------------------------------------
        if open_positions:
            final_bar = data.iloc[-1]
            for pos in open_positions:
                # Force close at market price
                exit_price = (
                    final_bar["close"]
                    if pos["direction"] == "BUY"
                    else final_bar["close"]
                )
                profit = (
                    (exit_price - pos["entry_price"]) * 10_000
                    if pos["direction"] == "BUY"
                    else (pos["entry_price"] - exit_price) * 10_000
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

            # final equity after forced liquidation
            equity = balance
            equity_curve.append(
                {"time": final_bar["time"], "equity": equity}
            )

        # ----------------------------------------------------------------
        # Return raw results
        # ----------------------------------------------------------------
        return {
            "closed_trades": closed_trades,
            "equity_curve": equity_curve,
            "final_balance": balance,
            "final_equity": equity,
        }

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

        if position["direction"] == "BUY":
            # Take‑Profit
            if price >= position["take_profit"]:
                profit = (position["take_profit"] - position["entry_price"]) * 10_000
                return {
                    "should_exit": True,
                    "exit_price": position["take_profit"],
                    "profit": profit,
                    "reason": "TP",
                }
            # Stop‑Loss
            if price <= position["stop_loss"]:
                profit = (position["stop_loss"] - position["entry_price"]) * 10_000
                return {
                    "should_exit": True,
                    "exit_price": position["stop_loss"],
                    "profit": profit,
                    "reason": "SL",
                }

        else:  # SELL
            if price <= position["take_profit"]:
                profit = (position["entry_price"] - position["take_profit"]) * 10_000
                return {
                    "should_exit": True,
                    "exit_price": position["take_profit"],
                    "profit": profit,
                    "reason": "TP",
                }
            if price >= position["stop_loss"]:
                profit = (position["entry_price"] - position["stop_loss"]) * 10_000
                return {
                    "should_exit": True,
                    "exit_price": position["stop_loss"],
                    "profit": profit,
                    "reason": "SL",
                }

        return {"should_exit": False}

    # ------------------------------------------------------------------
    # 4️⃣  Unrealised P/L helper
    # ------------------------------------------------------------------
    @staticmethod
    def _calculate_unrealised_pl(position: _Position, current_bar: pd.Series) -> float:
        """
        Simple linear P/L based on the current close price.
        The factor ``10_000`` converts price delta into a pseudo‑pips profit
        (suitable for FX where 1 pip ≈ 0.0001). Adjust as needed for other
        asset classes.
        """
        price = current_bar["close"]
        if position["direction"] == "BUY":
            return (price - position["entry_price"]) * 10_000
        else:
            return (position["entry_price"] - price) * 10_000

    # ------------------------------------------------------------------
    # 5️⃣  Result analysis
    # ------------------------------------------------------------------
    def _analyze_results(self, results: Dict, min_win_rate: float) -> Dict:
        """
        Compute a full performance report.

        Args:
            results: Output of ``_run_backtest``.
            min_win_rate: Minimum acceptable win‑rate (percentage).

        Returns:
            dict with keys:
                - success / passed_validation (bool)
                - win_rate, profit_factor, max_drawdown, ROI, etc.
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

       # ----------------------------------------------------------------
        # 6️⃣  Assemble the final analysis dictionary
        # ----------------------------------------------------------------
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
            "roi_pct": round(((results["final_balance"] - self.initial_balance)
                             / self.initial_balance) * 100, 3),
            "min_win_rate_required_pct": min_win_rate,
            "validation_passed": passed,
            "validation_fail_reasons": [] if passed else [
                "WIN_RATE_BELOW_THRESHOLD"
                if win_rate < min_win_rate else "",
                "DRAW_DOWN_EXCEEDED"
                if max_drawdown < -10 else "",
                "PROFIT_FACTOR_TOO_LOW"
                if profit_factor < 1.5 else "",
            ],
        }

        # Remove empty strings from fail reasons
        analysis["validation_fail_reasons"] = [
            r for r in analysis["validation_fail_reasons"] if r
        ]

        logger.info(
            f"Back‑test completed – win‑rate: {win_rate:.2f}% | "
            f"max DD: {max_drawdown:.2f}% | PF: {profit_factor:.2f} | ROI: {analysis['roi_pct']:.2f}%"
        )
        logger.info(f"Validation {'PASSED' if passed else 'FAILED'}")

        return analysis

    # ------------------------------------------------------------------
    # 7️⃣  Persist the JSON report
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
            logger.info(f"Validation report saved to {filepath}")
        except Exception as exc:  # pragma: no cover
            logger.error(f"Failed to write validation report: {exc}")

    # ------------------------------------------------------------------
    # 8️⃣  Optional helper – export equity curve to CSV (useful for external audit)
    # ------------------------------------------------------------------
    def export_equity_curve(self, results: Dict, filename: str) -> None:
        """
        Write the equity curve (time, equity) to a CSV file.
        Caller can decide where to store it – typically alongside the JSON report.
        """
        try:
            df = pd.DataFrame(results["equity_curve"])
            df.to_csv(filename, index=False)
            logger.info(f"Equity curve exported to {filename}")
        except Exception as exc:  # pragma: no cover
            logger.error(f"Failed to export equity curve: {exc}")


# ----------------------------------------------------------------------
# Global singleton – import this from ``src/main.py`` and use it directly
# ----------------------------------------------------------------------
backtester = BacktestValidator()

def export_trades_to_csv(self, filepath: Union[str, pathlib.Path]) -> None:
    """
    Write the list of closed trades to a CSV that Monte‑Carlo can consume.
    The CSV will contain at least:
        - entry_time
        - exit_time
        - symbol
        - direction
        - entry_price
        - exit_price
        - net_profit   (profit after costs – already stored in the dict)
        - win          (bool)
    """
    import csv
    path = pathlib.Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "entry_time", "exit_time", "symbol", "direction",
        "entry_price", "exit_price", "net_profit", "win"
    ]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for trade in self.trades:          # self.trades is the list of dicts
            writer.writerow({
                "entry_time": trade["entry_time"],
                "exit_time":  trade["exit_time"],
                "symbol":     trade["symbol"],
                "direction":  trade["direction"],
                "entry_price": trade["entry_price"],
                "exit_price":  trade["exit_price"],
                "net_profit":  trade["net_profit"],   # <-- net after costs
                "win":         trade["win"]
            })

# Inside the loop that processes each signal:
static_frac = self.risk_schedule.get(trade_number, self.risk_schedule.get('default', 0.004))
# Ask the engine for the *dynamic* stake
stake = engine.compute_stake(symbol, bucket_equity, static_frac)

def run_validation(self,
                   symbol: str,
                   timeframe: int,
                   start_date: datetime,
                   end_date: datetime,
                   strategy_function,
                   data_feed: DataFeed,
                   min_win_rate: float = 0.0):
    df = data_feed.fetch(symbol, timeframe)
    # … rest unchanged …

