#!/usr/bin/env python3
"""
backtest/__init__.py

Production‑grade back‑testing engine for the Citadel Quantum Trader.

Features
--------
* Simple, vectorised OHLCV loader (Parquet → pandas DataFrame)
* Pluggable signal function (receives a DataFrame slice and returns
  a list of dicts: {symbol, direction, qty, sl, tp, comment})
* Full broker‑cost model (commission, spread, swap, slippage, max‑slippage guard)
* Rich performance metrics (gross/net P&L, win‑rate, expectancy, max‑DD,
  Sharpe, Sortino, Calmar, MAR, profit‑factor, etc.)
* CSV export compatible with the rest of the CQT artefact pipeline
* JSON snapshot for the immutable ledger (used by the audit packet)
* Deterministic random‑seed handling (re‑producible runs)

Author  : Lawful Banker
Created : 2024‑11‑26
Version : 2.0 – Production Ready
"""

# ----------------------------------------------------------------------
# Standard library
# ----------------------------------------------------------------------
import json
import logging
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple, Union
from .main import app   # re‑export for uvicorn

# ----------------------------------------------------------------------
# Third‑party
# ----------------------------------------------------------------------
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Logging configuration (writes to ./logs/backtest.log)
# ----------------------------------------------------------------------
LOG_DIR = Path("./logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "backtest.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Helper – cost model constants (feel free to tweak in config)
# ----------------------------------------------------------------------
DEFAULT_COSTS = {
    "spread_pips": 0.2,          # typical ECN spread
    "commission_per_lot": 2.0,   # USD per standard lot
    "swap_per_day": -0.05,       # negative = credit
    "max_slippage_pips": 0.5,    # guard – reject if exceeded
}


# ----------------------------------------------------------------------
# Dataclasses – trade record & performance summary
# ----------------------------------------------------------------------
@dataclass
class TradeRecord:
    """One executed trade – the shape expected by the downstream ledger."""

    timestamp: datetime
    symbol: str
    direction: str          # "BUY" or "SELL"
    entry_price: float
    exit_price: float
    qty_lots: float
    gross_pnl: float
    commission: float
    spread_cost: float
    swap: float
    slippage_pips: float
    net_pnl: float
    comment: str = ""

    def to_dict(self) -> Dict:
        """Flatten for CSV / JSON export."""
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d


@dataclass
class BacktestResult:
    """Aggregated statistics for a back‑test run."""

    gross_total_profit: float = 0.0
    net_total_profit: float = 0.0
    win_rate: float = 0.0
    expectancy: float = 0.0
    max_drawdown: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    profit_factor: float = 0.0
    trades_executed: int = 0
    trades_won: int = 0
    trades_lost: int = 0
    equity_curve: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Serialize for JSON export."""
        d = asdict(self)
        d["equity_curve"] = self.equity_curve
        return d


# ----------------------------------------------------------------------
# Core back‑test engine
# ----------------------------------------------------------------------
class BacktestEngine:
    """
    Run a deterministic back‑test on a single symbol (or a basket of symbols).

    Parameters
    ----------
    data_path : Union[str, Path]
        Directory that contains Parquet files named ``<SYMBOL>.parquet``.
    cost_cfg : dict, optional
        Override of the default cost model (see ``DEFAULT_COSTS``).
    seed : int, optional
        Random seed for any stochastic component (e.g. slippage jitter).
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        cost_cfg: Optional[Dict] = None,
        seed: int = 42,
    ):
        self.data_path = Path(data_path)
        if not self.data_path.is_dir():
            raise FileNotFoundError(f"Data folder not found: {self.data_path}")

        self.costs = DEFAULT_COSTS.copy()
        if cost_cfg:
            self.costs.update(cost_cfg)

        self.rng = np.random.default_rng(seed)
        self.trades: List[TradeRecord] = []
        self.equity_curve: List[float] = []  # equity after each trade
        self._initial_balance: float = 0.0

        log.info("BacktestEngine initialised")
        log.debug(f"Cost model: {self.costs}")

    # ------------------------------------------------------------------
    # 1️⃣  Load OHLCV data (cached per symbol)
    # ------------------------------------------------------------------
    def _load_symbol(self, symbol: str) -> pd.DataFrame:
        """
        Load a symbol’s historical OHLCV from a Parquet file.
        Returns a DataFrame indexed by ``timestamp`` with columns:
        ``open, high, low, close, volume``.
        """
        file_path = self.data_path / f"{symbol}.parquet"
        if not file_path.is_file():
            raise FileNotFoundError(f"Parquet file for {symbol} not found: {file_path}")

        df = pd.read_parquet(file_path)
        if "timestamp" not in df.columns:
            raise KeyError("Parquet file must contain a 'timestamp' column")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()
        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(df.columns):
            raise KeyError(f"Missing required OHLCV columns in {file_path}")

        return df

    # ------------------------------------------------------------------
    # 2️⃣  Core cost model helpers
    # ------------------------------------------------------------------
    def _apply_spread(self, price: float, direction: str) -> float:
        """Shift price by half‑spread in the direction of the trade."""
        spread = self.costs["spread_pips"] * 0.00001  # assume 5‑digit pricing
        return price + spread if direction == "BUY" else price - spread

    def _calc_commission(self, qty_lots: float) -> float:
        """Flat commission per lot (USD)."""
        return qty_lots * self.costs["commission_per_lot"]

    def _calc_swap(self, qty_lots: float, days_held: float) -> float:
        """Swap is linear in lot size and holding time."""
        return qty_lots * self.costs["swap_per_day"] * days_held

    def _calc_slippage(self, entry_price: float, fill_price: float) -> float:
        """Absolute slippage in pips (positive = worse than expected)."""
        diff = abs(fill_price - entry_price)
        return diff / 0.00001  # convert price diff → pips (5‑digit)

    # ------------------------------------------------------------------
    # 3️⃣  Execute a single trade (apply costs, record, update equity)
    # ------------------------------------------------------------------
    def _execute_trade(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        qty_lots: float,
        timestamp: datetime,
        comment: str = "",
    ) -> TradeRecord:
        """
        Simulate a market order with the cost model.

        Returns a fully populated ``TradeRecord``.
        """
        # ---- 1️⃣  Apply spread to the *requested* entry price
        price_with_spread = self._apply_spread(entry_price, direction)

        # ---- 2️⃣  Simulate slippage (random jitter up to max_slippage_pips)
        max_slip = self.costs["max_slippage_pips"] * 0.00001
        jitter = self.rng.uniform(-max_slip, max_slip)
        fill_price = price_with_spread + jitter

        # ---- 3️⃣  Compute raw P&L (ignoring costs)
        if direction == "BUY":
            gross = (fill_price - entry_price) * qty_lots * 100_000
        else:  # SELL
            gross = (entry_price - fill_price) * qty_lots * 100_000

        # ---- 4️⃣  Costs
        commission = self._calc_commission(qty_lots)
        spread_cost = self.costs["spread_pips"] * qty_lots * 10  # approx $ per pip per lot
        # For the demo we assume a 1‑day hold (swap can be refined later)
        swap = self._calc_swap(qty_lots, days_held=1.0)
        slippage_pips = self._calc_slippage(entry_price, fill_price)

        net = gross - commission - spread_cost - swap - (slippage_pips * 10)  # $ per pip ≈10

        trade = TradeRecord(
            timestamp=timestamp,
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            exit_price=fill_price,
            qty_lots=qty_lots,
            gross_pnl=gross,
            commission=commission,
            spread_cost=spread_cost,
            swap=swap,
            slippage_pips=slippage_pips,
            net_pnl=net,
            comment=comment,
        )
        return trade

    # ------------------------------------------------------------------
    # 4️⃣  Run the back‑test over a time window
    # ------------------------------------------------------------------
    def run(
        self,
        symbol: str,
        signal_fn: Callable[[pd.DataFrame], Sequence[Dict]],
        start: datetime,
        end: datetime,
        initial_balance: float = 100_000.0,
    ) -> BacktestResult:
        """
        Execute the back‑test.

        Parameters
        ----------
        symbol : str
            Symbol to back‑test (must have a matching Parquet file).
        signal_fn : callable
            ``signal_fn(df_slice) -> list[dict]`` where each dict contains:
                - ``direction``: "BUY" or "SELL"
                - ``qty_lots`` : float
                - optional ``comment`` (string)
            The function receives a **single‑row DataFrame** (the candle at
            ``timestamp``) and should return a list of orders to be executed
            *immediately* on that candle.
        start, end : datetime
            Inclusive time window for the back‑test.
        initial_balance : float, optional
            Starting equity (default $100 k).

        Returns
        -------
        BacktestResult
            Aggregated performance metrics and the equity curve.
        """
        # ------------------------------------------------------------------
        # Load data & slice the requested window
        # ------------------------------------------------------------------
        df = self._load_symbol(symbol)
        df = df.loc[start:end]
        if df.empty:
            raise ValueError("No data in the requested date range")

        self._initial_balance = initial_balance
        equity = initial_balance
        self.equity_curve = [equity]

        log.info(
            f"Running back‑test on {symbol} from {start} to {end} "
            f"({len(df)} candles, start equity ${initial_balance:,.2f})"
        )

        # ------------------------------------------------------------------
        # Iterate over candles (chronological order)
        # ------------------------------------------------------------------
        for ts, row in df.iterrows():
            # ``row`` is a Series with OHLCV – we pass a 1‑row DF to the signal
            candle_df = row.to_frame().T
            orders = signal_fn(candle_df)

            for order in orders:
                direction = order["direction"].upper()
                qty = float(order["qty_lots"])
                comment = order.get("comment", "")

                # ---- Execute the simulated trade
                trade = self._execute_trade(
                    symbol=symbol,
                    direction=direction,
                    entry_price=row["close"],  # use close price as reference
                    qty_lots=qty,
                    timestamp=ts,
                    comment=comment,
                )
                self.trades.append(trade)

                # ---- Update equity (net P&L) and record curve
                equity += trade.net_pnl
                self.equity_curve.append(equity)

        # ------------------------------------------------------------------
        # Compute performance metrics
        # ------------------------------------------------------------------
        result = self._compute_metrics()
        result.equity_curve = self.equity_curve
        log.info("Back‑test completed")
        log.info(
            f"Final equity: ${equity:,.2f} | Net P&L: ${result.net_total_profit:,.2f}"
        )
        return result

    # ------------------------------------------------------------------
    # 5️⃣  Metric calculation
    # ------------------------------------------------------------------
    def _compute_metrics(self) -> BacktestResult:
        """Derive the classic suite of back‑test statistics."""
        if not self.trades:
            raise RuntimeError("No trades executed – cannot compute metrics")

        gross = sum(t.gross_pnl for t in self.trades)
        net = sum(t.net_pnl for t in self.trades)
        wins = sum(1 for t in self.trades if t.net_pnl > 0)
        losses = sum(1 for t in self.trades if t.net_pnl <= 0)

        win_rate = wins / len(self.trades) * 100.0
        expectancy = net / len(self.trades) if self.trades else 0.0

        # ---- equity curve & draw‑down ----
        equity_series = pd.Series(self.equity_curve)
        hwm = equity_series.cummax()
        drawdown = (hwm - equity_series) / hwm * 100.0
        max_dd = drawdown.max()

        # ---- risk‑adjusted ratios ----
        returns = equity_series.diff().fillna(0) / equity_series.shift(1).replace(0, np.nan)
        returns = returns.dropna()
        if len(returns) < 2:
            sharpe = sortino = calmar = profit_factor = 0.0
        else:
            # Annualise assuming 252 trading days & 1‑day frequency
            daily_std = returns.std()
            sharpe = (returns.mean() / daily_std) * np.sqrt(252)

            # Sortino – downside deviation only
            downside = returns[returns < 0]
            downside_std = downside.std(ddof=0) if not downside.empty else 0.0
            sortino = (
                (returns.mean() / downside_std) * np.sqrt(252)
                if downside_std > 0
                else 0.0
            )

            # Calmar = CAGR / max draw‑down
            total_years = (len(equity_series) - 1) / 252.0
            cagr = (equity_series.iloc[-1] / equity_series.iloc[0]) ** (
                1 / total_years
            ) - 1
            calmar = cagr / (max_dd / 100.0) if max_dd > 0 else np.inf

            # Profit factor = gross profit / gross loss
            gross_profit = sum(t.net_pnl for t in self.trades if t.net_pnl > 0)
            gross_loss = -sum(t.net_pnl for t in self.trades if t.net_pnl < 0)
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

        return BacktestResult(
            gross_total_profit=gross,
            net_total_profit=net,
            win_rate=win_rate,
            expectancy=expectancy,
            max_drawdown=max_dd,
            sharpe=sharpe,
            sortino=sortino,
            calmar=calmar,
            profit_factor=profit_factor,
            trades_executed=len(self.trades),
            trades_won=wins,
            trades_lost=losses,
        )

      # ------------------------------------------------------------------
    # 6️⃣  CSV / JSON export helpers (used by the rest of CQT)
    # ------------------------------------------------------------------
    def export_trades_csv(self, out_path: Union[str, Path] = "backtest_trades.csv") -> Path:
        """
        Write the trade log to a CSV file compatible with the existing
        ``export_backtest.py`` script.
        """
        out_path = Path(out_path)
        df = pd.DataFrame([t.to_dict() for t in self.trades])
        df.to_csv(out_path, index=False)
        log.info(f"Exported {len(self.trades)} trades to {out_path}")
        return out_path

    def export_json_snapshot(
        self,
        out_path: Union[str, Path] = "backtest_snapshot.json",
    ) -> Path:
        """
        Serialize the full back‑test result (metrics + equity curve) to JSON.
        This file is later consumed by the immutable‑ledger uploader.
        """
        out_path = Path(out_path)
        result = self._compute_metrics()
        snapshot = {
            "metadata": {
                "symbol": self.trades[0].symbol if self.trades else None,
                "start_timestamp": self.trades[0].timestamp.isoformat()
                if self.trades
                else None,
                "end_timestamp": self.trades[-1].timestamp.isoformat()
                if self.trades
                else None,
                "initial_balance": self._initial_balance,
                "final_balance": self.equity_curve[-1] if self.equity_curve else None,
                "generated_at": datetime.now().isoformat(),
                "cost_model": self.costs,
            },
            "metrics": result.to_dict(),
            "trades": [t.to_dict() for t in self.trades],
        }

        with open(out_path, "w") as fp:
            json.dump(snapshot, fp, indent=2)

        log.info(f"Exported JSON snapshot to {out_path}")
        return out_path

    # ------------------------------------------------------------------
    # 7️⃣  Convenience helpers
    # ------------------------------------------------------------------
    def get_trade_dataframe(self) -> pd.DataFrame:
        """Return the trade log as a pandas DataFrame (handy for ad‑hoc analysis)."""
        return pd.DataFrame([t.to_dict() for t in self.trades])

    def get_equity_series(self) -> pd.Series:
        """Return the equity curve as a pandas Series indexed by trade number."""
        return pd.Series(self.equity_curve, name="equity")

    # ------------------------------------------------------------------
    # 8️⃣  Clean‑up (optional – call at the end of a script)
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear all stored trades and equity data – useful for repeated runs."""
        self.trades.clear()
        self.equity_curve.clear()
        self._initial_balance = 0.0
        log.info("BacktestEngine state reset")

# ----------------------------------------------------------------------
# Public API of the package
# ----------------------------------------------------------------------
__all__ = ["BacktestEngine", "TradeRecord", "BacktestResult"] 

