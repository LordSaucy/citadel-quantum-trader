#!/usr/bin/env python3
"""
offline_backtest.py
~~~~~~~~~~~~~~~~~~

A **production‑ready** offline back‑testing harness for the Citadel Quantum Trader
(CQT).  It is deliberately self‑contained – it does **not** require any live
broker connections, external APIs, or a running Docker stack.  The script can be
invoked from the command line or imported as a module.

Features
--------
* Loads OHLCV market data from CSV (or any pandas‑compatible source).
* Executes a user‑provided *strategy* callable that receives a pandas
  ``DataFrame`` and returns a list of ``Trade`` objects.
* Persists every simulated trade with the same schema used by the live
  ``TradeLogger`` (SQLite + JSON metadata) so you can reuse the same analysis
  tools you use in production.
* Generates a concise performance report (win‑rate, profit factor, Sharpe,
  draw‑down, etc.) and writes it to CSV/JSON.
* Fully typed, heavily logged, and equipped with argument validation – ready
  for CI pipelines or scheduled nightly runs.

Usage (CLI)
-----------
>>> python offline_backtest.py \\
...     --data ./sample_data/EURUSD_1h.csv \\
...     --strategy my_strategy.py:MyStrategy \\
...     --start 2024-01-01 \\
...     --end   2024-06-30 \\
...     --output ./backtest_results/run_20240630

The ``--strategy`` argument expects a *module:path.to.Class* (or a plain
function) that conforms to the ``BaseStrategy`` interface described below.

If you prefer to embed the back‑test in another Python program, simply import
the ``run_backtest`` function:

>>> from src.offline_backtest import run_backtest, BacktestConfig
>>> cfg = BacktestConfig(
...     data_path="data/EURUSD_1h.csv",
...     strategy="my_strategy:MyStrategy",
...     start_date=date(2024, 1, 1),
...     end_date=date(2024, 6, 30),
...     output_dir=Path("./bt_output"))
>>> results = run_backtest(cfg)

"""

# ----------------------------------------------------------------------
# Standard library
# ----------------------------------------------------------------------
import argparse
import importlib
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Iterable, List, Mapping, Sequence, Tuple, Union

# ----------------------------------------------------------------------
# Third‑party
# ----------------------------------------------------------------------
import pandas as pd
from pandas import DataFrame

# ----------------------------------------------------------------------
# CQT internal imports (the same logger used by the live engine)
# ----------------------------------------------------------------------
# NOTE: ``src`` is a package (see src/__init__.py).  Importing from the
# package works both when the script is executed as ``python -m src.offline_backtest``
# and when it is run directly from the repository root.
from src.trade_logger import TradeLogger

# ----------------------------------------------------------------------
# Logging configuration (mirrors the live engine)
# ----------------------------------------------------------------------
LOG_DIR = Path("./logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s – %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "offline_backtest.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Trade:
    """
    Simple immutable representation of a simulated trade.
    Mirrors the columns of the live ``trades`` table so that the same
    ``TradeLogger`` can persist it without modification.
    """

    timestamp: datetime
    symbol: str
    direction: str  # "BUY" or "SELL"
    entry_price: float
    exit_price: float
    lot_size: float
    stop_loss: float
    take_profit: float
    profit_loss: float
    profit_loss_pips: float
    entry_quality: int = 0
    confluence_score: int = 0
    mtf_alignment: float = 0.0
    market_regime: str = "UNKNOWN"
    session: str = "UNKNOWN"
    volatility_state: str = "NORMAL"
    rr_ratio: float = 2.0
    stack_level: int = 1
    platform: str = "OFFLINE"
    magic_number: int = 0
    comment: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Mapping[str, Any]:
        """Convert the dataclass to a plain dict ready for DB insertion."""
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        # SQLite stores timestamps as ISO strings
        d["timestamp"] = self.timestamp.isoformat()
        # ``metadata`` must be JSON‑serialisable
        d["metadata"] = json.dumps(self.metadata)
        return d


class BaseStrategy:
    """
    Abstract base class for a back‑testing strategy.

    Sub‑classes must implement ``generate_signals`` which receives a
    ``DataFrame`` containing OHLCV data (indexed by ``datetime``) and returns
    an iterable of :class:`Trade` objects.

    The class can hold any state it needs (e.g. indicator caches) because the
    same instance is reused for the whole back‑test period.
    """

    def generate_signals(self, ohlcv: DataFrame) -> Iterable[Trade]:
        raise NotImplementedError(
            "Strategy classes must implement ``generate_signals``"
        )


# ----------------------------------------------------------------------
# Configuration container
# ----------------------------------------------------------------------


@dataclass
class BacktestConfig:
    """
    All parameters required to run a back‑test.  Using a dataclass makes the
    ``run_backtest`` signature clean and enables easy serialization (e.g. for
    CI pipelines).
    """

    data_path: Path
    """Path to the CSV (or any pandas‑readable) market data file."""

    strategy: str
    """Module path to the strategy, e.g. ``my_strat:MyStrategy``."""

    start_date: date
    """Inclusive start date for the back‑test."""

    end_date: date
    """Inclusive end date for the back‑test."""

    output_dir: Path
    """Directory where SQLite DB, CSV/JSON reports and logs will be written."""

    # Optional overrides – rarely needed but kept for flexibility
    timezone: str = "UTC"
    """Timezone of the input timestamps (pandas will localise if needed)."""

    # Internally populated – not part of the public API
    _strategy_callable: Callable[[DataFrame], Iterable[Trade]] = field(
        init=False, repr=False, compare=False, default=None
    )


# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------


def _load_market_data(path: Path, tz: str = "UTC") -> DataFrame:
    """
    Load OHLCV data from a CSV file.

    Expected columns: ``timestamp, open, high, low, close, volume``.
    The ``timestamp`` column must be parseable by ``pandas.to_datetime``.
    """
    log.info(f"Loading market data from {path}")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    if "timestamp" not in df.columns:
        raise ValueError("CSV must contain a ``timestamp`` column")
    df = df.set_index("timestamp")
    df.index = df.index.tz_localize(tz) if df.index.tz is None else df.index.tz_convert(tz)
    # Ensure required OHLCV columns exist
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}")
    log.info(f"Loaded {len(df)} rows of market data")
    return df


def _import_strategy(spec: str) -> Callable[[DataFrame], Iterable[Trade]]:
    """
    Resolve a ``module:path`` specification to a callable that yields ``Trade``.
    The specification can point to:
    * a subclass of :class:`BaseStrategy` (instantiated with no args)
    * a plain function that receives a ``DataFrame`` and returns an iterable
      of ``Trade`` objects.
    """
    if ":" not in spec:
        raise ValueError(
            "Strategy spec must be in the form ``module:object`` (e.g. "
            "``my_strat:MyStrategy``)"
        )
    module_name, obj_name = spec.split(":", 1)
    log.info(f"Importing strategy {obj_name} from module {module_name}")
    module = importlib.import_module(module_name)

    obj = getattr(module, obj_name, None)
    if obj is None:
        raise AttributeError(f"Object {obj_name!r} not found in module {module_name}")

    # If it's a subclass of BaseStrategy we instantiate it
    if isinstance(obj, type) and issubclass(obj, BaseStrategy):
        instance = obj()
        if not hasattr(instance, "generate_signals"):
            raise TypeError(
                f"{obj_name} does not implement ``generate_signals``"
            )
        return instance.generate_signals

    # Otherwise we expect a plain callable (function)
    if callable(obj):
        return obj

    raise TypeError(
        f"The resolved object {obj_name!r} is neither a BaseStrategy subclass "
        "nor a callable function"
    )


def _filter_date_range(df: DataFrame, start: date, end: date) -> DataFrame:
    """Slice the DataFrame to the inclusive date range."""
    mask = (df.index.date >= start) & (df.index.date <= end)
    filtered = df.loc[mask]
    if filtered.empty:
        raise ValueError(
            f"No data available between {start} and {end}"
        )
    log.info(
        f"Selected {len(filtered)} rows between {start} and {end}"
    )
    return filtered


def _persist_trades(trades: Sequence[Trade], db_path: Path) -> None:
    """
    Persist a list of ``Trade`` objects using the same schema as the live
    engine (SQLite).  The function creates the DB (and tables) if they do not
    exist – this mirrors the behaviour of the live ``TradeLogger``.
    """
    logger = TradeLogger(str(db_path))
    for t in trades:
        # Insert manually because we already have a fully‑populated dict
        cursor = logger.connection.cursor()
        data = t.as_dict()
        columns = ", ".join(data.keys())
        placeholders = ", ".join(["?"] * len(data))
        sql = f"INSERT INTO trades ({columns}) VALUES ({placeholders})"
        cursor.execute(sql, tuple(data.values()))
    logger.connection.commit()
    logger.close()
    log.info(f"Persisted {len(trades)} trades to {db_path}")


def _calc_performance(trades: Sequence[Trade]) -> Mapping[str, Any]:
    """
    Compute a handful of classic performance metrics from a list of trades.
    Returns a plain dict that can be easily serialised to CSV/JSON.
    """
    if not trades:
        return {}

    profits = [t.profit_loss for t in trades]
    pips = [t.profit_loss_pips for t in trades]
    wins = sum(1 for p in profits if p > 0)
    losses = len(profits) - wins
    total_profit = sum(profits)
    total_pips = sum(pips)

    # Profit factor = gross profit / gross loss (absolute)
    gross_profit = sum(p for p in profits if p > 0)
    gross_loss = -sum(p for p in profits if p < 0)  # make positive
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Simple Sharpe (assume risk‑free rate = 0, daily returns approximated)
    returns = pd.Series(profits)
    sharpe = returns.mean() / returns.std(ddof=0) * (252**0.5) if returns.std() != 0 else 0.0

    # Max draw‑down (equity curve)
    equity_curve = pd.Series([0])
    cum = 0.0
    for p in profits:
        cum += p
        equity_curve = equity_curve.append(pd.Series([cum]), ignore_index=True)
    roll_max = equity_curve.cummax()
    drawdowns = roll_max - equity_curve
    max_dd = drawdowns.max()

    return {
        "total_trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate_%": round(wins / len(trades) * 100, 2),
        "total_profit": round(total_profit, 4),
        "total_pips": round(total_pips, 2),
        "profit_factor": round(profit_factor, 4),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown": round(max_dd, 4),
    }


def _write_report(report: Mapping[str, Any], out_dir: Path) -> None:
    """Write the performance dict to CSV and JSON for downstream consumption."""
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "performance_report.csv"
    json_path = out_dir / "performance_report.json"

    # CSV – single‑row file
    pd.DataFrame([report]).to_csv(csv_path, index=False)
    # JSON – pretty printed
    with json.open(json_path, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=4, ensure_ascii=False)

    log.info(f"Wrote performance report to {csv_path} and {json_path}")


# ----------------------------------------------------------------------
# Core orchestration function
# ----------------------------------------------------------------------


def run_backtest(cfg: BacktestConfig) -> Mapping[str, Any]:
    """
    Execute a full back‑test according to ``cfg`` and return the performance
    summary dict.

    The function performs the following steps:

    1. Load market data.
    2. Slice to the requested date range.
    3. Dynamically import the strategy and generate simulated trades.
    4. Persist trades to an SQLite DB (so the same reporting tools used in
       production can be reused).
    5. Compute performance metrics.
    6. Write CSV/JSON reports to ``cfg.output_dir``.

    Parameters
    ----------
    cfg: BacktestConfig
        Fully populated configuration object.

    Returns
    -------
    dict
        Performance metrics (same structure as the JSON report).
    """
    # ------------------------------------------------------------------
    # 0️⃣  Prepare output directory
    # ------------------------------------------------------------------
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1️⃣  Load & filter market data
    # ------------------------------------------------------------------
    market_df = _load_market_data(cfg.data_path, tz=cfg.timezone)
    market_df = _filter_date_range(
        market_df, cfg.start_date, cfg.end_date
    )

    # ------------------------------------------------------------------
    # 2️⃣  Resolve strategy callable (lazy import)
    # ------------------------------------------------------------------
    if cfg._strategy_callable is None:
        cfg._strategy_callable = _import_strategy(cfg.strategy)

    # ------------------------------------------------------------------
    # 3️⃣  Generate simulated trades
    # ------------------------------------------------------------------
    log.info("Running strategy to generate trades …")
    trades_iter = cfg._strategy_callable(market_df)
    trades = list(trades_iter)  # materialise – we need them twice
    log.info(f"Strategy produced {len(trades)} trades")

    # ------------------------------------------------------------------
    # 4️⃣  Persist trades (SQLite – same schema as live engine)
    # ------------------------------------------------------------------
    db_path = cfg.output_dir / "backtest_trades.db"
    _persist_trades(trades, db_path)

    # ------------------------------------------------------------------
    # 5️⃣  Compute performance metrics
    # ------------------------------------------------------------------
    performance = _calc_performance(trades)

    # ------------------------------------------------------------------
    # 6️⃣  Write reports (CSV + JSON)
    # ------------------------------------------------------------------
    _write_report(performance, cfg.output_dir)

    log.info("Back‑test completed successfully")
    return performance


# ----------------------------------------------------------------------
# CLI entry‑point
# ----------------------------------------------------------------------
def _parse_args(argv: Sequence[str] | None = None) -> BacktestConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Offline back‑testing harness for Citadel Quantum Trader. "
            "Loads market data, runs a strategy, stores trades, and "
            "produces a performance report."
        )
    )
    parser.add_argument(
        "--data",
        required=True,
        type=Path,
        help="Path to CSV file containing OHLCV data "
             "(must have columns: timestamp, open, high, low, close, volume).",
    )
    parser.add_argument(
        "--strategy",
        required=True,
        help=(
            "Strategy spec in the form ``module:object``. ``object`` can be "
            "a subclass of ``BaseStrategy`` (will be instantiated) or a plain "
            "function that receives a DataFrame and yields ``Trade`` objects."
        ),
    )
    parser.add_argument(
        "--start",
        required=True,
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        help="Inclusive start date (YYYY‑MM‑DD).",
    )
    parser.add_argument(
        "--end",
        required=True,
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        help="Inclusive end date (YYYY‑MM‑DD).",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Directory where SQLite DB, CSV/JSON reports and logs will be written.",
    )
    parser.add_argument(
        "--timezone",
        default="UTC",
        help="Timezone of the input timestamps (default: UTC).",
    )

    args = parser.parse_args(argv)

    if args.start > args.end:
        parser.error("Start date must be earlier than or equal to end date.")

    # Assemble the configuration dataclass
    cfg = BacktestConfig(
        data_path=args.data,
        strategy=args.strategy,
        start_date=args.start,
        end_date=args.end,
        output_dir=args.output,
        timezone=args.timezone,
    )
    return cfg


def main(argv: Sequence[str] | None = None) -> None:
    """
    CLI entry point – parses arguments, runs the back‑test and exits with
    an appropriate status code.
    """
    try:
        cfg = _parse_args(argv)
        performance = run_backtest(cfg)

        # Pretty‑print a short summary to stdout
        print("\n=== Back‑test performance summary ===")
        for key, value in performance.items():
            print(f"{key:20}: {value}")

        # Successful exit
        sys.exit(0)

    except Exception as exc:  # pragma: no cover
        # Log the full traceback to the file handler and a concise message to stderr
        log.exception("Back‑test failed")
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main() 
