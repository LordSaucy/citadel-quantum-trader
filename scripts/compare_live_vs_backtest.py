#!/usr/bin/env python3
"""
compare_live_vs_backtest.py

Usage (run on the host after the 30‑day paper run finishes):

    python scripts/compare_live_vs_backtest.py \
        --db-url postgresql://citadel:paper_secret@localhost:5432/citadel_paper \
        --start 2024-09-01 \
        --end   2024-09-30
"""

import argparse
import os
import json
from datetime import datetime

import pandas as pd
import psycopg2
from sqlalchemy import create_engine

# -----------------------------------------------------------------
# 1️⃣ Import your existing back‑test engine & strategy
# -----------------------------------------------------------------
# Adjust the import path to match your project layout
from src.backtest_engine import BacktestEngine   # <-- the class you already have
from src.strategy import generate_signals        # <-- your signal generator function

# -----------------------------------------------------------------
# 2️⃣ Helper – fetch live trades from the paper DB
# -----------------------------------------------------------------
def fetch_live_trades(db_url: str, start: datetime, end: datetime) -> pd.DataFrame:
    """
    Pull the raw trade table that the paper bot wrote.
    Expected columns: symbol, direction, volume, entry_price, exit_price,
    entry_time, exit_time, pnl (gross), etc.
    """
    engine = create_engine(db_url)
    sql = """
        SELECT *
        FROM paper_trades          -- <-- adjust table name if different
        WHERE entry_time >= %(start)s
          AND entry_time <= %(end)s
        ORDER BY entry_time;
    """
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={'start': start, 'end': end})
    return df


# -----------------------------------------------------------------
# 3️⃣ Run a back‑test on the *same* slice of market data
# -----------------------------------------------------------------
def run_backtest_on_slice(symbol: str, timeframe: int,
                         start: datetime, end: datetime,
                         strategy_func) -> dict:
    """
    Re‑use the same BacktestEngine you already ship.
    Returns the same analysis dict that the validator produces.
    """
    engine = BacktestEngine(initial_balance=10_000)
    # Load market data from MT5 (or from a cached CSV if you prefer)
    data = engine.load_historical_data(symbol, timeframe, start, end)
    data = engine._add_indicators(data)          # same preprocessing as live
    data = engine.generate_signals(data)         # apply your strategy
    # Use the same internal method that the validator uses to compute metrics
    # (you can expose a thin wrapper in BacktestEngine for this)
    results = engine.run_backtest(symbol, timeframe, start, end)
    # The BacktestEngine already returns a dict with closed_trades, equity_curve, etc.
    # If you need win‑rate/expectancy, call the same analysis routine you use in
    # BacktestValidator._analyze_results (you can import it or duplicate the logic)
    from src.backtest_validator import BacktestValidator
    validator = BacktestValidator()
    analysis = validator._analyze_results(results, min_win_rate=0.0)  # no threshold
    return analysis


# -----------------------------------------------------------------
# 4️⃣ Main – orchestrate the comparison
# -----------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Live vs. back‑test comparison")
    parser.add_argument('--db-url', required=True,
                        help='SQLAlchemy URL to the paper PostgreSQL DB')
    parser.add_argument('--start', required=True,
                        help='Start date (YYYY‑MM‑DD)')
    parser.add_argument('--end', required=True,
                        help='End date (YYYY‑MM‑DD)')
    parser.add_argument('--symbol', default='EURUSD',
                        help='Symbol to back‑test (must match live DB)')
    parser.add_argument('--timeframe', type=int, default=5,
                        help='MT5 timeframe constant (e.g., 5 = M5)')
    args = parser.parse_args()

    start_dt = datetime.strptime(args.start, '%Y-%m-%d')
    end_dt   = datetime.strptime(args.end,   '%Y-%m-%d')

    # -------------------------------------------------------------
    # ① Pull live paper trades
    # -------------------------------------------------------------
    live_df = fetch_live_trades(args.db_url, start_dt, end_dt)
    if live_df.empty:
        print("⚠️ No live trades found for the given window.")
        return

    # -------------------------------------------------------------
    # ② Compute live metrics (gross only – costs are already baked in)
    # -------------------------------------------------------------
    live_gross
