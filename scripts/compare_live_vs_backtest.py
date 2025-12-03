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
    # ② Compute live‑paper metrics (gross & net are already stored)
    # -------------------------------------------------------------
    live_total   = live_df['pnl'].sum()
    live_wins    = live_df[live_df['pnl'] > 0]
    live_losses  = live_df[live_df['pnl'] <= 0]

    live_win_rate = (len(live_wins) / len(live_df)) * 100
    live_avg_win  = live_wins['pnl'].mean() if not live_wins.empty else 0.0
    live_avg_loss = live_losses['pnl'].mean() if not live_losses.empty else 0.0

    # Profit factor (gross)
    gross_profit = live_wins['pnl'].sum()
    gross_loss   = abs(live_losses['pnl'].sum())
    live_profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # -------------------------------------------------------------
    # ③ Run the *same* back‑test on the historical slice
    # -------------------------------------------------------------
    backtest_analysis = run_backtest_on_slice(
        symbol=args.symbol,
        timeframe=args.timeframe,
        start=start_dt,
        end=end_dt,
        strategy_func=generate_signals,   # your strategy function
    )

    # -------------------------------------------------------------
    # ④ Print a concise comparison table
    # -------------------------------------------------------------
    def fmt(v):
        """Pretty‑print numbers (2 decimals, % where appropriate)."""
        if isinstance(v, float):
            return f"{v:,.2f}"
        return str(v)

    print("\n===== LIVE PAPER‑TRADING vs. BACK‑TEST (same window) =====\n")
    print("{:<30} {:>15} {:>15}".format("Metric", "Live Paper", "Back‑Test"))
    print("-" * 62)

    rows = [
        ("Total P&L (net)",          live_total,                     backtest_analysis["net_total_profit"]),
        ("Total P&L (gross)",        live_df["gross_profit"].sum() if "gross_profit" in live_df.columns else live_total,
                                    backtest_analysis["gross_total_profit"]),
        ("Win‑rate (%)",            live_win_rate,                  backtest_analysis["win_rate"]),
        ("Avg. Win (net)",          live_avg_win,                   backtest_analysis["avg_win_net"]),
        ("Avg. Loss (net)",         live_avg_loss,                  backtest_analysis["avg_loss_net"]),
        ("Profit Factor",           live_profit_factor,             backtest_analysis["profit_factor"]),
        ("Max Draw‑down (%)",       live_df["drawdown_pct"].min() if "drawdown_pct" in live_df.columns else float('nan'),
                                    backtest_analysis["max_drawdown"]),
        ("ROI % (net)",             (live_total / 10_000) * 100,   backtest_analysis["roi_pct_net"]),
        ("Avg. Cost / Trade",       live_df["costs"].mean() if "costs" in live_df.columns else 0.0,
                                    backtest_analysis["average_cost_per_trade"]),
    ]

    for name, live_val, back_val in rows:
        print("{:<30} {:>15} {:>15}".format(name, fmt(live_val), fmt(back_val)))

    print("\n✅ Comparison complete.\n")
    # -------------------------------------------------------------
    # ⑤ (Optional) Dump a JSON report for auditors / CI
    # -------------------------------------------------------------
    report = {
        "window": {"symbol": args.symbol, "start": args.start, "end": args.end},
        "live": {
            "total_net": live_total,
            "win_rate": live_win_rate,
            "profit_factor": live_profit_factor,
            "max_drawdown": live_df["drawdown_pct"].min() if "drawdown_pct" in live_df.columns else None,
            "roi_pct": (live_total / 10_000) * 100,
        },
        "backtest": backtest_analysis,
        "delta": {
            "total_net_diff": live_total - backtest_analysis["net_total_profit"],
            "win_rate_diff": live_win_rate - backtest_analysis["win_rate"],
            "profit_factor_diff": live_profit_factor - backtest_analysis["profit_factor"],
            "roi_pct_diff": (live_total / 10_000) * 100 - backtest_analysis["roi_pct_net"],
        },
    }

    out_path = f"comparison_{args.symbol}_{args.start}_{args.end}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"📁 Detailed JSON report written to {out_path}")

    
