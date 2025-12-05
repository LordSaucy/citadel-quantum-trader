#!/usr/bin/env python3
"""
compare_live_vs_backtest.py

Compares live paper-trading results against back-test results on the same
market data window. Useful for validating strategy consistency and
identifying divergence between paper trading and back-test simulations.

Usage (run on the host after the 30‑day paper run finishes):

    python scripts/compare_live_vs_backtest.py \\
        --db-url postgresql://citadel:paper_secret@localhost:5432/citadel_paper \\
        --start 2024-09-01 \\
        --end   2024-09-30

✅ FIXED: Removed unused function parameter "strategy_func"
"""

import argparse
import json
import logging
from datetime import datetime

import pandas as pd
from sqlalchemy import create_engine

# -----------------------------------------------------------------
# Import your existing back‑test engine & validator
# -----------------------------------------------------------------
from src.backtest_engine import BacktestEngine
from src.backtest_validator import BacktestValidator

# -----------------------------------------------------------------
# Logging
# -----------------------------------------------------------------
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------
# 1️⃣ Helper – fetch live trades from the paper DB
# -----------------------------------------------------------------
def fetch_live_trades(db_url: str, start: datetime, end: datetime) -> pd.DataFrame:
    """
    Pull the raw trade table that the paper bot wrote.
    
    Expected columns: symbol, direction, volume, entry_price, exit_price,
    entry_time, exit_time, pnl (gross), etc.
    
    Parameters
    ----------
    db_url : str
        SQLAlchemy database URL
    start : datetime
        Start of time window
    end : datetime
        End of time window
    
    Returns
    -------
    pd.DataFrame
        Live trades from the paper database
    
    Raises
    ------
    Exception
        If database connection fails or query returns no results
    """
    engine = create_engine(db_url)
    sql = """
        SELECT *
        FROM paper_trades
        WHERE entry_time >= %(start)s
          AND entry_time <= %(end)s
        ORDER BY entry_time;
    """
    try:
        with engine.connect() as conn:
            df = pd.read_sql(sql, conn, params={'start': start, 'end': end})
        logger.info(f"Fetched {len(df)} live trades from {start} to {end}")
        return df
    except Exception as exc:
        logger.error(f"Failed to fetch live trades: {exc}")
        raise


# -----------------------------------------------------------------
# 2️⃣ Run a back‑test on the *same* slice of market data
# -----------------------------------------------------------------
def run_backtest_on_slice(
    symbol: str,
    timeframe: int,
    start: datetime,
    end: datetime
) -> dict:
    """
    Re‑use the same BacktestEngine to run back‑test on historical data.
    
    ✅ FIXED: Removed unused parameter "strategy_func".
             The strategy is built into the BacktestEngine already.
    
    Parameters
    ----------
    symbol : str
        Trading pair (e.g., "EURUSD")
    timeframe : int
        MT5 timeframe constant (e.g., 5 = M5)
    start : datetime
        Start of back-test window
    end : datetime
        End of back-test window
    
    Returns
    -------
    dict
        Analysis dict with metrics: net_total_profit, gross_total_profit,
        win_rate, avg_win_net, avg_loss_net, profit_factor, max_drawdown,
        roi_pct_net, average_cost_per_trade
    
    Raises
    ------
    Exception
        If back-test fails or historical data cannot be loaded
    """
    try:
        # Initialize engine with standard capital
        engine = BacktestEngine(initial_balance=10_000)
        
        # Load market data from MT5 (or from a cached CSV if preferred)
        data = engine.load_historical_data(symbol, timeframe, start, end)
        
        # Apply preprocessing (indicators)
        data = engine._add_indicators(data)
        
        # Apply strategy (engine has built-in signal generation)
        data = engine.generate_signals(data)
        
        # Run the back-test
        results = engine.run_backtest(symbol, timeframe, start, end)
        
        # Use BacktestValidator to analyze results
        # (same analysis as the validator uses)
        validator = BacktestValidator()
        analysis = validator._analyze_results(results, min_win_rate=0.0)
        
        logger.info(f"Back-test complete for {symbol} {timeframe} ({start} to {end})")
        return analysis
    except Exception as exc:
        logger.error(f"Back-test failed: {exc}")
        raise


# -----------------------------------------------------------------
# 3️⃣ Helper – format numbers for display
# -----------------------------------------------------------------
def fmt_number(value) -> str:
    """Pretty‑print numbers (2 decimals)."""
    if isinstance(value, float):
        return f"{value:,.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


# -----------------------------------------------------------------
# 4️⃣ Helper – compute live paper metrics
# -----------------------------------------------------------------
def compute_live_metrics(live_df: pd.DataFrame) -> dict:
    """
    Compute standard trading metrics from live paper trades.
    
    Parameters
    ----------
    live_df : pd.DataFrame
        DataFrame of live trades
    
    Returns
    -------
    dict
        Metrics dict with: total_net, win_rate, avg_win, avg_loss,
        profit_factor, max_drawdown, roi_pct
    """
    total_net = live_df['pnl'].sum()
    wins = live_df[live_df['pnl'] > 0]
    losses = live_df[live_df['pnl'] <= 0]
    
    win_rate = (len(wins) / len(live_df) * 100) if len(live_df) > 0 else 0.0
    avg_win = wins['pnl'].mean() if not wins.empty else 0.0
    avg_loss = losses['pnl'].mean() if not losses.empty else 0.0
    
    # Profit factor (gross)
    gross_profit = wins['pnl'].sum() if not wins.empty else 0.0
    gross_loss = abs(losses['pnl'].sum()) if not losses.empty else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Max drawdown (if available in data)
    max_drawdown = (
        live_df['drawdown_pct'].min()
        if 'drawdown_pct' in live_df.columns
        else float('nan')
    )
    
    # ROI
    roi_pct = (total_net / 10_000) * 100
    
    return {
        'total_net': total_net,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'roi_pct': roi_pct,
    }


# -----------------------------------------------------------------
# 5️⃣ Main – orchestrate the comparison
# -----------------------------------------------------------------
def main():
    """
    Main entry point: fetch live trades, run back-test, compare results.
    """
    parser = argparse.ArgumentParser(
        description="Live vs. back-test comparison (same market window)"
    )
    parser.add_argument(
        '--db-url',
        required=True,
        help='SQLAlchemy URL to the paper PostgreSQL DB'
    )
    parser.add_argument(
        '--start',
        required=True,
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end',
        required=True,
        help='End date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--symbol',
        default='EURUSD',
        help='Symbol to back-test (must match live DB)'
    )
    parser.add_argument(
        '--timeframe',
        type=int,
        default=5,
        help='MT5 timeframe constant (e.g., 5 = M5)'
    )
    args = parser.parse_args()

    # Parse dates
    try:
        start_dt = datetime.strptime(args.start, '%Y-%m-%d')
        end_dt = datetime.strptime(args.end, '%Y-%m-%d')
    except ValueError as exc:
        logger.error(f"Invalid date format: {exc}")
        raise

    # --------- ① Pull live paper trades ---------
    try:
        live_df = fetch_live_trades(args.db_url, start_dt, end_dt)
    except Exception as exc:
        logger.error(f"Failed to fetch live trades: {exc}")
        return

    if live_df.empty:
        logger.warning("No live trades found for the given window.")
        return

    # --------- ② Compute live paper metrics ---------
    live_metrics = compute_live_metrics(live_df)

    # --------- ③ Run the back‑test on the historical slice ---------
    try:
        backtest_analysis = run_backtest_on_slice(
            symbol=args.symbol,
            timeframe=args.timeframe,
            start=start_dt,
            end=end_dt
            # ✅ FIXED: Removed unused parameter "strategy_func"
        )
    except Exception as exc:
        logger.error(f"Back-test failed: {exc}")
        return

    # --------- ④ Print a concise comparison table ---------
    print("\n===== LIVE PAPER‑TRADING vs. BACK‑TEST (same window) =====\n")
    print("{:<30} {:>15} {:>15}".format("Metric", "Live Paper", "Back‑Test"))
    print("-" * 62)

    rows = [
        ("Total P&L (net)", live_metrics['total_net'], backtest_analysis["net_total_profit"]),
        ("Win‑rate (%)", live_metrics['win_rate'], backtest_analysis["win_rate"]),
        ("Avg. Win (net)", live_metrics['avg_win'], backtest_analysis["avg_win_net"]),
        ("Avg. Loss (net)", live_metrics['avg_loss'], backtest_analysis["avg_loss_net"]),
        ("Profit Factor", live_metrics['profit_factor'], backtest_analysis["profit_factor"]),
        ("Max Draw‑down (%)", live_metrics['max_drawdown'], backtest_analysis["max_drawdown"]),
        ("ROI % (net)", live_metrics['roi_pct'], backtest_analysis["roi_pct_net"]),
        (
            "Avg. Cost / Trade",
            live_df['costs'].mean() if 'costs' in live_df.columns else 0.0,
            backtest_analysis.get("average_cost_per_trade", 0.0)
        ),
    ]

    for name, live_val, back_val in rows:
        print("{:<30} {:>15} {:>15}".format(name, fmt_number(live_val), fmt_number(back_val)))

    print("\n✅ Comparison complete.\n")

    # --------- ⑤ Dump a JSON report for auditors / CI ---------
    report = {
        "window": {
            "symbol": args.symbol,
            "start": args.start,
            "end": args.end,
            "timeframe": args.timeframe,
        },
        "live": {
            "total_net": live_metrics['total_net'],
            "win_rate": live_metrics['win_rate'],
            "avg_win": live_metrics['avg_win'],
            "avg_loss": live_metrics['avg_loss'],
            "profit_factor": live_metrics['profit_factor'],
            "max_drawdown": float(live_metrics['max_drawdown']) if not pd.isna(live_metrics['max_drawdown']) else None,
            "roi_pct": live_metrics['roi_pct'],
        },
        "backtest": backtest_analysis,
        "delta": {
            "total_net_diff": live_metrics['total_net'] - backtest_analysis["net_total_profit"],
            "win_rate_diff": live_metrics['win_rate'] - backtest_analysis["win_rate"],
            "profit_factor_diff": live_metrics['profit_factor'] - backtest_analysis["profit_factor"],
            "roi_pct_diff": live_metrics['roi_pct'] - backtest_analysis["roi_pct_net"],
        },
    }

    out_path = f"comparison_{args.symbol}_{args.start}_{args.end}.json"
    try:
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Detailed JSON report written to {out_path}")
        print(f"📁 Detailed JSON report written to {out_path}\n")
    except Exception as exc:
        logger.error(f"Could not write report to {out_path}: {exc}")


if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        main()
    except Exception as exc:
        logger.exception("Comparison failed: %s", exc)
        raise
