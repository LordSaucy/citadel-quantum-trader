# src/monte_carlo/run_mc.py
"""
Monte‑Carlo bootstrap driver.

Usage
-----
python -m monte_carlo.run_mc \
    --trades backtest_trades.csv \
    --iterations 10000 \
    --output mc_summary.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import List, Dict

import pandas as pd

# Import the bootstrap engine we just wrote
from ._bootstrap import run_bootstrap


def _load_trades(csv_path: pathlib.Path) -> pd.DataFrame:
    """
    Load the CSV produced by ``BacktestValidator``.
    Expected columns (at minimum):
        - net_profit   (float)   – profit after commission/spread/slippage
        - win          (bool)    – True if net_profit > 0
    If you exported a different column name, adjust the code accordingly.
    """
    df = pd.read_csv(csv_path)

    # The validator we added earlier writes a column called ``net_profit``.
    # If you kept the original name ``profit`` you can rename it here:
    if "profit" in df.columns and "net_profit" not in df.columns:
        df = df.rename(columns={"profit": "net_profit"})

    if "net_profit" not in df.columns:
        sys.stderr.write(
            f"ERROR: CSV does not contain a 'net_profit' column. "
            f"Columns present: {list(df.columns)}\n"
        )
        sys.exit(1)

    return df


def _aggregate_statistics(samples: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Given a list of per‑iteration dicts, compute the *mean* and the
    *95 % confidence interval* (2.5 % / 97.5 %) for each metric.
    """
    import numpy as np

    # Convert list of dicts → DataFrame for easy aggregation
    df = pd.DataFrame(samples)

    summary = {}
    for col in df.columns:
        vals = df[col].values
        mean = np.mean(vals)
        lo = np.percentile(vals, 2.5)
        hi = np.percentile(vals, 97.5)
        summary[col] = {"mean": float(mean), "ci_2.5%": float(lo), "ci_97.5%": float(hi)}
    return summary


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Monte‑Carlo bootstrap analysis of back‑test trade results."
    )
    parser.add_argument(
        "--trades",
        type=pathlib.Path,
        required=True,
        help="CSV file produced by BacktestValidator (must contain a 'net_profit' column).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10_000,
        help="Number of bootstrap samples (default: 10 000).",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("mc_summary.json"),
        help="Path where the JSON summary will be written (default: mc_summary.json).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (optional).",
    )

    args = parser.parse_args(argv)

    # -----------------------------------------------------------------
    # 1️⃣ Load trades → extract net P&L series
    # -----------------------------------------------------------------
    df = _load_trades(args.trades)
    pnl_series = df["net_profit"].tolist()

    if not pnl_series:
        sys.stderr.write("ERROR: No trades found in the CSV.\n")
        return 1

    # -----------------------------------------------------------------
    # 2️⃣ Run the bootstrap engine
    # -----------------------------------------------------------------
    samples = run_bootstrap(pnl_series, iterations=args.iterations, random_seed=args.seed)

    # -----------------------------------------------------------------
    # 3️⃣ Aggregate per‑metric statistics (mean + 95 % CI)
    # -----------------------------------------------------------------
    summary = _aggregate_statistics(samples)

    # -----------------------------------------------------------------
    # 4️⃣ Write JSON output
    # -----------------------------------------------------------------
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Monte‑Carlo summary written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
