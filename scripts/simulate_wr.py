#!/usr/bin/env python3
"""
simulate_wr.py – Monte‑Carlo simulation of a high‑win‑rate, fixed‑fraction bet.

Simulates 10,000 equity curves across 1,000 trades to assess the probability
distribution of final wealth under compound risk management.

User‑configurable via environment variables (WR, RR, RISK_FRAC, etc.)

✅ FIXED: Removed EOF statement (had no side effects)
✅ FIXED: Added missing imports (os, json)
"""

import json
import logging
import os
import random
from typing import List

import numpy as np
import pandas as pd

# =====================================================================
# Logging
# =====================================================================
logger = logging.getLogger(__name__)

# =====================================================================
# User‑configurable parameters (override via env vars or CLI)
# =====================================================================
WR          = float(os.getenv("WR", "0.999"))          # win‑rate
RR          = float(os.getenv("RR", "2.0"))            # reward‑to‑risk
RISK_FRAC   = float(os.getenv("RISK_FRAC", "0.01"))    # % of equity risked each trade
N_TRADES    = int(os.getenv("N_TRADES", "1000"))
N_PATHS     = int(os.getenv("N_PATHS", "10000"))
SEED        = int(os.getenv("SEED", "42"))
PLOT_EXAMPLES = int(os.getenv("PLOT_EXAMPLES", "5"))

# =====================================================================
# Initialize random seeds
# =====================================================================
random.seed(SEED)
np.random.seed(SEED)


def run_one_path() -> List[float]:
    """
    Simulate one equity curve (random walk with fixed‑fraction bet sizing).
    
    Returns list of equity values, one per trade (plus initial equity at index 0).
    """
    equity = 1.0
    equities = [equity]
    
    for _ in range(N_TRADES):
        stake = equity * RISK_FRAC
        # Win with probability WR, gain stake * RR; lose with probability (1-WR), lose stake
        pnl = stake * RR if random.random() < WR else -stake
        equity += pnl
        equities.append(equity)
    
    return equities


def main() -> None:
    """Run Monte‑Carlo simulation and display statistics."""
    
    # ===================================================================
    # Monte‑Carlo run – generate all equity curves
    # ===================================================================
    all_curves: List[List[float]] = []
    final_vals = np.empty(N_PATHS)
    
    for i in range(N_PATHS):
        curve = run_one_path()
        all_curves.append(curve)
        final_vals[i] = curve[-1]
    
    # ===================================================================
    # Compute statistics
    # ===================================================================
    median = np.median(final_vals)
    p5, p95 = np.percentile(final_vals, [5, 95])
    mean = final_vals.mean()
    std = final_vals.std()
    prob_loss = (final_vals < 1.0).mean()
    
    # Theoretical growth (Kelly‑like approach)
    theo_growth_per_trade = WR * (1 + RR * RISK_FRAC) + (1 - WR) * (1 - RISK_FRAC)
    theo_total_growth = theo_growth_per_trade ** N_TRADES
    
    # ===================================================================
    # Display results
    # ===================================================================
    print("\n" + "="*50)
    print("Monte‑Carlo Results – Fixed‑Fraction Bet Sizing")
    print("="*50)
    print(f"WR={WR:.4%}, RR={RR}, risk={RISK_FRAC:.2%}, trades={N_TRADES:,}")
    print(f"Number of paths: {N_PATHS:,}")
    print(f"\nMedian equity     : {median:,.4f}× initial")
    print(f"5th / 95th pct   : {p5:,.4f}× – {p95:,.4f}×")
    print(f"Mean equity      : {mean:,.4f}×")
    print(f"Std dev          : {std:,.4f}×")
    print(f"P(final equity < 1.0) : {prob_loss:.6%}")
    print(f"\nTheoretical growth per trade  : {theo_growth_per_trade:.6f}")
    print(f"Theoretical total growth      : {theo_total_growth:,.4f}×")
    print("="*50 + "\n")
    
    # ===================================================================
    # Optional: Save results to JSON
    # ===================================================================
    results = {
        "config": {
            "wr": WR,
            "rr": RR,
            "risk_frac": RISK_FRAC,
            "n_trades": N_TRADES,
            "n_paths": N_PATHS,
            "seed": SEED,
        },
        "statistics": {
            "median": float(median),
            "p5": float(p5),
            "p95": float(p95),
            "mean": float(mean),
            "std": float(std),
            "prob_loss": float(prob_loss),
            "theo_growth_per_trade": float(theo_growth_per_trade),
            "theo_total_growth": float(theo_total_growth),
        },
    }
    
    results_file = "mc_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_file}")
    
    # ===================================================================
    # Optional: Visualize sample equity curves (requires matplotlib)
    # ===================================================================
    if PLOT_EXAMPLES > 0:
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 6))
            num_plots = min(PLOT_EXAMPLES, N_PATHS)
            
            for i in range(num_plots):
                plt.plot(all_curves[i], lw=1, alpha=0.7)
            
            plt.title(f"{num_plots} sample equity curves (WR={WR:.1%}, RR={RR}, Risk={RISK_FRAC:.2%})")
            plt.xlabel("Trade number")
            plt.ylabel("Equity (× initial)")
            plt.grid(True, which='both', ls='--', lw=0.5, alpha=0.7)
            plt.tight_layout()
            plt.savefig("mc_equity_curves.png", dpi=100)
            print("Plot saved to mc_equity_curves.png")
            plt.show()
        except ImportError:
            print("Matplotlib not installed – skipping visualization")


if __name__ == "__main__":
    main()
