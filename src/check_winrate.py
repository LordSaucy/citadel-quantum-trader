#!/usr/bin/env python3
"""
check_winrate.py

Loads the most recent `backtest_trades.csv` (produced by BacktestValidator)
and prints a concise win‑rate / loss‑rate summary.
Exits with code 1 if any of the production hard limits are violated.
"""

import sys
import pandas as pd
from pathlib import Path

CSV_PATH = Path("backtest_trades.csv")

if not CSV_PATH.is_file():
    print("❌ CSV file not found – run a back‑test first.", file=sys.stderr)
    sys.exit(1)

df = pd.read_csv(CSV_PATH)

total = len(df)
wins = (df["net_profit"] > 0).sum()
losses = (df["net_profit"] < 0).sum()
break_evens = (df["net_profit"] == 0).sum()
win_rate = wins / total * 100 if total else 0.0

print(f"📊 Back‑test trade summary")
print(f"   Total trades   : {total}")
print(f"   Wins (>0)      : {wins} ({win_rate:.2f} %)")
print(f"   Break‑evens    : {break_evens}")
print(f"   Losses (<0)    : {losses}")

# ---- Hard limits ----------------------------------------------------
hard_fails = []

# Example hard limits – adjust if your config differs
MAX_DRAW_DOWN_PCT = 20.0          # % (negative numbers are bad)
MAX_RISK_PER_TRADE_PCT = 1.5     # %

# Load the analysis JSON that BacktestValidator writes (if present)
analysis_path = Path("backtest_summary.json")
if analysis_path.is_file():
    import json
    analysis = json.load(open(analysis_path))
    drawdown = abs(analysis.get("max_drawdown_pct", 0))
    risk_per_trade = analysis.get("risk_per_trade_pct", 0)

    if drawdown > MAX_DRAW_DOWN_PCT:
        hard_fails.append(f"Draw‑down {drawdown:.2f}% > {MAX_DRAW_DOWN_PCT}%")
    if risk_per_trade > MAX_RISK_PER_TRADE_PCT:
        hard_fails.append(
            f"Risk‑per‑trade {risk_per_trade:.2f}% > {MAX_RISK_PER_TRADE_PCT}%"
        )
else:
    print("⚠️  No backtest_summary.json – skipping draw‑down / risk checks.", file=sys.stderr)

if hard_fails:
    print("\n❌ HARD CONSTRAINT VIOLATIONS:", file=sys.stderr)
    for msg in hard_fails:
        print(f"   • {msg}", file=sys.stderr)
    sys.exit(1)

print("\n✅ All hard constraints satisfied.")
sys.exit(0)
