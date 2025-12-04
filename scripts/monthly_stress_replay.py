#!/usr/bin/env python3
# -------------------------------------------------
# monthly_stress_replay.py
# Runs a single “what‑if” stress test on a high‑volatility
# slice of historic data (e.g., GBPUSD 2022‑09‑09).
# -------------------------------------------------
import pandas as pd
from pathlib import Path
import sys
import traceback

# -----------------------------------------------------------------
# Adjust these imports to match your repo layout
# -----------------------------------------------------------------
# The engine lives in src/advanced_execution_engine.py
sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))
from advanced_execution_engine import AdvancedExecutionEngine

# -----------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------
# Path to the Parquet slice you prepared earlier (see PDF 5‑8)
DATA_PATH = Path("/data/stress/GBPUSD_2022-09-09.parquet")

if not DATA_PATH.is_file():
    sys.stderr.write(f"❌ Stress slice not found: {DATA_PATH}\n")
    sys.exit(1)

# -----------------------------------------------------------------
# Load the slice and run the engine once
# -----------------------------------------------------------------
df = pd.read_parquet(DATA_PATH)

engine = AdvancedExecutionEngine()
# The engine already knows how to ingest a DataFrame – if not, we add a thin wrapper:
if not hasattr(engine, "load_market_data"):
    # Very small shim – the real engine expects a stream of ticks,
    # but for a one‑off stress test we can just set an attribute.
    engine.market_data = df

engine.run_once()                     # processes the whole slice
stats = engine.collect_statistics()   # you already have this helper in the codebase

# -----------------------------------------------------------------
# Pretty‑print the results
# -----------------------------------------------------------------
print("\n=== Monthly Stress‑Replay Results ===")
print(f"Win‑rate          : {stats.get('win_rate', 0):.2%}")
print(f"Average R‑R       : {stats.get('avg_rr', 0):.2f}")
print(f"Max draw‑down     : {stats.get('max_dd', 0):.2%}")
print(f"Total trades      : {stats.get('trades', 0)}")
print(f"Total profit (USD): {stats.get('total_profit_usd', 0):,.2f}")
