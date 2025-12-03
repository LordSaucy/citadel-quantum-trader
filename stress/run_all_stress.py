#!/usr/bin/env python3
import subprocess
import sys
import datetime

# Define the stress windows you care about
STRESS_CASES = [
    {
        "symbol": "GBPUSD",
        "file": "data/stress/GBPUSD_20220909.parquet",
        "start": "2022-09-09T00:00:00Z",
        "end":   "2022-09-09T23:59:59Z"
    },
    {
        "symbol": "EURJPY",
        "file": "data/stress/EURJPY_20201120.parquet",
        "start": "2020-11-20T00:00:00Z",
        "end":   "2020-11-20T23:59:59Z"
    },
    # add more cases as needed
]

LATENCIES = [0.1, 0.2, 0.5]   # seconds
LIRS = [0.55, 0.65, 0.75]    # mocked LIR values

overall_success = True

for case in STRESS_CASES:
    for lat in LATENCIES:
        for lir in LIRS:
            cmd = [
                sys.executable, "-m", "src.stress.run_stress",
                "--data", case["file"],
                "--symbol", case["symbol"],
                "--timeframe", "5",
                "--start", case["start"],
                "--end", case["end"],
                "--latency", str(lat),
                "--lir", str(lir)
            ]
            print("\n=== RUNNING:", " ".join(cmd), "===\n")
            rc = subprocess.call(cmd)
            if rc != 0:
                overall_success = False
                # you could break early if you want, or continue to collect all failures
print("\n=== STRESS‑SUITE COMPLETE ===")
sys.exit(0 if overall_success else 1)
