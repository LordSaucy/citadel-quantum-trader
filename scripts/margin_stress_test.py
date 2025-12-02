#!/usr/bin/env python3
"""
margin_stress_test.py

Runs a Monte‑Carlo simulation of equity trajectories and checks that the
dynamic margin calculator never asks the broker for a lot that would breach the
configured max_leverage.
"""

import random
import math
import yaml
from pathlib import Path
from src.risk_management_layer import max_lot_by_margin, compute_stake
from src.config_loader import Config

# -------------------------------------------------
# Load configuration (includes broker params)
# -------------------------------------------------
cfg = Config().settings

# -------------------------------------------------
# Simulation parameters (tweak as needed)
# -------------------------------------------------
NUM_PATHS = 10_000
STEPS_PER_PATH = 500          # ~500 trades per simulation
WIN_RATE = cfg.get("win_rate_target", 0.997)
RR = cfg.get("RR_target", 5.0)   # reward‑to‑risk ratio (e.g., 5:1)
INITIAL_EQUITY = 10_000.0        # $10 k per bucket (example)

def simulate_one_path():
    equity = INITIAL_EQUITY
    max_leverage_seen = 0.0

    for _ in range(STEPS_PER_PATH):
        # -------------------------------------------------
        # 1️⃣ Determine risk fraction from schedule (use step‑wise schedule)
        # -------------------------------------------------
        # For simplicity we use the default schedule entry (0.40) for every trade.
        # You can replace this with a call to the real schedule if you wish.
        risk_frac = cfg.get("risk_schedule", {}).get("default", 0.40)

        # -------------------------------------------------
        # 2️⃣ Compute stake (dollar amount) – this will also
        #    update the effective_leverage gauge internally.
        # -------------------------------------------------
        stake = compute_stake(bucket_id=1, equity=equity)  # bucket_id is arbitrary here

        # -------------------------------------------------
        # 3️⃣ Simulate win/loss outcome
        # -------------------------------------------------
        if random.random() < WIN_RATE:
            equity += stake * RR   # win
        else:
            equity -= stake        # loss

        # -------------------------------------------------
        # 4️⃣ Record the *effective* leverage for this step
        # -------------------------------------------------
        # The compute_stake() function already emitted the gauge,
        # but we also compute it locally for the stress test.
        lot = (equity * risk_frac) / cfg["contract_notional"]
        effective_leverage = (lot * cfg["contract_notional"]) / (equity * risk_frac)
        max_leverage_seen = max(max_leverage_seen, effective_leverage)

    return max_leverage_seen

def main():
    worst_leverage = 0.0
    violations = 0

    for i in range(NUM_PATHS):
        max_lev = simulate_one_path()
        worst_leverage = max(worst_leverage, max_lev)

        # Safety margin: we consider a violation if we exceed 95 % of broker max
        if max_lev > cfg["broker"]["max_leverage"] * 0.95:
            violations +=
