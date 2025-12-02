#!/usr/bin/env python3
# ------------------------------------------------------------
# simulate_wr.py
# Monte‑Carlo simulation of a high‑win‑rate, fixed‑fraction bet.
# ------------------------------------------------------------
import random, numpy as np, pandas as pd
# ------------------------------------------------------------------
# USER‑CONFIGURABLE PARAMETERS (override via env vars or CLI later)
# ------------------------------------------------------------------
WR          = float(os.getenv("WR", "0.999"))          # win‑rate
RR          = float(os.getenv("RR", "2.0"))            # reward‑to‑risk
RISK_FRAC   = float(os.getenv("RISK_FRAC", "0.01"))    # % of equity risked each trade
N_TRADES    = int(os.getenv("N_TRADES", "1000"))
N_PATHS     = int(os.getenv("N_PATHS", "10000"))
SEED        = int(os.getenv("SEED", "42"))
PLOT_EXAMPLES = int(os.getenv("PLOT_EXAMPLES", "5"))
# ------------------------------------------------------------------
random.seed(SEED); np.random.seed(SEED)

def run_one_path():
    equity = 1.0
    equities = [equity]
    for _ in range(N_TRADES):
        stake = equity * RISK_FRAC
        pnl = stake * RR if random.random() < WR else -stake
        equity += pnl
        equities.append(equity)
    return equities

# ------------------------------------------------------------------
# Monte‑Carlo run
# ------------------------------------------------------------------
all_curves = []
final_vals = np.empty(N_PATHS)
for i in range(N_PATHS):
    curve = run_one_path()
    all_curves.append(curve)
    final_vals[i] = curve[-1]

# ------------------------------------------------------------------
# Statistics
# ------------------------------------------------------------------
median = np.median(final_vals)
p5, p95 = np.percentile(final_vals, [5, 95])
mean = final_vals.mean()
std  = final_vals.std()
prob_loss = (final_vals < 1.0).mean()
theo_growth_per_trade = WR * (1 + RR * RISK_FRAC) + (1 - WR) * (1 - RISK_FRAC)
theo_total_growth = theo_growth_per_trade ** N_TRADES

print("\n=== Monte‑Carlo Results ===")
print(f"WR={WR:.4%}, RR={RR}, risk={RISK_FRAC:.2%}, trades={N_TRADES:,}")
print(f"Paths={N_PATHS:,}")
print(f"Median equity   : {median:,.4f}×")
print(f"5‑th / 95‑th pct: {p5:,.4f}× – {p95:,.4f}×")
print(f"Mean equity     : {mean:,.4f}×")
print(f"Std‑dev         : {std:,.4f}×")
print(f"P(loss)         : {prob_loss:.6%}")
print(f"Theoretical growth per trade : {theo_growth_per_trade:.6f}")
print(f"Theoretical total growth      : {theo_total_growth:,.4f}×")

# ------------------------------------------------------------------
# Optional plot (requires matplotlib)
# ------------------------------------------------------------------
if PLOT_EXAMPLES > 0:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,6))
    for i in range(min(PLOT_EXAMPLES, N_PATHS)):
        plt.plot(all_curves[i], lw=1, alpha=0.7)
    plt.title(f"{PLOT_EXAMPLES} sample equity curves (WR={WR:.1%}, RR={RR})")
    plt.xlabel("Trade number")
    plt.ylabel("Equity (× initial)")
    plt.grid(True, which='both', ls='--', lw=0.5, alpha=0.7)
    plt.show()
EOF
