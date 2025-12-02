import numpy as np
import json
import argparse

def simulate_drop(base_wr, base_rr, drop_wr, weeks=1, trades_per_week=30, risk=0.01):
    """Return equity series with a low‑win‑rate segment."""
    equity = [1.0]  # start with 1.0 (100 %)
    for week in range(52):
        wr = drop_wr if week < weeks else base_wr
        for _ in range(trades_per_week):
            stake = equity[-1] * risk
            if np.random.rand() < wr:
                equity.append(equity[-1] + stake * base_rr)
            else:
                equity.append(equity[-1] - stake)
    return equity

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base-wr", type=float, default=0.938)
    p.add_argument("--base-rr", type=float, default=5.8)
    p.add_argument("--drop-wr", type=float, default=0.70)
    p.add_argument("--weeks", type=int, default=1)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    eq = simulate_drop(args.base_wr, args.base_rr,
                       args.drop_wr, args.weeks)
    # Compute max draw‑down
    peak = np.maximum.accumulate(eq)
    dd = 1 - np.min(eq / peak)
    result = {"final_equity": eq[-1],
              "max_drawdown": dd,
              "equity_series": eq}
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
