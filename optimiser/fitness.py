import pandas as pd
from src.backtest import Backtester   # <- existing back‑test class
import numpy as np

def fitness(params: dict, data_path: str) -> float:
    """
    Returns a scalar fitness value.
    Higher = better.

    params – dictionary of all tunable hyper‑parameters.
    data_path – folder with CSV files (one per symbol).
    """
    # 1️⃣ Load the data (the back‑test engine expects a dict of DataFrames)
    dfs = {}
    for csv_file in Path(data_path).glob("*.csv"):
        symbol = csv_file.stem
        dfs[symbol] = pd.read_csv(csv_file, parse_dates=["timestamp"])

    # 2️⃣ Initialise the back‑tester with the candidate params
    bt = Backtester(
        data=dfs,
        risk_schedule=params["risk_schedule"],   # e.g. [1.0,1.0,0.6,0.5,0.4,...]
        smc_weights=params["smc_weights"],       # list of 7 floats
        rr_target=params["rr_target"],           # usually 5.0
        win_rate_target=params["win_rate_target"],
        max_drawdown=params["max_drawdown"],     # e.g. 0.12 (12 %)
    )

    # 3️⃣ Run the simulation (you can limit to the last 30 days)
    result = bt.run(backtest_window_days=30)

    # 4️⃣ Compute a composite score.
    #    Example: Sharpe * 0.6  +  win_rate * 0.3  –  max_dd * 0.1
    #    You can tweak the weights to reflect business priorities.
    sharpe = result.sharpe
    win_rate = result.win_rate
    max_dd = result.max_drawdown

    score = (0.6 * sharpe) + (0.3 * win_rate) - (0.1 * max_dd)

    # 5️⃣ Penalise infeasible solutions (e.g., risk > 5 % of AUM)
    if result.max_drawdown > params["max_drawdown"]:
        score -= 5.0   # heavy penalty

    return score
