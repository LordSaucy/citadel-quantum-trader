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

def fitness(params: dict, data_path: str) -> float:
    # Load data once (cached globally for speed)
    dfs = load_data(data_path)


    # Split each symbol’s dataframe into train/val (80/20 chronological)
    train_dfs = {}
    val_dfs   = {}
    for sym, df in dfs.items():
        split_idx = int(len(df) * 0.8)
        train_dfs[sym] = df.iloc[:split_idx]
        val_dfs[sym]   = df.iloc[split_idx:]


    # Train‑phase back‑test (used for Sharpe, win‑rate)
    bt_train = Backtester(data=train_dfs, **params)
    res_train = bt_train.run()


    # Validation‑phase back‑test (used for penalty)
    bt_val = Backtester(data=val_dfs, **params)
    res_val = bt_val.run()


    # Composite score: reward train performance, penalise validation drop
    train_score = (0.6 * res_train.sharpe) + (0.3 * res_train.win_rate) - (0.1 * res_train.max_drawdown)
    val_score   = (0.6 * res_val.sharpe)   + (0.3 * res_val.win_rate)   - (0.1 * res_val.max_drawdown)


    # If validation is > 5 % worse than training, apply heavy penalty
    if val_score < train_score * 0.95:
        train_score -= 5.0   # big penalty


    return train_score

scores = []
for holdout_sym in dfs.keys():
    train_symbols = {k: v for k, v in dfs.items() if k != holdout_sym}
    bt = Backtester(data=train_symbols, **params)
    scores.append(bt.run().sharpe)


# Use the mean Sharpe across hold‑outs as the fitness component
mean_sharpe = np.mean(scores)
l2_penalty = 0.01 * np.sum(np.square(params["smc_weights"]))
score = base_score - l2_penalty
