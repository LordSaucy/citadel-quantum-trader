#!/usr/bin/env python3
"""
run_walkforward.py

A thin wrapper that performs a classic walk‑forward (rolling‑window)
back‑test:

    1️⃣  Split the historical data chronologically:
        train_len  = 60 days
        test_len   = 30 days
        roll_step  = 10 days   (or any step you prefer)

    2️⃣  Run BacktestValidator on the *train* slice – this is where you would
        calibrate hyper‑parameters (e.g. risk schedule, confluence thresholds,
        optimiser‑derived weights, etc.).  In the minimal version we simply
        call the validator – you can plug any optimisation routine here.

    3️⃣  Run the validator on the *test* slice and record the performance
        metrics (win‑rate, ROI, profit‑factor, max draw‑down, etc.).

    4️⃣  Slide the window forward and repeat until we run out of data.

The results of every roll are written to a CSV file:
    walkforward_results.csv
"""

import argparse
import csv
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, List, Dict

import pandas as pd

# ----------------------------------------------------------------------
# 1️⃣  Imports from your existing code base
# ----------------------------------------------------------------------
from ..backtest_validator import BacktestValidator
# If you have a separate module that builds the signal function, import it.
# Example: from ..strategy import build_signal_function
# For the demo we will use a *dummy* strategy that you replace later.
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# 2️⃣  Dummy strategy – replace with your real one
# ----------------------------------------------------------------------
def dummy_strategy(data: pd.DataFrame) -> Dict:
    """
    Very simple placeholder strategy.
    It returns a BUY signal on the last bar of the supplied data.
    Replace this with your actual signal‑generation function (e.g. the one
    used by the optimiser or the SMC‑confluence engine).
    """
    last = data.iloc[-1]
    return {
        "symbol": "EURUSD",
        "direction": "BUY",
        "entry_price": last["close"],
        "stop_loss": last["close"] - 0.0010,   # 10‑pip SL
        "take_profit": last["close"] + 0.0020,  # 20‑pip TP
        "volume": 1.0
    }


# ----------------------------------------------------------------------
# 3️⃣  Helper – load the full historical dataset once
# ----------------------------------------------------------------------
def load_history(csv_path: Path) -> pd.DataFrame:
    """
    Load a CSV that contains OHLCV data for a single symbol.
    Expected columns: time, open, high, low, close, volume (time in ISO format).
    """
    df = pd.read_csv(csv_path, parse_dates=["time"])
    df.sort_values("time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ----------------------------------------------------------------------
# 4️⃣  Core walk‑forward driver
# ----------------------------------------------------------------------
def walk_forward(
    data: pd.DataFrame,
    train_days: int,
    test_days: int,
    step_days: int,
    validator_factory: Callable[[], BacktestValidator],
    strategy_fn: Callable[[pd.DataFrame], Dict],
    min_win_rate: float = 0.0,
) -> List[Dict]:
    """
    Perform the rolling‑window back‑test.

    Parameters
    ----------
    data : pd.DataFrame
        Full chronological price series.
    train_days, test_days, step_days : int
        Window lengths (in calendar days).  The script works on *row* counts,
        so we convert days → number of rows using the data’s frequency.
    validator_factory : Callable[[], BacktestValidator]
        Factory that returns a fresh BacktestValidator instance for each roll.
    strategy_fn : Callable[[pd.DataFrame], Dict]
        Your signal‑generation function.  It receives the *train* slice when
        calibrating and the *test* slice when evaluating.
    min_win_rate : float, optional
        Minimum win‑rate required for a roll to be considered “passed”.
        Used only for the boolean `passed_validation` field that the
        validator already returns.

    Returns
    -------
    List[Dict]
        One dict per roll containing the roll dates and the analysis dict
        returned by BacktestValidator.
    """
    # ------------------------------------------------------------------
    # Derive row‑counts from the data frequency (assumes regular spacing)
    # ------------------------------------------------------------------
    if data.empty:
        raise ValueError("Historical data is empty")

    # Determine the median delta between rows (in days)
    delta = (data["time"].iloc[1] - data["time"].iloc[0]).total_seconds() / 86400.0
    rows_per_day = int(round(1 / delta))

    train_rows = train_days * rows_per_day
    test_rows = test_days * rows_per_day
    step_rows = step_days * rows_per_day

    results = []

    start_idx = 0
    roll_num = 1

    while start_idx + train_rows + test_rows <= len(data):
        train_slice = data.iloc[start_idx : start_idx + train_rows]
        test_slice = data.iloc[
            start_idx + train_rows : start_idx + train_rows + test_rows
        ]

        # --------------------------------------------------------------
        # 2️⃣  Calibration on the train set (you can tune hyper‑params here)
        # --------------------------------------------------------------
        validator = validator_factory()
        # If you have an optimiser that needs the train data, call it here.
        # Example:
        #   best_params = optimiser.fit(train_slice)
        #   validator.update_parameters(best_params)

        # We still run the validator on the train slice just to keep the
        # same code path – you may skip it if you have no calibration.
        _ = validator.run_validation(
            symbol="EURUSD",
            timeframe=5,                     # MT5 M5 – adjust if needed
            start_date=train_slice["time"].iloc[0],
            end_date=train_slice["time"].iloc[-1],
            strategy_function=strategy_fn,
            min_win_rate=min_win_rate,
        )

        # --------------------------------------------------------------
        # 3️⃣  Evaluation on the test set
        # --------------------------------------------------------------
        analysis = validator.run_validation(
            symbol="EURUSD",
            timeframe=5,
            start_date=test_slice["time"].iloc[0],
            end_date=test_slice["time"].iloc[-1],
            strategy_function=strategy_fn,
            min_win_rate=min_win_rate,
        )

        # --------------------------------------------------------------
        # 4️⃣  Store the roll result
        # --------------------------------------------------------------
        results.append(
            {
                "roll": roll_num,
                "train_start": train_slice["time"].iloc[0].date(),
                "train_end": train_slice["time"].iloc[-1].date(),
                "test_start": test_slice["time"].iloc[0].date(),
                "test_end": test_slice["time"].iloc[-1].date(),
                **analysis,  # unpack all metrics returned by the validator
            }
        )

        # --------------------------------------------------------------
        # 5️⃣  Advance the window
        # --------------------------------------------------------------
        start_idx += step_rows
        roll_num += 1

    return results


# ----------------------------------------------------------------------
# 5️⃣  CLI entry point – convenient for manual runs
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Walk‑forward (rolling‑window) back‑test for CQT"
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="CSV file with full OHLCV history (must contain a 'time' column)",
    )
    parser.add_argument(
        "--train-days",
        type=int,
        default=60,
        help="Length of the training window (calendar days)",
    )
    parser.add_argument(
        "--test-days",
        type=int,
        default=30,
        help="Length of the testing window (calendar days)",
    )
    parser.add_argument(
        "--step-days",
        type=int,
        default=10,
        help="How many days to slide the window forward after each roll",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("walkforward_results.csv"),
        help="Destination CSV file for the aggregated results",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load the full history once
    # ------------------------------------------------------------------
    full_data = load_history(args.data)

    # ------------------------------------------------------------------
    # Factory that returns a fresh validator for each roll
    # ------------------------------------------------------------------
    def validator_factory() -> BacktestValidator:
        # You can tweak the initial balance / risk per trade here if you wish
        return BacktestValidator(initial_balance=10_000, risk_per_trade=2.0)

    # ------------------------------------------------------------------
    # Run the walk‑forward experiment
    # ------------------------------------------------------------------
    rolls = walk_forward(
        data=full_data,
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
        validator_factory=validator_factory,
        strategy_fn=dummy_strategy,   # replace with your real strategy
        min_win_rate=0.0,             # you can raise this if you need a pass/fail flag
    )

    # ------------------------------------------------------------------
    # Write results to CSV (header = dict keys)
    # ------------------------------------------------------------------
    if not rolls:
        print("⚠️ No rolls were generated – check your window sizes vs. data length.")
        return

    fieldnames = list(rolls[0].keys())
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rolls:
            writer.writerow(row)

    print(f"✅ Walk‑forward completed – {len(rolls)} rolls written to {args.out}")


if __name__ == "__main__":
    main()
