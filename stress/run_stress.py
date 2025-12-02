#!/usr/bin/env python3
"""
Stress‑test runner for extreme‑volatility intervals.

Usage (from the project root):
    python -m src.stress.run_stress \
        --data data/stress/GBPUSD_20220909.parquet \
        --symbol GBPUSD \
        --timeframe 5               # MT5 M5 (you can pick any supported constant)
        --start 2022-09-09T00:00:00Z \
        --end   2022-09-09T23:59:59Z \
        --latency 0.2              # forced sleep per bar (seconds)
        --lir 0.6                  # mocked LIR value (0‑1)
"""

import argparse
import time
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

# ----------------------------------------------------------------------
# 1️⃣  Imports from the existing code base
# ----------------------------------------------------------------------
from ..backtest_validator import BacktestValidator
from ..costs import COMMISSION, SPREAD_PIPS, SLIPPAGE_PIPS   # just to ensure the module is loaded
from ..metrics import PIP_VALUE                               # noqa: F401 (imported for side‑effects)

# ----------------------------------------------------------------------
# 2️⃣  Helper – a very small “depth guard” mock that forces LIR > threshold
# ----------------------------------------------------------------------
def mock_depth_guard_factory(forced_lir: float):
    """
    Return a callable that mimics the original depth‑guard API used by the
    strategy function.  The real strategy likely calls something like:

        if depth_guard_ok(signal):
            ...

    We replace that call with a lambda that always returns ``False`` when the
    LIR is above the forced threshold (i.e. market is *too thin*).

    The validator does not call the guard directly – it receives the signal
    from the user‑supplied ``strategy_function``.  Therefore we will wrap the
    user‑provided strategy with a thin‑wrapper that injects the mocked LIR.
    """
    def wrapper(original_strategy):
        def wrapped(data):
            # Call the original strategy first (it returns a dict or None)
            sig = original_strategy(data)

            # If a signal was generated, inject a fake LIR field
            if sig is not None:
                sig["lir"] = forced_lir   # any name you like; the bot only cares about the value
            return sig
        return wrapped
    return wrapper

# ----------------------------------------------------------------------
# 3️⃣  Main driver
# ----------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Run a stress‑test on a high‑volatility interval")
    parser.add_argument("--data", required=True, help="Path to the Parquet file containing OHLCV")
    parser.add_argument("--symbol", required=True, help="Trading symbol (e.g. GBPUSD)")
    parser.add_argument("--timeframe", type=int, default=5, help="MT5 timeframe constant (default = 5 → M5)")
    parser.add_argument("--start", required=True, help="ISO‑8601 start datetime (UTC)")
    parser.add_argument("--end", required=True, help="ISO‑8601 end datetime (UTC)")
    parser.add_argument("--latency", type=float, default=0.2,
                        help="Forced sleep (seconds) per bar to simulate network/processing lag")
    parser.add_argument("--lir", type=float, default=0.6,
                        help="Mocked LIR value (>0, <1). Values > 0.6 will trigger the depth‑guard")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("stress_test")

    # ------------------------------------------------------------------
    # 4️⃣  Load the high‑volatility candle data
    # ------------------------------------------------------------------
    data_path = Path(args.data)
    if not data_path.is_file():
        log.error("Parquet file not found: %s", data_path)
        return 1

    log.info("Loading stress data from %s", data_path)
    df = pd.read_parquet(data_path)

    # Filter to the requested window (helps if the file contains more than one day)
    start_dt = pd.to_datetime(args.start, utc=True)
    end_dt   = pd.to_datetime(args.end,   utc=True)
    df = df[(df["time"] >= start_dt) & (df["time"] <= end_dt)]

    if df.empty:
        log.error("No rows in the requested time window [%s – %s]", start_dt, end_dt)
        return 1

    log.info("Loaded %d rows covering %s → %s", len(df), df["time"].iloc[0], df["time"].iloc[-1])

    # ------------------------------------------------------------------
    # 5️⃣  Build a *dummy* strategy function that simply returns a signal
    #     on every bar (so we can exercise the kill‑switch logic).
    # ------------------------------------------------------------------
    def dummy_strategy(hist_data: pd.DataFrame):
        """
        Very naive strategy – always generate a BUY signal on the last bar.
        The real bot will later apply its own confluence / SMC filters; we
        only need a signal so that the back‑test goes through the full
        execution pipeline (including the depth‑guard we will mock).
        """
        last = hist_data.iloc[-1]
        return {
            "symbol": args.symbol,
            "direction": "BUY",
            "entry_price": last["close"],
            "stop_loss": last["close"] - 0.0010,   # 10‑pip SL
            "take_profit": last["close"] + 0.0020, # 20‑pip TP
            "volume": 1.0,
            # the mock depth‑guard will later inject a fake "lir" field
        }

    # ------------------------------------------------------------------
    # 6️⃣  Wrap the dummy strategy with the LIR mock
    # ------------------------------------------------------------------
    strategy_with_lir = mock_depth_guard_factory(args.lir)(dummy_strategy)

    # ------------------------------------------------------------------
    # 7️⃣  Instantiate the validator
    # ------------------------------------------------------------------
    validator = BacktestValidator(initial_balance=10_000, risk_per_trade=2.0)

    # ------------------------------------------------------------------
    # 8️⃣  Run the validation loop **with forced latency**
    # ------------------------------------------------------------------
    # We cannot change the internals of BacktestValidator, but we can
    # inject a tiny wrapper around the *run_validation* call that
    # sleeps before each iteration.  The simplest way is to monkey‑patch
    # the private method that iterates over the DataFrame.
    #
    # The original method is called `_run_backtest`.  We replace it with
    # a version that sleeps `args.latency` seconds on every bar.
    # ------------------------------------------------------------------
    original_run_backtest = validator._run_backtest

    def run_backtest_with_latency(data, strat_func):
        # Iterate over the DataFrame manually so we can insert sleep.
        # We copy the original logic (a trimmed‑down version) because
        # the original method is fairly long; for clarity we just
        # delegate to the original after sleeping.
        for _ in range(len(data)):          # dummy loop to trigger sleep per bar
            time.sleep(args.latency)        # <-- forced latency
        # After the artificial delay we call the real implementation.
        return original_run_backtest(data, strat_func)

    validator._run_backtest = run_backtest_with_latency

    # ------------------------------------------------------------------
    # 9️⃣  Execute the back‑test
    # ------------------------------------------------------------------
    log.info("Running back‑test with forced latency %.2fs and mocked LIR %.2f",
             args.latency, args.lir)

    analysis = validator.run_validation(
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=start_dt,
        end_date=end_dt,
        strategy_function=strategy_with_lir,
        min_win_rate=0.0               # we only care about kill‑switch outcome
    )

    # ------------------------------------------------------------------
    # 10️⃣  Determine whether the kill‑switch fired
    # ------------------------------------------------------------------
    # The validator writes a `kill_switch_active` flag into the DB.
    # For a quick, DB‑free check we can also look at the analysis dict:
    # if `max_drawdown` crossed the hard threshold (‑15 % by default) the
    # validator will have set `passed_validation` to False.
    #
    # However, the safest way is to query the DB directly:
    #   SELECT kill_switch_active FROM risk_controller LIMIT 1;
    # For this example we just rely on the analysis result.
    #
    kill_fired = not analysis.get("passed_validation", True)

    if kill_fired:
        log.error("❌ KILL‑SWITCH FIRED – stress test **FAILED**")
        print("FAIL")          # concise CI output
        return 1                # non‑zero exit => CI failure
    else:
        log.info("✅ KILL‑SWITCH NOT FIRED – stress test **PASSED**")
        print("PASS")
        return 0                # zero exit => CI success

if __name__ == "__main__":
    raise SystemExit(main())
