# baseline_run.py  (run once and keep the JSON)
from src.backtest_validator import BacktestValidator
from datetime import datetime, timedelta
import json, pathlib

def dummy_strategy(df):
    # replace with your real strategy import
    from src.strategy import generate_signal   # <-- your real function
    return generate_signal(df)

validator = BacktestValidator(initial_balance=10_000, risk_per_trade=2.0)
analysis = validator.run_validation(
    symbol="EURUSD",
    timeframe=5,                     # MT5 M5
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 3, 31),
    strategy_function=dummy_strategy,
    min_win_rate=0.0                 # we just want the numbers
)

# store the baseline
baseline_path = pathlib.Path("baseline_expectancy.json")
baseline_path.write_text(json.dumps(analysis, indent=2))
print("Baseline saved →", baseline_path)
